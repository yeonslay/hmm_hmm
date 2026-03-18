import glob
import importlib.util
import json
import os
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.utils import import_utils

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


Qwen3VLForConditionalGeneration = getattr(transformers, "Qwen3VLForConditionalGeneration")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROCESSOR_ID = MODEL_ID
VIDEO_DIR = str(PROJECT_ROOT / "raw/accident/videos")
METADATA_CSV = str(PROJECT_ROOT / "raw/accident/test_metadata.csv")
OUTPUT_CSV = str(BASE_DIR / "submission.csv")
DEBUG_CSV = str(BASE_DIR / "debug_results.csv")

# ── VRAM 최적화 (24GB GPU, 4-bit 양자화 모델 기준 추론 메모리 절감) ──
# 전략: fps=2로 토큰 절약 → 절약분을 해상도에 투자 (유형 분류 & 좌표 정밀도 향상)
# 평가 지표 분석: S_type(0/1) 틀리면 전체 0, S_time σ=1초 → fps=2면 충분, S_spatial σ=0.1

MAX_FRAMES = 64          # 30초 × 2fps = 60프레임으로 cap 안 걸림
TARGET_FPS = 2.0         # σ_t=1초에 fps=2 충분, 토큰 절약
MIN_FRAMES = 8           # 매우 짧은 영상용
MAX_PIXELS = 512 * 768   # 고해상도: 유형 판별 & 좌표 정밀도 핵심
MIN_PIXELS = 28 * 28     # 프레임당 최소 픽셀
MAX_NEW_TOKENS = 220
INFERENCE_BATCH_SIZE = 2

VALID_TYPES = ["rear-end", "head-on", "sideswipe", "t-bone", "single"]


def to_float_or(value: Any, default: float) -> float:
    return float(default if value is None else value)


def to_int_or(value: Any, default: int) -> int:
    return int(default if value is None else value)

SSYSTEM_PROMPT = (
    "You are a traffic crash analyst working on a benchmark where every video contains exactly one traffic crash. "
    "Your task is to localize the first impact moment, estimate the accident-location center from vehicle bounding boxes, and classify the crash type. "
    "Never answer that no crash is visible. If evidence is partial, make the best grounded estimate from the frames shown."
)

USER_PROMPT_TEMPLATE = """The following {n_frames} frames are sampled from a {duration:.1f}-second dashcam video.
Each frame is labeled with its exact timestamp [t=X.Xs].

Return exactly one JSON object with these keys:
{{
  "accident_time": <seconds from start where first physical impact occurs, between 0.0 and {duration:.1f}>,
  "center_x": <normalized x in [0,1] of the accident-location center>,
  "center_y": <normalized y in [0,1] of the accident-location center>,
  "type": <"rear-end"|"head-on"|"sideswipe"|"t-bone"|"single">,
  "confidence": <float in [0,1]>,
  "reasoning": <one short sentence>
}}

Crash type definitions:
- single   : vehicle vs non-vehicle object (wall, guardrail, pole, tree, barrier, curb)
- rear-end : same direction — front of A hits rear of B
- head-on  : opposite direction — front of A hits front of B
- t-bone   : perpendicular (~90°) — front of A hits side of B
- sideswipe: parallel — side of A hits side of B (same or opposite direction)

Rules:
- Every video contains one crash, so do not say no crash, unknown, or none.
- Use the first physical impact, not the aftermath.
- accident_time MUST match one of the [t=X.Xs] timestamps shown, or interpolate between two adjacent ones.
- For rear-end, head-on, t-bone, and sideswipe crashes, first identify the two vehicles involved in the first impact, imagine a bounding box around each vehicle at the impact moment, and set (center_x, center_y) to the midpoint between the centers of those two bounding boxes.
- For single crashes, identify the crashing vehicle, imagine a bounding box around that vehicle at the first impact moment, and set (center_x, center_y) to the center of that bounding box.
- Use normalized coordinates in [0,1], where (0,0) is the top-left and (1,1) is the bottom-right of the frame.
- Do not use the frame center unless it coincides with the bounding-box-based accident-location center.
- If the exact boxes are unclear, estimate the most likely vehicle boxes from the visible frames and use their centers consistently.
- Choose exactly one crash type from the allowed list.
- Output JSON only, with no markdown fences or extra text.
"""


@dataclass
class Prediction:
    path: str
    accident_time: float
    center_x: float
    center_y: float
    type: str
    confidence: float
    reasoning: str
    raw: str
    method: str
    fallback_used: bool
    issues: str


# ═══════════════════════════════════════════════════════
#  메타데이터 로드
# ═══════════════════════════════════════════════════════

def load_metadata(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """test_metadata.csv를 로드하여 {path: {duration, no_frames, height, width, ...}} 딕셔너리로 반환."""
    if not os.path.exists(csv_path):
        print(f"[WARN] Metadata CSV not found: {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    meta = {}
    for _, row in df.iterrows():
        duration = row.get("duration")
        no_frames = row.get("no_frames")
        height = row.get("height")
        width = row.get("width")
        quality = row.get("quality")
        meta[row["path"]] = {
            "duration": to_float_or(duration, 10.0),
            "no_frames": to_int_or(no_frames, 0),
            "height": to_int_or(height, 720),
            "width": to_int_or(width, 1280),
            "quality": str(quality if quality is not None else "Fine"),
        }
    return meta


def compute_video_fps(duration: float, no_frames: int, height: int, width: int) -> float:
    """영상 메타데이터 기반 최적 FPS 계산. 단순하고 일관된 전략."""
    if duration <= 0:
        return TARGET_FPS

    fps = TARGET_FPS

    # 긴 영상에서 MAX_FRAMES 초과 시 fps 자동 하향
    if duration * fps > MAX_FRAMES:
        fps = MAX_FRAMES / duration

    # 짧은 영상에서 최소 프레임 수 확보
    if duration * fps < MIN_FRAMES:
        fps = min(MIN_FRAMES / duration, 4.0)

    fps = max(0.5, min(4.0, fps))
    return round(fps, 2)


# ═══════════════════════════════════════════════════════
#  모델 로드 & 추론
# ═══════════════════════════════════════════════════════

def load_model() -> Tuple[Any, Any]:
    if getattr(import_utils, "is_flash_attn_3_available", lambda: False)():
        attn_implementation = "flash_attention_3"
    elif import_utils.is_flash_attn_2_available():
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[INFO] Loading {MODEL_ID} in 4-bit ...")
    print(f"[INFO] Attention backend: {attn_implementation}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
    processor.tokenizer.padding_side = "left"
    model.eval()
    print("[INFO] Model ready.\n")
    return model, processor


def sample_frames_with_timestamps(video_path: str, fps: float, duration: float) -> List[Tuple[Image.Image, float]]:
    """cv2로 프레임 추출, (PIL Image, timestamp_seconds) 리스트 반환."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_duration = total_frames_actual / original_fps if original_fps > 0 else duration

    # 샘플링할 프레임 수 계산
    n_frames = max(MIN_FRAMES, min(MAX_FRAMES, int(duration * fps)))
    # 균등 간격으로 프레임 인덱스 생성
    frame_indices = np.linspace(0, total_frames_actual - 1, n_frames, dtype=int)

    results = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        # 해상도 축소 (VRAM 절약)
        max_side = 640
        w, h = pil_img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        timestamp = (idx / max(total_frames_actual - 1, 1)) * duration
        results.append((pil_img, round(timestamp, 1)))

    cap.release()
    return results


def build_frame_messages(video_path: str, fps: float, duration: float) -> Tuple[List[Dict[str, Any]], int]:
    """이미지 프레임 + 타임스탬프 라벨로 메시지 구성. (messages, n_frames) 반환."""
    frames_with_ts = sample_frames_with_timestamps(video_path, fps, duration)
    n_frames = len(frames_with_ts)

    user_prompt = USER_PROMPT_TEMPLATE.format(duration=duration, fps=fps, n_frames=n_frames)

    # 각 프레임을 [t=X.Xs] 라벨과 함께 interleave
    content_parts: List[Dict[str, Any]] = []
    for pil_img, ts in frames_with_ts:
        content_parts.append({"type": "text", "text": f"[t={ts}s]"})
        content_parts.append({"type": "image", "image": pil_img, "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS})

    content_parts.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]
    return messages, n_frames


def generate_raw_batch(
    model: Any,
    processor: Any,
    messages_batch: List[List[Dict[str, Any]]],
) -> List[str]:
    text_inputs = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    vision_result = process_vision_info(messages_batch)
    image_inputs = vision_result[0]
    video_inputs = vision_result[1]
    processor_kwargs: Dict[str, Any] = {
        "text": text_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs is not None:
        processor_kwargs["images"] = image_inputs
    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs

    inputs = processor(**processor_kwargs).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0,
            top_p=None,
        )

    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    return [
        text.strip()
        for text in processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        )
    ]


def generate_raw(model: Any, processor: Any, messages: List[Dict[str, Any]]) -> str:
    return generate_raw_batch(model, processor, [messages])[0]


# ═══════════════════════════════════════════════════════
#  출력 파싱 & 후처리
# ═══════════════════════════════════════════════════════

def extract_json(clean: str) -> Dict[str, Any]:
    match = re.search(r"```json\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*?\})", clean, re.DOTALL)
    if not match:
        raise ValueError("json not found")
    return json.loads(match.group(1))


def normalize_type(value: Any) -> str:
    if not isinstance(value, str):
        return "single"
    text = value.strip().lower()
    aliases = {
        "rear end": "rear-end",
        "rear_end": "rear-end",
        "rear-ended": "rear-end",
        "head on": "head-on",
        "head_on": "head-on",
        "side swipe": "sideswipe",
        "side-swipe": "sideswipe",
        "tbone": "t-bone",
        "t bone": "t-bone",
    }
    text = aliases.get(text, text)
    return text if text in VALID_TYPES else "single"


def coerce_time(value: Any, duration: float, sampled_frames: int) -> float:
    """모델 출력의 시간 값을 검증 & 클램프.
    
    이미지 모드에서는 모델이 직접 초 단위로 답하므로, 범위만 검증.
    """
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return duration / 2

    # 범위 클램프
    return max(0.0, min(duration, candidate))


def parse_output(raw: str, video_path: str, duration: float, method: str, sampled_frames: int) -> Prediction:
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        data = extract_json(clean)
    except (ValueError, json.JSONDecodeError):
        data = {}

    reasoning = str(data.get("reasoning", clean[:160])).strip()
    accident_time = coerce_time(data.get("accident_time"), duration, sampled_frames)
    center_x = float(data.get("center_x", 0.5) or 0.5)
    center_y = float(data.get("center_y", 0.5) or 0.5)
    confidence = float(data.get("confidence", 0.0) or 0.0)
    collision_type = normalize_type(data.get("type", "single"))

    accident_time = max(0.0, min(duration, accident_time))
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    confidence = max(0.0, min(1.0, confidence))

    issues: List[str] = []
    lowered = clean.lower()
    if not data:
        issues.append("parse_failed")
    if "no crash" in lowered or "no accident" in lowered:
        issues.append("forbidden_no_crash")
    if "not detectable" in lowered or "no physical impact" in lowered:
        issues.append("forbidden_no_crash")
    if accident_time <= 0.01:
        issues.append("time_zero")
    if abs(center_x - 0.5) < 1e-6 and abs(center_y - 0.5) < 1e-6:
        issues.append("center_default")
    if confidence < 0.2:
        issues.append("low_confidence")

    if "forbidden_no_crash" in issues:
        accident_time = duration / 2
        issues = [issue for issue in issues if issue != "time_zero"]

    relative_base = Path(VIDEO_DIR).parent.resolve()
    resolved_path = Path(video_path).resolve()
    try:
        relative_path = resolved_path.relative_to(relative_base)
    except ValueError:
        relative_path = resolved_path.name

    return Prediction(
        path=str(relative_path),
        accident_time=accident_time,
        center_x=center_x,
        center_y=center_y,
        type=collision_type,
        confidence=confidence,
        reasoning=reasoning,
        raw=clean,
        method=method,
        fallback_used=False,
        issues=",".join(issues),
    )


# ═══════════════════════════════════════════════════════
#  메인 추론 파이프라인
# ═══════════════════════════════════════════════════════

def infer_video(
    model: Any,
    processor: Any,
    video_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """단일 비디오에 대해 이미지 프레임 + 타임스탬프 모드로 추론."""
    # 메타데이터에서 정보 가져오기 (없으면 기본값)
    if meta:
        duration = meta["duration"]
        no_frames = meta["no_frames"]
        height = meta["height"]
        width = meta["width"]
    else:
        duration = 10.0
        no_frames = 0
        height = 720
        width = 1280

    # 영상별 최적 FPS 계산
    fps = compute_video_fps(duration, no_frames, height, width)

    try:
        messages, sampled_frames = build_frame_messages(video_path, fps, duration)
        raw = generate_raw(model, processor, messages)
        prediction = parse_output(raw, video_path, duration, "frames", sampled_frames)
    except Exception as exc:
        prediction = Prediction(
            path=str(Path(video_path).name),
            accident_time=duration / 2,
            center_x=0.5,
            center_y=0.5,
            type="single",
            confidence=0.0,
            reasoning=f"frame_error: {str(exc)[:120]}",
            raw="",
            method="frame_error",
            fallback_used=False,
            issues="frame_error,parse_failed",
        )

    return asdict(prediction)


def infer_videos_batch(
    model: Any,
    processor: Any,
    video_paths: List[str],
    metas: List[Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    messages_batch: List[List[Dict[str, Any]]] = []

    try:
        for video_path, meta in zip(video_paths, metas):
            if meta:
                duration = meta["duration"]
                no_frames = meta["no_frames"]
                height = meta["height"]
                width = meta["width"]
            else:
                duration = 10.0
                no_frames = 0
                height = 720
                width = 1280

            fps = compute_video_fps(duration, no_frames, height, width)
            messages, sampled_frames = build_frame_messages(video_path, fps, duration)
            messages_batch.append(messages)
            prepared.append(
                {
                    "video_path": video_path,
                    "duration": duration,
                    "sampled_frames": sampled_frames,
                }
            )

        raws = generate_raw_batch(model, processor, messages_batch)
        return [
            asdict(
                parse_output(
                    raw,
                    item["video_path"],
                    item["duration"],
                    f"frames_batch{len(video_paths)}",
                    item["sampled_frames"],
                )
            )
            for item, raw in zip(prepared, raws)
        ]
    except Exception:
        return [infer_video(model, processor, video_path, meta=meta) for video_path, meta in zip(video_paths, metas)]


def run_all(model: Any, processor: Any, video_dir: str) -> pd.DataFrame:
    # 메타데이터 로드
    metadata = load_metadata(METADATA_CSV)
    print(f"[INFO] Loaded metadata for {len(metadata)} videos")

    paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not paths:
        raise FileNotFoundError(f"No .mp4 files found in: {video_dir}")

    print(f"[INFO] {len(paths)} videos found. Starting...\n")
    records: List[Dict[str, Any]] = []
    progress_bar = tqdm(total=len(paths), desc="Video inference", unit="video") if tqdm is not None else None

    try:
        for batch_start in range(0, len(paths), INFERENCE_BATCH_SIZE):
            batch_paths = paths[batch_start : batch_start + INFERENCE_BATCH_SIZE]
            batch_metas: List[Optional[Dict[str, Any]]] = []

            for offset, video_path in enumerate(batch_paths, start=batch_start + 1):
                video_name = Path(video_path).name
                meta_key = f"videos/{video_name}"
                meta = metadata.get(meta_key)
                batch_metas.append(meta)

                if meta:
                    fps = compute_video_fps(meta["duration"], meta["no_frames"], meta["height"], meta["width"])
                    est_frames = int(meta["duration"] * fps)
                    print(f"[{offset}/{len(paths)}] {video_name} | {meta['duration']:.1f}s {meta['width']}x{meta['height']} → fps={fps} (~{est_frames}f)")
                else:
                    print(f"[{offset}/{len(paths)}] {video_name} | no metadata")

            batch_records = infer_videos_batch(model, processor, batch_paths, batch_metas)
            for record in batch_records:
                print(
                    f"  -> {record['accident_time']:.2f}s | "
                    f"({record['center_x']:.3f}, {record['center_y']:.3f}) | "
                    f"{record['type']} | conf={record['confidence']:.2f} | {record['method']}"
                )
                records.append(record)

            if progress_bar is not None:
                progress_bar.update(len(batch_paths))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return pd.DataFrame(records)


def save_submission(df: pd.DataFrame, path: str) -> None:
    df.to_csv(DEBUG_CSV, index=False)
    print(f"[INFO] debug saved -> {DEBUG_CSV}")
    submission = df[["path", "accident_time", "center_x", "center_y", "type"]].copy()
    submission["accident_time"] = submission["accident_time"].astype(float).round(2)
    submission["center_x"] = submission["center_x"].astype(float).round(3)
    submission["center_y"] = submission["center_y"].astype(float).round(3)
    submission.to_csv(path, index=False)
    print(f"[INFO] submission saved -> {path}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    model, processor = load_model()
    dataframe = run_all(model, processor, VIDEO_DIR)
    save_submission(dataframe, OUTPUT_CSV)
    print("\nDone")

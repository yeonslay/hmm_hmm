from pathlib import Path

import pandas as pd

import baseline


RUN_DIR = Path(__file__).resolve().parent
OUTPUT_SUBMISSION = RUN_DIR / "outputs" / "submission_baseline.csv"
OUTPUT_DEBUG_CSV = RUN_DIR / "outputs" / "debug_results_baseline.csv"
OUTPUT_DEBUG_JSON = RUN_DIR / "outputs" / "debug_results_baseline.json"
CHECKPOINT_EVERY = 10


def save_outputs(df: pd.DataFrame) -> None:
    OUTPUT_DEBUG_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DEBUG_JSON.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_DEBUG_CSV, index=False)
    df.to_json(OUTPUT_DEBUG_JSON, orient="records", indent=2)

    submission = df[["path", "accident_time", "center_x", "center_y", "type"]].copy()
    submission["accident_time"] = submission["accident_time"].astype(float).round(2)
    submission["center_x"] = submission["center_x"].astype(float).round(3)
    submission["center_y"] = submission["center_y"].astype(float).round(3)
    submission.to_csv(OUTPUT_SUBMISSION, index=False)


def main() -> None:
    model, processor = baseline.load_model()
    metadata = baseline.load_metadata(baseline.METADATA_CSV)
    print(f"[INFO] Loaded metadata for {len(metadata)} videos")

    paths = sorted(Path(baseline.VIDEO_DIR).glob("*.mp4"))
    if not paths:
        raise FileNotFoundError(f"No .mp4 files found in: {baseline.VIDEO_DIR}")

    records = []
    iterator = enumerate(paths, start=1)
    if baseline.tqdm is not None:
        iterator = enumerate(baseline.tqdm(paths, desc="Video inference", unit="video"), start=1)

    for index, video_path in iterator:
        video_name = video_path.name
        meta_key = f"videos/{video_name}"
        meta = metadata.get(meta_key)

        if meta:
            fps = baseline.compute_video_fps(meta["duration"], meta["no_frames"], meta["height"], meta["width"])
            est_frames = int(meta["duration"] * fps)
            print(f"[{index}/{len(paths)}] {video_name} | {meta['duration']:.1f}s {meta['width']}x{meta['height']} → fps={fps} (~{est_frames}f)")
        else:
            print(f"[{index}/{len(paths)}] {video_name} | no metadata")

        record = baseline.infer_video(model, processor, str(video_path), meta=meta)
        print(
            f"  -> {record['accident_time']:.2f}s | "
            f"({record['center_x']:.3f}, {record['center_y']:.3f}) | "
            f"{record['type']} | conf={record['confidence']:.2f} | {record['method']}"
        )
        records.append(record)

        if index % CHECKPOINT_EVERY == 0:
            save_outputs(pd.DataFrame(records))
            print(f"[INFO] Checkpoint saved at {index}/{len(paths)} videos")

    dataframe = pd.DataFrame(records)
    save_outputs(dataframe)
    print(f"[INFO] Saved {OUTPUT_SUBMISSION}")
    print(f"[INFO] Saved {OUTPUT_DEBUG_CSV}")
    print(f"[INFO] Saved {OUTPUT_DEBUG_JSON}")


if __name__ == "__main__":
    main()

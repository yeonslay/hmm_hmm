# Inference Ready Exp5

Run from this folder with `./run_in_qwen.sh`.

Files:
- `baseline_flash_attn_batch2.py`: copied inference baseline with project-root-relative dataset paths
- `baseline.py`: import shim for `run_full_baseline.py`
- `run_full_baseline.py`: copied batch runner with outputs written under `outputs/`
- `run_in_qwen.sh`: launcher pinned to `/opt/conda/envs/qwen/bin/python`

Expected data paths:
- videos: `../raw/accident/videos`
- metadata: `../raw/accident/test_metadata.csv`

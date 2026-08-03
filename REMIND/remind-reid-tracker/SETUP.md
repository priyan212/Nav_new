# Self-contained setup (this checkout)

Everything this project needs at runtime lives inside this repo directory —
no other project's environment, cache, or files are referenced.

- **Python env**: `.venv/` — a conda env created with `conda create -p .venv`
  (Python 3.12 + CUDA 12.8 torch/torchvision + all deps in `requirements.txt`).
- **YOLO weights**: `yolo/yolo11l-seg.pt` (YOLO11-Large segmentation, the
  detector used in the paper's main results).
- **DINOv3 weights**: cached under `.cache/huggingface/` on first run.
  `facebook/dinov3-vits16-pretrain-lvd1689m` is a gated model — you must
  accept Meta's license on the model page and provide a HuggingFace **read**
  token once. The token is stored at `.cache/huggingface/token`
  (mode 600, listed in `.gitignore`, never committed).
- **Ultralytics config/cache**: `.cache/ultralytics/`.

`run.sh` and `run_reid.sh` set `HF_HOME` / `YOLO_CONFIG_DIR` to these
repo-local paths and activate `.venv` automatically, so you never need to
export anything globally.

## Running on one video

```bash
mkdir -p testData/videos/my_room
cp /path/to/video.mp4 testData/videos/my_room/video.mp4
./run.sh my_room yolo11l-seg.pt --input-video-fps 5 --output-fps 25 --save-output-video
```

Outputs land in `outputs/video_runs/<scene>_<timestamp>/`: `tracking.mp4`,
`frames.csv`, `detections.jsonl`, `summary.json`.

## Running the two-video re-identification demo

This reproduces the paper's "leave and re-enter" scenario (Fig. 1): video1
builds the object memory (dual-bank appearance, part, background descriptors
+ neighbor-context graph), video2 — the same room from a different
angle/time — is matched against that same in-memory catalogue with no reset
in between.

```bash
./run_reid.sh my_room yolo11l-seg.pt \
  --video1 /path/to/first_visit.mp4 \
  --video2 /path/to/revisit_different_angle.mp4 \
  --input-video-fps 5 --output-fps 25
```

Outputs land in `outputs/video_runs/<scene>_reid_<timestamp>/`:
- `video1_tracking.mp4` / `video2_tracking.mp4` — rendered with persistent
  `ID <n>` labels; an object keeps the **same ID and color** in both videos
  if REMIND re-identified it.
- `video1_frames.csv`, `video2_frames.csv`, `*_detections.jsonl` — per-frame
  logs (same schema as the single-video mode, plus a
  `reidentified_from_video1` count column in video2's CSV).
- `reid_report.json` — the quantitative result: how many objects seen in
  video1 were correctly matched in video2 (`reidentification_rate`), which
  ones were missed, and which detections became new identities in video2.

## Re-creating the environment from scratch

```bash
conda create -p .venv python=3.12 -y
conda activate ./.venv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

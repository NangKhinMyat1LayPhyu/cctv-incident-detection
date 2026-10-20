# Usage

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[service,dev]"
```

Runs on an NVIDIA GPU (CUDA), an Apple Silicon Mac (MPS) or CPU. Every tool is a subcommand of
`incident-recon`; run it alone for the list.

## Commands

```bash
# Analyse one clip: annotated video, tracks, incident report with key frames
incident-recon analyze clip.mp4 --weights best.pt --out outputs/analyze

# Live alerts on one or more cameras (RTSP URL, webcam number or video file)
incident-recon live --source rtsp://host:8554/cam1 --weights best.pt --seconds 0 --out outputs/live
incident-recon live --source north=rtsp://.../cam1 --source south=rtsp://.../cam2 --weights best.pt --seconds 0

# Dashboard for the live run (http://127.0.0.1:8000)
incident-recon serve --live-dir outputs/live --weights best.pt

# Score the detector on the held-out set
incident-recon eval --weights best.pt --videos path/to/TAD --out results/heldout_mine.csv
incident-recon score --results results/heldout_mine.csv --videos path/to/TAD
```

Add `--realtime` to `live` to play a video file at its own frame rate, like a camera.

## Docker

```bash
docker build -t incident-recon .
docker run --rm -v "$PWD/data:/data" -v "$HOME/models:/models" incident-recon \
    analyze /data/clip.mp4 --weights /models/best.pt --out /data/out
```

## Tests

```bash
pytest -q && ruff check .
```

## Training

```bash
python scripts/prepare_detrac.py --images path/to/DETRAC-Images --annotations path/to/DETRAC-Train-Annotations-XML --out data/detrac_yolo
python scripts/pseudo_label.py --data data/detrac_yolo
python scripts/train.py --data data/detrac_yolo/data.yaml --degrade 0.3
python scripts/make_ablation_sets.py --data data/detrac_yolo/data.yaml --out data/ablation
python scripts/eval_ablation.py --sets data/ablation --model "COCO=yolo11n.pt" --model "fine-tuned=best.pt"
```

## Export and benchmark

```bash
pip install -e ".[export]"
python scripts/export.py --weights best.pt --out models/export
python scripts/bench_models.py --video clip.mp4 --model pytorch=best.pt --model coreml16=models/export/best_coreml16.mlpackage
```

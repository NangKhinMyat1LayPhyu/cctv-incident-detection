"""Detect and track vehicles in a video, saving an annotated video and a tracks CSV.

Works with any YOLO model (COCO-pretrained or our fine-tuned one): classes are
looked up by name in the model itself.
"""

from __future__ import annotations

import csv
import tempfile
import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

from .device import get_device

# Classes we track, by NAME. Class numbers differ between models
# (COCO: car=2, motorcycle=3 ... ours: car=0, motorcycle=1 ...), names do not.
VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck", "bicycle"}
PERSON_NAMES = {"person"}


def tracker_settings(conf: float, low_conf: float | None) -> dict:
    """ByteTrack settings for our two confidence levels.

    conf:     a box needs this score to start a track or to match in ByteTrack's first stage.
    low_conf: boxes scoring between low_conf and conf may only CONTINUE an existing track
              (ByteTrack's second stage, its main idea: keep partly hidden vehicles tracked).
              None = the original setup: the detector drops every box under conf, so the
              second stage never receives anything.
    """
    cfg = YAML.load(check_yaml("bytetrack.yaml"))
    if low_conf is not None:
        cfg.update(track_high_thresh=conf, new_track_thresh=conf, track_low_thresh=low_conf)
    return cfg


def detector_conf(conf: float, low_conf: float | None) -> float:
    """Lowest score the detector must keep for the tracker settings above."""
    return conf if low_conf is None else low_conf


def track_video(
    video_path: str | Path,
    out_dir: str | Path,
    weights: str = "yolo11n.pt",
    conf: float = 0.3,
    imgsz: int = 640,
    include_people: bool = True,
    save_video: bool = True,
    device: str | None = None,
    low_conf: float | None = None,
) -> dict:
    """Run detection + tracking on one video.

    Writes:
      <out_dir>/<name>_tracked.mp4  annotated video (unless save_video=False)
      <out_dir>/<name>_tracks.csv   one row per vehicle per frame
    device: None = best available (get_device()).
    low_conf: see tracker_settings (None = original setup).
    Returns a small summary dict (frames, fps, unique ids per class).
    """
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = YOLO(weights, task="detect")
    device = device or get_device()

    # Class numbers differ between models; look them up by name (model.names: {id: name})
    wanted = VEHICLE_NAMES | (PERSON_NAMES if include_people else set())
    classes = {i: name for i, name in model.names.items() if name in wanted}

    video_out = out_dir / f"{video_path.stem}_tracked.mp4"
    csv_out = out_dir / f"{video_path.stem}_tracks.csv"
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)) \
        if save_video else None

    seen: dict[str, set[int]] = {name: set() for name in classes.values()}
    n_frames = 0
    t0 = time.perf_counter()

    tracker = "bytetrack.yaml"
    if low_conf is not None:     # a tracker file with our thresholds, for model.track()
        tracker = str(Path(tempfile.mkdtemp()) / "bytetrack.yaml")
        YAML.save(tracker, tracker_settings(conf, low_conf))

    with open(csv_out, "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["frame", "time_s", "track_id", "class", "conf", "x1", "y1", "x2", "y2"])

        results = model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            tracker=tracker,
            classes=list(classes),
            conf=detector_conf(conf, low_conf),
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        for frame_idx, r in enumerate(results):
            n_frames += 1
            if writer is not None:
                writer.write(r.plot())
            boxes = r.boxes
            if boxes is None or boxes.id is None:
                continue  # nothing tracked in this frame
            for tid, cls, cf, xyxy in zip(
                boxes.id.int().tolist(),
                boxes.cls.int().tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.tolist(),
            ):
                name = classes.get(cls, str(cls))
                seen.setdefault(name, set()).add(tid)
                rows.writerow([frame_idx, round(frame_idx / fps, 3), tid, name, round(cf, 3),
                               *(round(v, 1) for v in xyxy)])

    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - t0
    return {
        "video": video_path.name,
        "device": device,
        "frames": n_frames,
        "processing_fps": round(n_frames / elapsed, 1) if elapsed else None,
        "unique_ids": {k: len(v) for k, v in seen.items() if v},
        "annotated_video": str(video_out) if save_video else None,
        "tracks_csv": str(csv_out),
    }

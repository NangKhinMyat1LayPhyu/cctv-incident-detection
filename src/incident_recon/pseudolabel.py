"""Add pseudo-labels for classes a dataset is missing, using a COCO-pretrained model.

UA-DETRAC labels only vehicles on the road. People and two-wheelers in its images
are unlabelled, which would teach our model that they are background. We fill
that gap with confident detections from a large COCO model.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from .device import get_device

# COCO class id -> our class id (see detrac.CLASS_NAMES)
COCO_TO_OURS = {0: 5, 1: 4, 3: 1}  # person, bicycle, motorcycle


def pseudo_label_split(
    images_dir: str | Path,
    labels_dir: str | Path,
    weights: str = "yolo11x.pt",
    conf: float = 0.5,
    imgsz: int = 960,
    batch: int = 16,
) -> Counter:
    """Append pseudo-labels to every label file in one split. Returns counts per class id."""
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    model = YOLO(weights)
    added: Counter = Counter()

    results = model.predict(
        source=str(images_dir),
        stream=True,                  # one result at a time, low memory
        classes=list(COCO_TO_OURS),   # only person, bicycle, motorcycle
        conf=conf,                    # only confident detections
        imgsz=imgsz,
        batch=batch,
        device=get_device(),
        verbose=False,
    )
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        lines = []
        # xywhn = centre x, centre y, width, height, already as fractions: YOLO format
        for cls, (cx, cy, w, h) in zip(r.boxes.cls.int().tolist(), r.boxes.xywhn.tolist()):
            ours = COCO_TO_OURS[cls]
            lines.append(f"{ours} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            added[ours] += 1

        label_path = labels_dir / f"{Path(r.path).stem}.txt"
        with open(label_path, "a") as f:  # "a" = append, keep the original vehicle labels
            f.write("\n".join(lines) + "\n")

    return added

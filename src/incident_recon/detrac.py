"""Convert UA-DETRAC (original XML annotations) to YOLO format.

UA-DETRAC: one folder of frames per video (MVI_xxxxx/img00001.jpg ...) and one
XML file per video describing every vehicle box in every frame.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

# Class list for the whole project. UA-DETRAC labels only car, van, bus and "others" (see CLASS_MAP);
# ids 1, 4, 5 (motorcycle, bicycle, person) are added by pseudo-labelling (pseudolabel.py).
CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle", "person"]

# UA-DETRAC vehicle_type -> our class id
CLASS_MAP = {"car": 0, "van": 0, "bus": 2, "others": 3}


def _box(el) -> tuple[float, float, float, float]:
    """Read left, top, width, height from a <box> element."""
    return tuple(float(el.get(k)) for k in ("left", "top", "width", "height"))


def read_weather(xml_path) -> str:
    """Return the sequence's weather without parsing the whole (large) file."""
    for _, el in ET.iterparse(str(xml_path)):
        if el.tag == "sequence_attribute":
            return el.get("sence_weather")  # sic: original spelling
    return "unknown"


def read_sequence(xml_path):
    """Read one UA-DETRAC XML file.

    Returns (name, ignored, frames):
      name     video name, e.g. "MVI_20011"
      ignored  list of (left, top, width, height) regions to black out
      frames   dict: frame number -> list of (vehicle_type, left, top, width, height)
    """
    root = ET.parse(str(xml_path)).getroot()

    ignored = []
    region = root.find("ignored_region")
    if region is not None:
        ignored = [_box(b) for b in region.findall("box")]

    frames = {}
    for frame in root.findall("frame"):
        objects = []
        for target in frame.iter("target"):
            vtype = target.find("attribute").get("vehicle_type")
            objects.append((vtype, *_box(target.find("box"))))
        frames[int(frame.get("num"))] = objects

    return root.get("name"), ignored, frames


def to_yolo_line(class_id, left, top, width, height, img_w, img_h) -> str | None:
    """Pixel box (top-left corner + size) -> YOLO line (class cx cy w h, as fractions)."""
    # Clip the box to the image, so no fraction is below 0 or above 1
    x1, y1 = max(0.0, left), max(0.0, top)
    x2, y2 = min(img_w, left + width), min(img_h, top + height)
    if x2 <= x1 or y2 <= y1:
        return None  # box is completely outside the image

    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def split_by_weather(xml_paths, val_per_weather=3, seed=0):
    """Split videos into train/val, taking the same number of val videos per weather."""
    rng = random.Random(seed)
    groups: dict[str, list[Path]] = {}
    for p in sorted(xml_paths):
        groups.setdefault(read_weather(p), []).append(p)

    train, val = [], []
    for weather in sorted(groups):
        paths = groups[weather][:]
        rng.shuffle(paths)
        val += paths[:val_per_weather]
        train += paths[val_per_weather:]
    return train, val


def convert_sequence(xml_path, images_dir, out_images, out_labels, every=10):
    """Convert one video: write every Nth frame (ignored regions blacked out) + its labels.

    Returns (images_written, boxes_written).
    """
    name, ignored, frames = read_sequence(xml_path)
    seq_dir = Path(images_dir) / name
    out_images, out_labels = Path(out_images), Path(out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = 0
    for num in sorted(frames)[::every]:
        img_path = seq_dir / f"img{num:05d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue  # frame image missing or unreadable
        img_h, img_w = img.shape[:2]

        # Black out regions where vehicles exist but were not labelled
        for left, top, width, height in ignored:
            x1, y1 = max(0, int(left)), max(0, int(top))
            x2, y2 = int(left + width), int(top + height)
            img[y1:y2, x1:x2] = 0

        lines = []
        for vtype, left, top, width, height in frames[num]:
            class_id = CLASS_MAP.get(vtype)
            if class_id is None:
                continue  # unknown vehicle type
            line = to_yolo_line(class_id, left, top, width, height, img_w, img_h)
            if line:
                lines.append(line)

        stem = f"{name}_{img_path.stem}"  # e.g. MVI_20011_img00001
        cv2.imwrite(str(out_images / f"{stem}.jpg"), img)
        (out_labels / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_images += 1
        n_boxes += len(lines)

    return n_images, n_boxes

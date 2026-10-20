"""Make degraded copies of training images, to look like poor CCTV crash footage.

Crash footage is often low resolution (320x240), grayscale (infrared night
cameras), dark, blurred and heavily compressed. UA-DETRAC images are much cleaner,
so degraded copies teach the model what bad footage looks like. Boxes do not move (the image keeps its size), so labels are copied as-is.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SUFFIX = "_deg"


def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Return a degraded copy of a BGR image, same size."""
    h, w = img.shape[:2]

    # 1. Low resolution: shrink to 25-50% (about 240-480 px wide), then enlarge back
    scale = rng.uniform(0.25, 0.5)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Grayscale, like infrared night cameras
    if rng.random() < 0.5:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 3. Darker, like night
    if rng.random() < 0.4:
        out = (out.astype(np.float32) * rng.uniform(0.3, 0.6)).astype(np.uint8)

    # 4. Motion blur, like fast vehicles
    if rng.random() < 0.3:
        k = rng.choice([5, 7, 9, 11])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k  # horizontal streak
        out = cv2.filter2D(out, -1, kernel)

    # 5. Heavy JPEG compression, like old CCTV recorders and re-uploaded videos
    if rng.random() < 0.5:
        quality = rng.randint(15, 40)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return out


def degrade_split(images_dir, labels_dir, fraction: float = 0.3, seed: int = 0) -> int:
    """Add degraded copies of a random fraction of a split's images. Returns copies made."""
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    originals = sorted(p for p in images_dir.glob("*.jpg") if not p.stem.endswith(SUFFIX))
    if any(images_dir.glob(f"*{SUFFIX}.jpg")):
        raise RuntimeError(f"Degraded copies already exist in {images_dir}")

    rng = random.Random(seed)
    chosen = rng.sample(originals, int(len(originals) * fraction))

    for img_path in chosen:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        stem = img_path.stem + SUFFIX
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), degrade(img, rng))
        shutil.copy(labels_dir / f"{img_path.stem}.txt", labels_dir / f"{stem}.txt")

    return len(chosen)

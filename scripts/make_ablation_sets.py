"""Build the evaluation sets for the detector comparison (ablation) table.

From the UA-DETRAC YOLO dataset (data.yaml with images/val, labels/val) it creates:
  <out>/ours/        val (clean) + val_deg (every image degraded), OUR class numbers
  <out>/coco/        the same two sets with labels renumbered to COCO class numbers,
                     so the original COCO model can be scored on exactly the same boxes
Each folder gets val.yaml and val_deg.yaml for Ultralytics' model.val().

Usage:
  python scripts/make_ablation_sets.py --data data/detrac_yolo/data.yaml --out data/ablation
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from incident_recon.degrade import degrade  # noqa: E402
from incident_recon.detrac import CLASS_NAMES  # noqa: E402

# our class number -> COCO class number (by name: car, motorcycle, bus, truck, bicycle, person)
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
OURS_TO_COCO = {i: next(k for k, v in COCO_NAMES.items() if v == n) for i, n in enumerate(CLASS_NAMES)}


def _write_yamls(root: Path, names: dict) -> None:
    """val.yaml (clean) and val_deg.yaml (degraded). 'train' is required by Ultralytics
    but not used by model.val(), so it simply points at the same folder."""
    lines = "\n".join(f"  {i}: {n}" for i, n in sorted(names.items()))
    for split in ("val", "val_deg"):
        (root / f"{split}.yaml").write_text(
            f"path: {root.resolve()}\ntrain: images/{split}\nval: images/{split}\nnames:\n{lines}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="UA-DETRAC data.yaml from prepare_detrac.py")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    src = Path(args.data).parent
    out = Path(args.out)
    images = sorted((src / "images" / "val").glob("*.jpg"))
    images = [im for im in images if not im.stem.endswith("_deg")]
    rng = random.Random(args.seed)

    for kind in ("ours", "coco"):
        for split in ("val", "val_deg"):
            (out / kind / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / kind / "labels" / split).mkdir(parents=True, exist_ok=True)

    for im_path in images:
        label = src / "labels" / "val" / f"{im_path.stem}.txt"
        lines = label.read_text().splitlines() if label.exists() else []
        coco_lines = [" ".join([str(OURS_TO_COCO[int(ln.split()[0])])] + ln.split()[1:]) for ln in lines if ln.strip()]

        img = cv2.imread(str(im_path))
        img_deg = degrade(img, rng)            # same degradation function as in training
        for kind, lab in (("ours", lines), ("coco", coco_lines)):
            shutil.copy(im_path, out / kind / "images" / "val" / im_path.name)
            cv2.imwrite(str(out / kind / "images" / "val_deg" / im_path.name), img_deg)
            for split in ("val", "val_deg"):
                (out / kind / "labels" / split / f"{im_path.stem}.txt").write_text("\n".join(lab) + ("\n" if lab else ""))

    _write_yamls(out / "ours", dict(enumerate(CLASS_NAMES)))
    _write_yamls(out / "coco", {i: f"class{i}" for i in range(80)} | COCO_NAMES)
    print(f"{len(images)} val images -> {out}/ours and {out}/coco (clean + degraded)")


if __name__ == "__main__":
    main()

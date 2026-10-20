"""Add person / bicycle / motorcycle pseudo-labels to a converted YOLO dataset.

Usage:
  python scripts/pseudo_label.py --data data/detrac_yolo
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incident_recon.detrac import CLASS_NAMES  # noqa: E402
from incident_recon.pseudolabel import pseudo_label_split  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Dataset root with images/ and labels/")
    p.add_argument("--weights", default="yolo11x.pt", help="Large COCO model used as the labeller")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    args = p.parse_args()

    data = Path(args.data)
    marker = data / "pseudo_labels.json"
    if marker.exists():
        # Labels are appended, so running twice would duplicate every pseudo-label
        sys.exit(f"Pseudo-labels already added ({marker}). Re-run the converter first to start clean.")

    record = {"weights": args.weights, "conf": args.conf, "imgsz": args.imgsz, "added": {}}
    for split in args.splits:
        print(f"Pseudo-labelling {split}...")
        added = pseudo_label_split(
            data / "images" / split,
            data / "labels" / split,
            weights=args.weights,
            conf=args.conf,
            imgsz=args.imgsz,
        )
        named = {CLASS_NAMES[k]: v for k, v in sorted(added.items())}
        record["added"][split] = named
        print(f"  {split}: added {named}")

    marker.write_text(json.dumps(record, indent=2))
    print(f"\nDone. Record: {marker}")


if __name__ == "__main__":
    main()

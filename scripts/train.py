"""Add degraded copies (optional) and fine-tune YOLO on the traffic dataset.

Usage:
  python scripts/train.py --data data/detrac_yolo/data.yaml --degrade 0.3
  python scripts/train.py --data ... --fraction 0.05 --epochs 2   # quick test first
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ultralytics import YOLO  # noqa: E402

from incident_recon.degrade import degrade_split  # noqa: E402
from incident_recon.device import get_device  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--model", default="yolo11n.pt", help="Pretrained weights to start from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=10, help="Stop if no improvement for N epochs")
    p.add_argument("--fraction", type=float, default=1.0, help="Use part of the train set (quick tests)")
    p.add_argument("--degrade", type=float, default=0.0, help="Fraction of train images to add degraded copies of")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="detrac_yolo11n")
    args = p.parse_args()

    root = Path(args.data).parent
    if args.degrade > 0:
        n = degrade_split(root / "images" / "train", root / "labels" / "train", args.degrade)
        print(f"Added {n} degraded training images")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        fraction=args.fraction,
        device=get_device(),
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=0,
        plots=True,
    )

    # Evaluate the best checkpoint on the validation set
    run_dir = Path(args.project) / args.name
    best = YOLO(run_dir / "weights" / "best.pt")
    metrics = best.val(data=args.data, imgsz=args.imgsz, device=get_device(), plots=False)

    names = best.names
    per_class = {names[c]: round(float(m), 3) for c, m in zip(metrics.box.ap_class_index, metrics.box.maps[metrics.box.ap_class_index])}
    summary = {
        "model": args.model,
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "degrade": args.degrade,
        "mAP50": round(float(metrics.box.map50), 3),
        "mAP50_95": round(float(metrics.box.map), 3),
        "mAP50_95_per_class": per_class,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nBest weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

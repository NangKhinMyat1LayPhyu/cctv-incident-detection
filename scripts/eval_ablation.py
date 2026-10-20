"""Score several detectors on clean and degraded UA-DETRAC val (the ablation table).

Every model is scored on the SAME images and boxes. The COCO model is scored on the
copy whose labels use COCO class numbers (made by make_ablation_sets.py), restricted
to our 6 classes.

Usage:
  python scripts/eval_ablation.py --sets data/ablation \
      --model "COCO pretrained=yolo11n.pt" \
      --model "fine-tuned=runs/detrac_nodeg/weights/best.pt" \
      --model "fine-tuned + degradation=runs/detrac_yolo11n/weights/best.pt" \
      --out results/detector_ablation.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ultralytics import YOLO  # noqa: E402

from incident_recon.device import get_device  # noqa: E402

COCO_IDS = [0, 1, 2, 3, 5, 7]   # person, bicycle, car, motorcycle, bus, truck


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sets", required=True, help="Output folder of make_ablation_sets.py")
    p.add_argument("--model", action="append", required=True, help='"NAME=weights", repeat per model')
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="ablation_results.json")
    args = p.parse_args()

    sets = Path(args.sets)
    rows = []
    for item in args.model:
        name, weights = item.split("=", 1)
        model = YOLO(weights)
        is_coco = len(model.names) == 80
        folder = sets / ("coco" if is_coco else "ours")
        row = {"model": name, "weights": weights}
        for split in ("val", "val_deg"):
            m = model.val(data=str(folder / f"{split}.yaml"), imgsz=args.imgsz, batch=32, device=get_device(),
                          classes=COCO_IDS if is_coco else None, plots=False, verbose=False)
            names = model.names
            row[split] = {
                "mAP50": round(float(m.box.map50), 3),
                "mAP50_95": round(float(m.box.map), 3),
                "per_class_mAP50": {names[int(c)]: round(float(v), 3)
                                    for c, v in zip(m.box.ap_class_index, m.box.ap50)},
            }
        rows.append(row)
        print(f"{name:28s}  clean mAP50 {row['val']['mAP50']:.3f}   degraded mAP50 {row['val_deg']['mAP50']:.3f}")

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print("\n| Detector | mAP@50 (UA-DETRAC val) | mAP@50 (degraded val) |\n|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['val']['mAP50']:.3f} | {r['val_deg']['mAP50']:.3f} |")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

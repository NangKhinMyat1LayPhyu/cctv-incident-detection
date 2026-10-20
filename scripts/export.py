"""Export the trained detector to deployment formats.

PyTorch is for training; for running, a model is exported once to a format a fast
runtime understands:
  onnx        ONNX, FP32, any batch size - portable: ONNX Runtime, TensorRT, OpenVINO, edge boxes
  onnx_static ONNX, FP32, batch 1, fixed shape - needed by ONNX Runtime's CoreML provider
              (it fails on the dynamic model) and by many edge runtimes
  coreml16    CoreML, FP16 weights - Apple's runtime, can use the Neural Engine
              (CoreML stores FP16 by default: a plain CoreML export is this same model)
  coreml8     CoreML, INT8 weights (8-bit lookup tables) - about half the size, may cost accuracy

Every export must be re-checked for accuracy (scripts/eval_model.py), not just speed.

Usage:
  python scripts/export.py --weights best.pt --out models/export
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ultralytics import YOLO  # noqa: E402

VARIANTS = {
    "onnx": dict(format="onnx", dynamic=True, simplify=True),   # dynamic: any batch size
    "onnx_static": dict(format="onnx", dynamic=False, simplify=True),
    "coreml16": dict(format="coreml", half=True),
    "coreml8": dict(format="coreml", int8=True),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True, help="Folder for the exported models")
    p.add_argument("--only", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    weights = Path(args.weights).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        made = Path(YOLO(weights).export(imgsz=args.imgsz, **VARIANTS[name]))
        target = out / f"{weights.stem}_{name}{made.suffix}"
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(made), target)   # exports land next to the weights; give each its own name
        size = sum(f.stat().st_size for f in target.rglob("*")) if target.is_dir() else target.stat().st_size
        print(f"{name:9s} -> {target} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

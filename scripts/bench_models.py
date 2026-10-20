"""Compare the speed of exported models on the same video frames.

Each model runs in its own process (so peak memory is its own), on the same frames,
after a warm-up. Reported per model and batch size:
  latency     wall time per predict() call (pre-processing + model + NMS), mean and p95
  per frame   latency / batch size
  memory      peak resident memory of the process
  agreement   how well its boxes match the PyTorch model's boxes on the same frames
              (a box matches if same class and IoU >= 0.5): a quick accuracy check,
              the real one is the held-out evaluation (scripts/eval_model.py).

Batch sizes above 1 are skipped for models that take one image at a time: CoreML
(Ultralytics' CoreML runtime only uses the FIRST image of a batch) and fixed-shape ONNX.

Usage:
  python scripts/bench_models.py --video clip.mp4 \
      --model pytorch=best.pt --model onnx=models/export/best_onnx.onnx \
      --model coreml16=models/export/best_coreml16.mlpackage --out results/bench
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from incident_recon.models import batch_one_only  # noqa: E402


def read_frames(video, n: int) -> list:
    cap = cv2.VideoCapture(str(video))
    frames = []
    while len(frames) < n:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    return frames


def run_one(spec: dict) -> dict:
    """Benchmark one model (runs inside its own process)."""
    from ultralytics import YOLO

    from incident_recon.track import PERSON_NAMES, VEHICLE_NAMES

    frames = read_frames(spec["video"], spec["frames"])
    model = YOLO(spec["path"], task="detect")
    classes = [i for i, name in model.names.items() if name in VEHICLE_NAMES | PERSON_NAMES]
    kw = dict(classes=classes, conf=spec["conf"], device=spec["device"], verbose=False)
    for img in frames[:10]:                                    # warm-up: first calls compile / load
        model.predict(img, **kw)

    out = {"label": spec["label"], "device": spec["device"], "batches": {}}
    boxes = []
    for bs in spec["batch_sizes"]:
        lat = []
        for i in range(0, len(frames) - bs + 1, bs):
            chunk = frames[i:i + bs]
            t = time.perf_counter()
            results = model.predict(chunk if bs > 1 else chunk[0], **kw)
            lat.append((time.perf_counter() - t) * 1000)
            if bs == 1:
                b = results[0].boxes
                boxes.append(np.c_[b.xyxy.cpu().numpy(), b.cls.cpu().numpy()])
        lat = np.array(lat)
        out["batches"][bs] = {"mean_ms": round(float(lat.mean()), 2),
                              "p95_ms": round(float(np.percentile(lat, 95)), 2),
                              "per_frame_ms": round(float(lat.mean() / bs), 2)}
    out["peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 0)  # bytes on macOS
    np.save(spec["boxes_file"], np.array(boxes, dtype=object), allow_pickle=True)
    return out


def iou(a, b) -> np.ndarray:
    """IoU matrix between box arrays a (N,4) and b (M,4), xyxy."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def agreement(ref: list, test: list) -> dict:
    """Share of reference boxes found by the test model, and of test boxes found by the reference."""
    matched = n_ref = n_test = 0
    for r, t in zip(ref, test):
        n_ref, n_test = n_ref + len(r), n_test + len(t)
        if len(r) and len(t):
            m = iou(r[:, :4], t[:, :4]) * (r[:, None, 4] == t[None, :, 4])
            used = np.zeros(len(t), dtype=bool)
            for i in np.argsort(-m.max(1)):          # greedy one-to-one matching, best pairs first
                cand = np.where(used, -1.0, m[i])
                j = int(cand.argmax())
                if cand[j] >= 0.5:
                    used[j] = True
                    matched += 1
    return {"recall_vs_pytorch": round(matched / max(n_ref, 1), 3),
            "precision_vs_pytorch": round(matched / max(n_test, 1), 3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", help="Clip to take the frames from")
    p.add_argument("--model", action="append", help="label=path[@device]; device default: mps (onnx@cpu for CPU)")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--out", default="results/bench")
    p.add_argument("--one", help=argparse.SUPPRESS)     # internal: run one spec (JSON) and print the result
    args = p.parse_args()

    if args.one:
        print("RESULT " + json.dumps(run_one(json.loads(args.one))))
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for text in args.model:
        label, _, rest = text.partition("=")
        path, _, device = rest.partition("@")
        path = str(Path(path).expanduser())
        spec = {"label": label, "path": path, "device": device or ("cpu" if path.endswith(".mlpackage") else "mps"),
                "video": str(Path(args.video).expanduser()), "frames": args.frames, "conf": args.conf,
                "batch_sizes": [1] if batch_one_only(path) else args.batch, "boxes_file": str(out / f"boxes_{label}.npy")}
        print(f"benchmarking {label} ({spec['device']})...", flush=True)
        proc = subprocess.run([sys.executable, __file__, "--one", json.dumps(spec)], capture_output=True, text=True)
        line = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
        if not line:
            print(proc.stdout[-2000:], proc.stderr[-2000:])
            sys.exit(f"{label} failed")
        rows.append(json.loads(line[0][7:]))

    ref = np.load(out / f"boxes_{rows[0]['label']}.npy", allow_pickle=True)
    for r in rows:
        r.update(agreement(ref, np.load(out / f"boxes_{r['label']}.npy", allow_pickle=True)))

    batch_sizes = sorted({int(b) for r in rows for b in r["batches"]})
    head = "| Model | Device | " + " | ".join(f"Batch {b}: mean / p95 / per frame" for b in batch_sizes) + \
           f" | Peak memory | Boxes vs {rows[0]['label']}: recall / precision |"
    md = [f"Frames: {args.frames} from `{Path(args.video).name}`, conf {args.conf}. "
          "Latency = one predict() call: pre-processing + model + NMS.", "",
          head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        cells = []
        for b in batch_sizes:
            s = r["batches"].get(str(b))
            cells.append(f"{s['mean_ms']:.1f} / {s['p95_ms']:.1f} / {s['per_frame_ms']:.1f} ms" if s else "-")
        md.append(f"| {r['label']} | {r['device']} | " + " | ".join(cells) +
                  f" | {r['peak_mb']:.0f} MB | {r['recall_vs_pytorch']:.3f} / {r['precision_vs_pytorch']:.3f} |")
    (out / "bench.md").write_text("\n".join(md) + "\n")
    (out / "bench.json").write_text(json.dumps(rows, indent=2))
    for f in out.glob("boxes_*.npy"):
        f.unlink()
    print("\n".join(md))


if __name__ == "__main__":
    main()

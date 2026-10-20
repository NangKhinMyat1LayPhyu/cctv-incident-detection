"""Run the full pipeline with one detector on every held-out (or dev) clip.

Steps: track_video (YOLO + ByteTrack) -> camera guard ->
track cleaning -> motion cues -> incident rules v1 and v2. Writes the per-clip table
(clip, label, v1, v2, v2_event) that scripts/eval_heldout.py scores, so every exported
model is judged on accuracy, not just speed.

The camera guard does not depend on the detector, so its result is cached per clip.

Usage:
  python scripts/eval_model.py --weights models/best_coreml16.mlpackage --videos path/to/TAD \
      --out results/export/heldout_coreml16.csv
  python scripts/eval_heldout.py --results results/export/heldout_coreml16.csv --videos path/to/TAD
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from incident_recon.camera import main_view  # noqa: E402
from incident_recon.clean import clean_tracks  # noqa: E402
from incident_recon.events import detect_incidents, motion_table  # noqa: E402
from incident_recon.track import track_video  # noqa: E402


def video_fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def verdicts(video: Path, tracks_csv: Path, fps: float, valid) -> dict:
    raw = pd.read_csv(tracks_csv)
    if raw.empty:
        return {"v1": False, "v2": False, "v2_event": ""}
    motion = motion_table(clean_tracks(raw, fps=fps), fps=fps, valid=valid)
    v1 = detect_incidents(motion, fps=fps)
    v2 = detect_incidents(motion, fps=fps, max_queue_neighbours=6)
    first = v2.iloc[0] if len(v2) else None
    return {"v1": bool(len(v1)), "v2": bool(len(v2)),
            "v2_event": f"{first['type']} @ {first['time_s']}s" if first is not None else ""}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--videos", required=True, nargs="+", help="Folder(s) with the clips (searched recursively)")
    p.add_argument("--split", default="results/split_v2.json")
    p.add_argument("--out", required=True, help="Per-clip results CSV")
    p.add_argument("--device", default=None, help="Default: best available (mps / cuda / cpu)")
    p.add_argument("--cache", default="outputs/camera_cache", help="Cached camera-guard results")
    p.add_argument("--low-conf", type=float, default=None,
                   help="Let boxes from this score up continue existing tracks (ByteTrack's second stage)")
    p.add_argument("--set", choices=["heldout", "dev"], default="heldout", help="Which clips of the split")
    p.add_argument("--tracks", default=None,
                   help="Keep the tracks CSVs in this folder, and reuse them if present (no detection rerun)")
    args = p.parse_args()

    split = json.loads(Path(args.split).read_text())
    if args.set == "dev":
        acc, norm = split["dev"]["dev_accident"], split["dev"]["dev_normal"]
    else:
        acc, norm = split["heldout_v2_accident"], split["heldout_v2_normal"]
    clips = [(c, "accident") for c in acc] + [(c, "normal") for c in norm]
    files = {v.stem: v for folder in args.videos for v in Path(folder).expanduser().rglob("*.mp4")}
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    rows, frames, t0 = [], 0, time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        track_dir = Path(args.tracks) if args.tracks else Path(tmp)
        for i, (clip, label) in enumerate(clips, 1):
            name = clip.removesuffix(".mp4")
            video = files[name]
            tracks_csv = track_dir / f"{name}_tracks.csv"
            if args.tracks and tracks_csv.exists():
                s = {"tracks_csv": str(tracks_csv), "frames": 0}
            else:
                s = track_video(video, track_dir, weights=str(Path(args.weights).expanduser()), save_video=False,
                                device=args.device, low_conf=args.low_conf)
            cached = cache / f"{name}.npy"
            if cached.exists():
                valid = np.load(cached)
            else:
                valid = main_view(video)["valid"]
                np.save(cached, valid)
            rows.append({"clip": name, "label": label,
                         **verdicts(video, Path(s["tracks_csv"]), video_fps(video), valid)})
            frames += s["frames"]
            print(f"[{i}/{len(clips)}] {name}: v1={rows[-1]['v1']} v2={rows[-1]['v2']} {rows[-1]['v2_event']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{frames} frames in {time.perf_counter() - t0:.0f} s. Per-clip results: {out}")


if __name__ == "__main__":
    main()

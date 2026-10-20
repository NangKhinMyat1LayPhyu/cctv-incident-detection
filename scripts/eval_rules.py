"""Compare incident-rule versions on saved tracks: accuracy AND stability.

The detector is run once per model (scripts/eval_model.py --tracks ...); this script only
re-runs the rules on those tracks, so a rule change is evaluated in seconds.

Stability: the same clip tracked by numerically different detectors (PyTorch, CoreML FP16,
CoreML INT8, ONNX on the CoreML provider) should get the same verdict. "Unstable clips" are
clips whose verdict is not the same for all models; a robust rule keeps that number low.

Usage:
  python scripts/eval_rules.py --set dev --videos path/to/TAD
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from incident_recon.clean import clean_tracks  # noqa: E402
from incident_recon.events import DETECTORS, motion_table  # noqa: E402


def load_clips(split_path, which):
    split = json.loads(Path(split_path).read_text())
    if which == "dev":
        acc, norm = split["dev"]["dev_accident"], split["dev"]["dev_normal"]
    else:
        acc, norm = split["heldout_v2_accident"], split["heldout_v2_normal"]
    return [(c.removesuffix(".mp4"), "accident") for c in acc] + [(c.removesuffix(".mp4"), "normal") for c in norm]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=["dev", "heldout"], default="dev")
    p.add_argument("--videos", required=True, nargs="+", help="Folder(s) with the clips (for frame rates)")
    p.add_argument("--models", nargs="+", default=["pytorch", "coreml16", "coreml8", "onnx_static"])
    p.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    p.add_argument("--tracks", default="outputs/tracks", help="<tracks>/<model>/<clip>_tracks.csv")
    p.add_argument("--split", default="results/split_v2.json")
    p.add_argument("--cache", default="outputs/camera_cache")
    p.add_argument("--out", default=None, help="Optional: write the per-clip verdict table here")
    args = p.parse_args()

    clips = load_clips(args.split, args.set)
    files = {v.stem: v for folder in args.videos for v in Path(folder).expanduser().rglob("*.mp4")}
    rows = []
    for clip, label in clips:
        cap = cv2.VideoCapture(str(files[clip]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        valid = np.load(Path(args.cache) / f"{clip}.npy")
        for model in args.models:
            raw = pd.read_csv(Path(args.tracks) / model / f"{clip}_tracks.csv")
            motion = motion_table(clean_tracks(raw, fps=fps), fps=fps, valid=valid) if not raw.empty else None
            for det in args.detectors:
                hit = motion is not None and len(DETECTORS[det](motion, fps)) > 0
                rows.append({"clip": clip, "label": label, "model": model, "detector": det, "hit": hit})
    t = pd.DataFrame(rows)

    print(f"{args.set}: {sum(lb == 'accident' for _, lb in clips)} accident + "
          f"{sum(lb == 'normal' for _, lb in clips)} normal clips, models: {', '.join(args.models)}\n")
    print("| Detector | Recall (min-max over models) | False-alarm clips (min-max) | Precision (min-max) | Unstable clips |")
    print("|---|---|---|---|---|")
    for det in args.detectors:
        d = t[t["detector"] == det]
        per_model = d.groupby("model").apply(lambda g: pd.Series({
            "tp": int(g.loc[g["label"] == "accident", "hit"].sum()),
            "fp": int(g.loc[g["label"] == "normal", "hit"].sum())}), include_groups=False)
        n_acc = sum(lb == "accident" for _, lb in clips)
        rec = per_model["tp"] / n_acc
        prec = per_model["tp"] / (per_model["tp"] + per_model["fp"]).replace(0, np.nan)
        unstable = int((d.groupby("clip")["hit"].nunique() > 1).sum())
        print(f"| {det} | {rec.min():.2f}-{rec.max():.2f} | {per_model['fp'].min()}-{per_model['fp'].max()} "
              f"| {prec.min():.2f}-{prec.max():.2f} | {unstable} of {len(clips)} |")

    if args.out:
        t.pivot_table(index=["clip", "label"], columns=["detector", "model"], values="hit").to_csv(args.out)


if __name__ == "__main__":
    main()

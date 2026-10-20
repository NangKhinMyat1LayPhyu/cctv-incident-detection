"""Score the crash detector on the held-out split: recall, precision, false alarms per hour.

Input is the per-clip verdict table from the held-out run (one row per clip, a True/False
column per detector version). Normal-clip lengths are read from the videos, to turn
false alarms into a rate per hour of footage. 95% intervals come from a bootstrap
(resample clips, accident and normal separately, 2000 times).

Clip-level scoring: an accident clip counts as detected if the detector raised any event
in it. Event timing is not checked (no crash-time labels yet), and each flagged normal clip
counts as one false alarm (the table only keeps the first event).

Usage:
  python scripts/eval_heldout.py --results results/heldout_v2_results.csv \
      --videos path/to/TAD --out results/heldout_v2_metrics.md
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DETECTORS = {"v1": "v1 (rules A + B)", "v2": "v2 (rule B queue filter)"}


def clip_seconds(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return n / fps


def metrics(acc_hit: np.ndarray, norm_hit: np.ndarray, norm_s: np.ndarray) -> dict:
    tp, fp = acc_hit.sum(), norm_hit.sum()
    return {"recall": tp / len(acc_hit),
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "fa_per_hour": fp / (norm_s.sum() / 3600)}


def bootstrap(acc_hit, norm_hit, norm_s, n: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n):
        a = rng.integers(0, len(acc_hit), len(acc_hit))
        b = rng.integers(0, len(norm_hit), len(norm_hit))
        runs.append(metrics(acc_hit[a], norm_hit[b], norm_s[b]))
    runs = pd.DataFrame(runs)
    return {k: tuple(np.nanpercentile(runs[k], [2.5, 97.5])) for k in runs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="CSV: clip,label,<detector columns>")
    p.add_argument("--videos", required=True, help="Folder with the clips (searched recursively)")
    p.add_argument("--split", default="results/split_v2.json")
    p.add_argument("--out", default="results/heldout_v2_metrics.md")
    args = p.parse_args()

    r = pd.read_csv(args.results)
    split = json.loads(Path(args.split).read_text())
    for label, key in [("accident", "heldout_v2_accident"), ("normal", "heldout_v2_normal")]:
        expected = {c.removesuffix(".mp4") for c in split[key]}
        if set(r.loc[r["label"] == label, "clip"]) != expected:
            sys.exit(f"{label} clips in {args.results} do not match {key} in {args.split}")

    files = {v.stem: v for v in Path(args.videos).expanduser().rglob("*.mp4")}
    normal = r[r["label"] == "normal"].copy()
    missing = sorted(set(normal["clip"]) - set(files))
    if missing:
        sys.exit(f"Videos not found: {missing}")
    normal["seconds"] = [clip_seconds(files[c]) for c in normal["clip"]]
    accident = r[r["label"] == "accident"]
    hours = normal["seconds"].sum() / 3600

    rows, summary = [], {}
    for col, name in DETECTORS.items():
        a, b, s = accident[col].to_numpy(bool), normal[col].to_numpy(bool), normal["seconds"].to_numpy()
        m, ci = metrics(a, b, s), bootstrap(a, b, s)
        summary[col] = {"detected": int(a.sum()), "false_alarm_clips": int(b.sum()),
                        **{k: round(float(v), 3) for k, v in m.items()},
                        **{f"{k}_ci95": [round(float(x), 3) for x in v] for k, v in ci.items()}}
        rows.append(f"| {name} | {a.sum()}/{len(a)} | {m['recall']:.2f} ({ci['recall'][0]:.2f}-{ci['recall'][1]:.2f}) "
                    f"| {m['precision']:.2f} ({ci['precision'][0]:.2f}-{ci['precision'][1]:.2f}) "
                    f"| {b.sum()}/{len(b)} | {m['fa_per_hour']:.0f} ({ci['fa_per_hour'][0]:.0f}-{ci['fa_per_hour'][1]:.0f}) |")

    flagged = normal[normal[list(DETECTORS)].any(axis=1)]
    md = ["# Held-out crash detection (split v2)", "",
          f"{len(accident)} accident clips, {len(normal)} normal clips "
          f"({normal['seconds'].sum() / 60:.1f} min of normal footage). Parameters frozen on the dev set.", "",
          "| Detector | Detected | Recall (95% CI) | Precision (95% CI) | False-alarm clips | False alarms / h (95% CI) |",
          "|---|---|---|---|---|---|", *rows, "",
          "## How it is scored",
          "- Clip level: an accident clip is detected if any event fires in it; event timing is not checked yet.",
          "- Each flagged normal clip counts as one false alarm.",
          "- 95% intervals: bootstrap over clips (2000 resamples, accident and normal resampled separately).", "",
          "## Read with care",
          f"- Only {hours * 60:.1f} min of normal footage: one false alarm moves the rate by "
          f"{1 / hours:.0f}/h. The rate is a rough indication, not a deployment estimate.",
          "- Precision depends on the 40:30 mix; real CCTV has far fewer crashes, so false alarms / h "
          "is the more honest number.",
          "- See `scope_check` in split_v2.json: some normal clips share cameras with dev clips, "
          "and videox14_1/2/3 are one event.", "",
          "## False alarms",
          *[f"- {c}: " + ", ".join(k for k in DETECTORS if f[k]) + (f" - {f['v2_event']}" if isinstance(f.get('v2_event'), str) else "")
            for c, f in flagged.set_index("clip").iterrows()]]

    out = Path(args.out)
    out.write_text("\n".join(md) + "\n")
    out.with_suffix(".json").write_text(json.dumps({"normal_hours": round(hours, 4), **summary}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()

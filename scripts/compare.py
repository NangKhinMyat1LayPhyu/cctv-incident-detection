"""Compare tracking quality of several runs on the same clips.

Usage:
  python scripts/compare.py --videos /path/to/clips \
      --run baseline=outputs/baseline --run finetuned=outputs/finetuned
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from incident_recon.report import run_report  # noqa: E402

METRICS = ["ids", "short_share", "flicker_ids", "stable_vehicles", "stable_people", "proc_fps"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True, help="Folder with the original clips")
    p.add_argument("--run", action="append", required=True, help="NAME=FOLDER, repeat per run")
    p.add_argument("--out", default="comparison.csv")
    args = p.parse_args()

    reports = {}
    for item in args.run:
        name, folder = item.split("=", 1)
        reports[name] = run_report(folder, args.videos)

    # One wide table: for each metric, one column per run
    combined = pd.concat({name: r[METRICS] for name, r in reports.items()}, axis=1)
    combined = combined.swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)
    combined.to_csv(args.out)

    pd.set_option("display.width", 200)
    for metric in METRICS:
        table = combined[metric]
        print(f"\n=== {metric} ===")
        print(table.to_string())
        if metric == "short_share":
            print("mean:", table.mean().round(2).to_dict())
        elif metric == "proc_fps":
            print("mean:", table.mean().round(1).to_dict())
        else:
            print("total:", table.sum().to_dict())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

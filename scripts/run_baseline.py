"""Detect and track vehicles in every video in a folder.

Usage:
  python scripts/run_baseline.py --videos /path/to/crash_clips --out outputs/baseline --limit 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incident_recon.track import track_video  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True, help="Folder of videos (searched recursively)")
    p.add_argument("--out", default="outputs/baseline")
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--limit", type=int, default=10, help="Max videos to process")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.3)
    args = p.parse_args()

    videos = sorted(v for v in Path(args.videos).rglob("*") if v.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"No videos found in {args.videos}")
    videos = videos[: args.limit]
    print(f"Processing {len(videos)} videos...")

    summaries = []
    for v in videos:
        try:
            s = track_video(v, args.out, weights=args.weights, conf=args.conf, imgsz=args.imgsz)
        except Exception as e:  # keep going if one clip is broken
            s = {"video": v.name, "error": str(e)}
        print(json.dumps(s))
        summaries.append(s)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nDone. Summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()

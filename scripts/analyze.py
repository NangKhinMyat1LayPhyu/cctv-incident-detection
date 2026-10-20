"""Analyse one clip end to end: detection + tracking, then the incident report.

Writes to --out: the annotated video, the tracks CSV, and the incident report
(<clip>.md, key-frame images, <clip>_events.json).

Usage:
  python scripts/analyze.py clip.mp4 --weights best.pt --out outputs/analyze
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incident_recon.incident_report import build_incident_report  # noqa: E402
from incident_recon.track import track_video  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="outputs/analyze")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--no-video", action="store_true", help="Skip writing the annotated video")
    args = p.parse_args()

    video = Path(args.video).expanduser()
    if not video.exists():
        sys.exit(f"No such video: {video}")
    print(f"Tracking {video.name}...")
    s = track_video(video, args.out, weights=str(Path(args.weights).expanduser()), conf=args.conf,
                    save_video=not args.no_video)
    print(f"  {s['frames']} frames at {s['processing_fps']} fps on {s['device']}")

    report = build_incident_report(video, s["tracks_csv"], args.out)
    events = json.loads((Path(args.out) / f"{video.stem}_events.json").read_text())
    print(f"{len(events)} incident(s) detected." if events else "No incident detected.")
    for ev in events:
        print(f"  {ev['type']} at {ev['time_s']} s: {ev['detail']}")
    print(f"Report: {report}")
    if s["annotated_video"]:
        print(f"Video:  {s['annotated_video']}")


if __name__ == "__main__":
    main()

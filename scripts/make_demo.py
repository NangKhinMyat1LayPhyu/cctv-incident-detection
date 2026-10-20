"""Render a short demo GIF of an incident: tracked vehicles, then the alert.

Uses the outputs of `incident-recon analyze` (tracks CSV and events JSON). Every tracked
vehicle gets a green box; from the event onwards the involved vehicles turn red and an alert
banner appears. Encoded with an ffmpeg palette, so the GIF stays small but sharp.

Usage:
  incident-recon analyze clip.mp4 --weights best.pt --out outputs/demo --no-video
  python scripts/make_demo.py --video clip.mp4 --analysis outputs/demo --start 15 --end 24 --out docs/demo.gif
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from incident_recon.clean import clean_tracks  # noqa: E402

GREEN, RED, WHITE, BLACK = (60, 200, 60), (40, 40, 230), (255, 255, 255), (0, 0, 0)
TITLES = {"sudden_stop": "Sudden stop", "stopped_vehicles": "Stopped vehicles"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--analysis", required=True, help="Folder with <clip>_tracks.csv and <clip>_events.json")
    p.add_argument("--start", type=float, required=True, help="Seconds")
    p.add_argument("--end", type=float, required=True, help="Seconds")
    p.add_argument("--out", required=True, help=".gif")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--fps", type=int, default=12, help="GIF frame rate")
    p.add_argument("--hold", type=float, default=0.0, help="Freeze the last frame this many seconds (to read the alert)")
    args = p.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is needed to encode the GIF")

    video = Path(args.video).expanduser()
    if not video.exists():
        sys.exit(f"No such video: {video}")
    folder = Path(args.analysis)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracks = clean_tracks(pd.read_csv(folder / f"{video.stem}_tracks.csv"), fps=fps)
    events = json.loads((folder / f"{video.stem}_events.json").read_text())
    by_frame = {f: g for f, g in tracks.groupby("frame")}

    first, last = int(args.start * fps), int(args.end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        writer = None
        for frame in range(first, last + 1):
            ok, img = cap.read()
            if not ok:
                break
            active = [e for e in events if e["frame"] <= frame]
            involved = {t for e in active for t in e["track_ids"]}
            if frame in by_frame:
                for r in by_frame[frame].to_dict("records"):
                    hit = r["track_id"] in involved
                    x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
                    cv2.rectangle(img, (x1, y1), (x2, y2), RED if hit else GREEN, 3 if hit else 1)
                    if hit:
                        cv2.putText(img, f"{r['class']} #{r['track_id']}", (x1, max(y1 - 6, 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
            h, w = img.shape[:2]
            if active:
                e = active[-1]
                cv2.rectangle(img, (0, 0), (w, 30), RED, -1)
                text = f"ALERT  {TITLES.get(e['type'], e['type'])}  {e['detail'].split(' went')[0]}  at {e['time_s']:.1f} s"
            else:
                cv2.rectangle(img, (0, 0), (w, 30), BLACK, -1)
                n = len(by_frame[frame]) if frame in by_frame else 0
                text = f"Monitoring  {frame / fps:5.1f} s   {n} vehicle{'' if n == 1 else 's'} tracked"
            cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(img)
            img_last = img
        cap.release()
        if writer is None:
            sys.exit("no frames in that time range")
        for _ in range(int(args.hold * fps)):    # repeat the last frame: nothing new is drawn
            writer.write(img_last)
        writer.release()

        palette = (f"fps={args.fps},scale={args.width}:-1:flags=lanczos,split[a][b];"
                   "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-vf", palette, "-loop", "0", args.out],
                       check=True)
    print(f"{args.out}: {Path(args.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

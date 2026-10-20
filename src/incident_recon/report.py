"""Measure tracking quality per clip from a run_baseline.py output folder,
so every model can be compared on the same clips in the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd

VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}


def clip_stats(csv_path, video_path, min_track_s: float = 1.0) -> dict:
    """Tracking-quality numbers for one clip."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    row = {"clip": Path(video_path).stem.replace("_x264", ""),
           "duration_s": round(n_frames / fps, 1)}

    t = pd.read_csv(csv_path)
    if t.empty:
        row.update(ids=0, short_share=None, flicker_ids=0, stable_vehicles=0, stable_people=0)
        return row

    # One class per track, by majority vote
    t["class_vote"] = t.groupby("track_id")["class"].transform(lambda s: s.mode()[0])
    tracks = t.groupby("track_id").agg(cls=("class_vote", "first"), frames=("frame", "size"))
    tracks["seconds"] = tracks["frames"] / fps
    stable = tracks[tracks["seconds"] >= min_track_s]

    row.update(
        ids=len(tracks),
        short_share=round(float((tracks["seconds"] < 0.5).mean()), 2),
        flicker_ids=int((t.groupby("track_id")["class"].nunique() > 1).sum()),
        stable_vehicles=int(stable["cls"].isin(VEHICLES).sum()),
        stable_people=int((stable["cls"] == "person").sum()),
    )
    return row


def run_report(run_dir, videos_dir) -> pd.DataFrame:
    """One row per clip for a run folder (needs the run's summary.json)."""
    run_dir, videos_dir = Path(run_dir), Path(videos_dir)
    summary = json.loads((run_dir / "summary.json").read_text())

    rows = []
    for s in summary:
        if "error" in s:
            continue
        video = next(videos_dir.rglob(s["video"]), None)
        if video is None:
            continue
        r = clip_stats(s["tracks_csv"], video)
        r["proc_fps"] = s["processing_fps"]
        rows.append(r)

    report = pd.DataFrame(rows).set_index("clip")
    report.to_csv(run_dir / "report.csv")
    return report

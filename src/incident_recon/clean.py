"""Clean raw tracks before counting, speed and crash detection.

Two problems seen in real footage:
1. Class flicker: the same track is 'car' in one frame and 'truck' in the next.
2. Junk tracks: IDs that live only a few frames (false detections, or a
   vehicle that briefly lost its ID).
"""

from __future__ import annotations

import pandas as pd


def clean_tracks(tracks: pd.DataFrame, fps: float, min_seconds: float = 0.5) -> pd.DataFrame:
    """Return a cleaned copy of a tracks table (columns as in *_tracks.csv)."""
    t = tracks.copy()

    # 1. One class per track: the class with the highest TOTAL confidence.
    score = t.groupby(["track_id", "class"])["conf"].sum()
    best = score.groupby(level="track_id").idxmax().map(lambda key: key[1])
    t["class"] = t["track_id"].map(best)

    # 2. Drop short tracks: fewer than min_seconds worth of frames.
    length = t.groupby("track_id")["frame"].transform("size")
    t = t[length >= min_seconds * fps]

    return t.reset_index(drop=True)

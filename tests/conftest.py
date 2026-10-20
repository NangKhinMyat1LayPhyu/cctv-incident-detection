"""Synthetic tracks for tests: boxes with known positions, so expected answers are known."""

import pandas as pd


def track(tid, frames, xs, ys, cls="car", conf=0.9, size=40.0, fps=10.0):
    """Rows of a tracks table (same columns as *_tracks.csv) for one vehicle.

    xs, ys: bottom-centre of the box per frame (lists, or one number = standing still).
    size:   box width and height in pixels.
    """
    frames = list(frames)
    xs = xs if isinstance(xs, list) else [xs] * len(frames)
    ys = ys if isinstance(ys, list) else [ys] * len(frames)
    cls = cls if isinstance(cls, list) else [cls] * len(frames)
    conf = conf if isinstance(conf, list) else [conf] * len(frames)
    return pd.DataFrame({
        "frame": frames,
        "time_s": [f / fps for f in frames],
        "track_id": tid,
        "class": cls,
        "conf": conf,
        "x1": [x - size / 2 for x in xs],
        "y1": [y - size for y in ys],
        "x2": [x + size / 2 for x in xs],
        "y2": ys,
    })


def tracks(*parts):
    return pd.concat(parts, ignore_index=True)

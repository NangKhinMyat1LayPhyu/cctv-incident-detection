"""Count vehicles crossing a virtual line drawn on the camera view.

Each vehicle is represented by the bottom-centre of its box (where the wheels
touch the road). A vehicle is counted once, when that point moves from one side
of the line to the other.

To ignore jitter (in a jam, boxes wobble a few pixels while the car stands
still), there is a "dead band" of `margin` pixels on both sides of the line.
A point inside the band is too close to call; the vehicle must be clearly on
one side, then clearly on the other side, to be counted.
"""

from __future__ import annotations

import math

import pandas as pd

COLUMNS = ["track_id", "class", "frame", "time_s", "direction"]


def _side(a, b, p) -> float:
    """> 0 if point p is on one side of line a->b, < 0 on the other, 0 on it."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _crosses(a, b, p0, p1) -> bool:
    """True if the movement p0 -> p1 crosses the line segment a-b."""
    # A point exactly ON the line counts as the ">= 0" side, so it is not missed or counted twice.
    return ((_side(a, b, p0) >= 0) != (_side(a, b, p1) >= 0)   # p0, p1 on opposite sides of the line
            and _side(p0, p1, a) * _side(p0, p1, b) <= 0)      # and between the segment's two ends


def _direction(a, b, p0, p1) -> str:
    """Name the crossing direction in screen terms."""
    if abs(b[0] - a[0]) >= abs(b[1] - a[1]):   # line is mostly horizontal
        return "down" if p1[1] > p0[1] else "up"
    return "right" if p1[0] > p0[0] else "left"


def count_crossings(tracks: pd.DataFrame, line, valid=None, margin: float = 0.15) -> pd.DataFrame:
    """One row per vehicle that crossed `line` = [(x1, y1), (x2, y2)].

    valid:  optional bool array (from camera.main_view); frames where the camera
            was moved are ignored, so the line is always on the right road.
    margin: dead band on each side of the line, as a FRACTION OF THE VEHICLE'S
            BOX HEIGHT (anti-jitter). Big nearby vehicles wobble by more pixels
            than small distant ones, so the band scales with the box.
    """
    a, b = line
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    events = []
    for tid, g in tracks.sort_values("frame").groupby("track_id"):
        xs = ((g["x1"] + g["x2"]) / 2).tolist()   # bottom-centre x
        ys = g["y2"].tolist()                     # bottom-centre y
        heights = (g["y2"] - g["y1"]).tolist()    # box height, for the dead band
        frames = g["frame"].tolist()

        side, anchor = 0, None   # last side the vehicle was CLEARLY on (+1 / -1), and where
        for i, f in enumerate(frames):
            if valid is not None and not valid[f]:
                side, anchor = 0, None   # camera moved: earlier positions are not comparable
                continue
            p = (xs[i], ys[i])
            dist = _side(a, b, p) / length   # signed distance to the line, in pixels
            if abs(dist) < margin * heights[i]:
                continue                     # inside the dead band: too close to call
            s = 1 if dist >= 0 else -1       # ">= 0" matches _crosses: ON the line = positive side
            if side != 0 and s != side and _crosses(a, b, anchor, p):
                events.append({
                    "track_id": tid,
                    "class": g["class"].iloc[i],
                    "frame": int(f),
                    "time_s": g["time_s"].iloc[i],
                    "direction": _direction(a, b, anchor, p),
                })
                break   # count each vehicle only once
            side, anchor = s, p
    return pd.DataFrame(events, columns=COLUMNS)

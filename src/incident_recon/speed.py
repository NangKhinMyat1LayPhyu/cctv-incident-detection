"""Estimate vehicle speeds (km/h) from tracks.

Pixels are not metres: because of perspective, a car far away moves only a few
pixels per second, a car near the camera many. A homography (a 3x3 matrix)
maps the image of the flat road onto a top-down "map" of the road in metres.
We fit it from 4 points whose real positions we know (lane-marking dashes).
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def fit_homography(image_pts, road_pts) -> np.ndarray:
    """3x3 matrix mapping image pixels -> road metres, from 4 matching points."""
    return cv2.getPerspectiveTransform(np.float32(image_pts), np.float32(road_pts))


def to_road(H, xs, ys) -> np.ndarray:
    """Map image points (pixels) to road positions (metres). Returns an (N, 2) array."""
    pts = np.float32(np.c_[xs, ys]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def track_speeds(tracks: pd.DataFrame, H, fps: float, valid=None, window_s: float = 1.0) -> pd.DataFrame:
    """Add road position (rx, ry in metres) and speed (km/h) to each row of a tracks table.

    Speed = distance travelled on the road over the last `window_s` seconds / time.
    A one-second window smooths out box jitter; the first second of each track has no speed.
    valid: optional bool array (camera.main_view) - frames with a moved camera are dropped.
    """
    t = tracks.sort_values(["track_id", "frame"]).copy()
    if valid is not None:
        t = t[valid[t["frame"].to_numpy()]]
    road = to_road(H, (t["x1"] + t["x2"]) / 2, t["y2"])   # bottom-centre = where wheels touch the road
    t["rx"], t["ry"] = road[:, 0], road[:, 1]

    win = int(round(window_s * fps))
    t["speed_kmh"] = np.nan
    for _, g in t.groupby("track_id", sort=False):
        f = g["frame"].to_numpy()
        x, y = g["rx"].to_numpy(), g["ry"].to_numpy()
        j = np.searchsorted(f, f - win, side="right") - 1   # row index ~one window earlier
        ok = j >= 0
        metres = np.hypot(x[ok] - x[j[ok]], y[ok] - y[j[ok]])
        seconds = (f[ok] - f[j[ok]]) / fps
        speed = np.full(len(g), np.nan)
        speed[ok] = metres / seconds * 3.6                  # m/s -> km/h
        t.loc[g.index, "speed_kmh"] = speed
    return t


def speed_summary(speeds: pd.DataFrame, stationary_kmh: float = 5.0) -> pd.DataFrame:
    """One row per vehicle: class, median speed, number of measurements, state and direction.

    state:     "stationary" if the median speed is below `stationary_kmh`, else "moving".
    direction: for moving vehicles, "towards" (moving down the image, towards a typical
               roadside camera) or "away". Stationary vehicles get "-": their tiny
               movements are just box wobble, not a real direction.
    """
    s = speeds.dropna(subset=["speed_kmh"])
    summary = (s.groupby("track_id")
                .agg(cls=("class", "first"),
                     median_kmh=("speed_kmh", "median"),
                     n=("speed_kmh", "size"),
                     dy=("y2", lambda y: y.iloc[-1] - y.iloc[0]))
                .round({"median_kmh": 1}))
    moving = summary["median_kmh"] >= stationary_kmh
    summary["state"] = np.where(moving, "moving", "stationary")
    summary["direction"] = np.where(~moving, "-", np.where(summary["dy"] > 0, "towards", "away"))
    return summary.drop(columns="dy")

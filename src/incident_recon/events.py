"""Measure motion cues that may indicate a crash, without any camera calibration.

Speed in km/h needs a calibrated camera (speed.py), and we only have that for
one clip. Here speed is measured in "vehicle sizes per second" instead: the
distance a vehicle moves divided by its own box size. A car crossing its own
length every second is moving; one moving 0.05 sizes per second is standing
still - whether it is near the camera (big box) or far away (small box).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}


def motion_table(tracks: pd.DataFrame, fps: float, valid=None, window_s: float = 0.5) -> pd.DataFrame:
    """Add 'nspeed' (box sizes per second, over the last `window_s`) to every track row."""
    t = tracks.sort_values(["track_id", "frame"]).copy()
    if valid is not None:
        t = t[valid[t["frame"].to_numpy()]]
    t["cx"] = (t["x1"] + t["x2"]) / 2
    t["size"] = np.sqrt((t["x2"] - t["x1"]) * (t["y2"] - t["y1"]))   # side of a square with the box's area

    win = int(round(window_s * fps))
    t["nspeed"] = np.nan
    for _, g in t.groupby("track_id", sort=False):
        f = g["frame"].to_numpy()
        cx, by, sz = g["cx"].to_numpy(), g["y2"].to_numpy(), g["size"].to_numpy()
        j = np.searchsorted(f, f - win, side="right") - 1
        ok = j >= 0
        dist = np.hypot(cx[ok] - cx[j[ok]], by[ok] - by[j[ok]])
        secs = (f[ok] - f[j[ok]]) / fps
        ns = np.full(len(g), np.nan)
        ns[ok] = dist / secs / ((sz[ok] + sz[j[ok]]) / 2)
        t.loc[g.index, "nspeed"] = ns
    return t


def sudden_stops(motion: pd.DataFrame, fps: float, fast: float = 1.0, slow: float = 0.15,
                 within_s: float = 2.0, hold_s: float = 1.0, min_fast_s: float = 0.0) -> pd.DataFrame:
    """Vehicles that go from moving (>= fast) to standing (<= slow) within `within_s`
    seconds, and then stay standing for at least `hold_s` seconds (so a brief wobble
    or a lost box does not count). One row per event.

    min_fast_s: the vehicle must have been fast for at least this long within those
    `within_s` seconds (0 = one frame is enough, as in v1/v2). A single fast frame is often
    box noise, and whether it crosses `fast` changes between runtimes (FP16, INT8, GPU).
    """
    events = []
    for tid, g in motion[motion["class"].isin(VEHICLES)].groupby("track_id"):
        f, ns = g["frame"].to_numpy(), g["nspeed"].to_numpy()
        fast_frames = f[ns >= fast]
        if len(fast_frames) == 0:
            continue
        for k in np.nonzero(ns <= slow)[0]:
            recent_fast = fast_frames[(fast_frames < f[k]) & (fast_frames >= f[k] - within_s * fps)]
            if len(recent_fast) == 0 or len(recent_fast) < min_fast_s * fps:
                continue
            after = (f >= f[k]) & (f <= f[k] + hold_s * fps)
            stays_slow = np.all(ns[after] <= slow * 2) and (f[after][-1] - f[k]) >= hold_s * fps * 0.8
            if stays_slow:
                events.append({"track_id": tid, "class": g["class"].iloc[k], "frame": int(f[k]),
                               "time_s": round(f[k] / fps, 2),
                               "from_nspeed": round(float(ns[f == recent_fast[-1]][0]), 2)})
                break   # one stop per vehicle
    return pd.DataFrame(events, columns=["track_id", "class", "frame", "time_s", "from_nspeed"])


def clip_features(motion: pd.DataFrame, fps: float, min_track_s: float = 1.0) -> dict:
    """A few numbers per clip, to compare accident vs normal clips."""
    lengths = motion.groupby("track_id")["frame"].agg(lambda f: (f.max() - f.min()) / fps)
    stable = lengths[lengths >= min_track_s].index
    m = motion[motion["track_id"].isin(stable)]
    veh = m[m["class"].isin(VEHICLES)]
    med = veh.groupby("track_id")["nspeed"].median()
    return {
        "vehicles": int(med.size),
        "moving": int((med >= 0.3).sum()),
        "stationary": int((med < 0.15).sum()),
        "people": int(m.loc[m["class"] == "person", "track_id"].nunique()),
        "sudden_stops": len(sudden_stops(motion, fps)),
    }
EVENT_COLUMNS = ["type", "time_s", "frame", "track_ids", "detail"]


def stopped_neighbours(veh: pd.DataFrame, stopped_ids, radius: float = 4.0) -> pd.Series:
    """For each stopped vehicle: how many OTHER stopped vehicles were within `radius`
    vehicle sizes of it at the same moment. Many = a queue (red light, jam)."""
    s = veh[veh["track_id"].isin(stopped_ids)].dropna(subset=["nspeed"])[["frame", "track_id", "cx", "y2", "size"]]
    q = s.merge(s, on="frame", suffixes=("_s", "_o"))
    q = q[q["track_id_s"] != q["track_id_o"]]
    near = q[np.hypot(q["cx_s"] - q["cx_o"], q["y2_s"] - q["y2_o"]) / q["size_s"] <= radius]
    return near.groupby("track_id_s")["track_id_o"].nunique().reindex(stopped_ids, fill_value=0)


def _stopped_near(motion: pd.DataFrame, stop: dict, fps: float, slow: float = 0.15, radius: float = 4.0) -> int:
    """How many OTHER vehicles stood still within `radius` vehicle sizes when `stop` happened."""
    at = motion[(motion["frame"] == stop["frame"]) & motion["class"].isin(VEHICLES)].dropna(subset=["nspeed"])
    me = at[at["track_id"] == stop["track_id"]]
    if me.empty:
        return 0
    me = me.iloc[0]
    others = at[(at["track_id"] != stop["track_id"]) & (at["nspeed"] <= slow)]
    return int((np.hypot(others["cx"] - me["cx"], others["y2"] - me["y2"]) / me["size"] <= radius).sum())


def detect_incidents(motion: pd.DataFrame, fps: float, min_track_s: float = 1.0,
                     min_stopped: int = 2, min_stopped_share: float = 0.3,
                     max_queue_neighbours: float | None = None, queue_filter_stops: bool = False,
                     min_fast_s: float = 0.0) -> pd.DataFrame:
    """Incident events in one clip, earliest first. An empty table means "no incident".

    Rule A - sudden stop: a vehicle goes from driving to standing within 2 s (see sudden_stops).
    Rule B - stopped vehicles: at least `min_stopped` vehicles stand still for their whole
             visible time, and they are at least `min_stopped_share` of all vehicles.
             (Traffic at a red light also stands still, but it is usually a small share
             of the vehicles in view; wrecks and their helpers in an aftermath scene are not.)
    max_queue_neighbours: None = detector-v1. A number (v2 uses 6) = rule B is ignored when
             the stopped vehicles form a queue: each one has more than this many other
             stopped vehicles within 4 vehicle sizes (median over the stopped vehicles).
    queue_filter_stops: v3 - the same queue test for rule A: a sudden stop with more than
             `max_queue_neighbours` stopped vehicles within 4 sizes is a car joining a queue.
    min_fast_s: v3 - rule A needs the vehicle fast for at least this long (see sudden_stops).
    """
    events = []

    # Rule A
    for s in sudden_stops(motion, fps, min_fast_s=min_fast_s).to_dict("records"):
        if queue_filter_stops and max_queue_neighbours is not None \
                and _stopped_near(motion, s, fps) > max_queue_neighbours:
            continue
        events.append({"type": "sudden_stop", "time_s": s["time_s"], "frame": s["frame"],
                       "track_ids": [int(s["track_id"])],
                       "detail": f"{s['class']} #{s['track_id']} went from {s['from_nspeed']} "
                                 f"sizes/s to standstill"})

    # Rule B (same vehicle definition as clip_features)
    first = motion.groupby("track_id")["frame"].min()
    last = motion.groupby("track_id")["frame"].max()
    stable = ((last - first) / fps >= min_track_s)
    veh = motion[motion["track_id"].isin(stable[stable].index) & motion["class"].isin(VEHICLES)]
    med = veh.groupby("track_id")["nspeed"].median()
    stopped = med[med < 0.15].index
    enough = len(stopped) >= min_stopped and len(stopped) / max(len(med), 1) >= min_stopped_share
    if enough and max_queue_neighbours is not None:
        # v2: a queue (each stopped vehicle packed between other stopped ones) is not an incident
        enough = stopped_neighbours(veh, stopped).median() <= max_queue_neighbours
    if enough:
        starts = first.loc[stopped].sort_values()
        frame = int(starts.iloc[min_stopped - 1])          # moment the 2nd stopped vehicle is in view
        events.append({"type": "stopped_vehicles", "time_s": round(frame / fps, 2), "frame": frame,
                       "track_ids": [int(t) for t in starts.index],
                       "detail": f"{len(stopped)} of {len(med)} vehicles standing still"})

    return pd.DataFrame(events, columns=EVENT_COLUMNS).sort_values("frame").reset_index(drop=True)


# Detector versions by name: motion table + fps -> events table
DETECTORS = {
    "v1": lambda motion, fps: detect_incidents(motion, fps),
    "v2": lambda motion, fps: detect_incidents(motion, fps, max_queue_neighbours=6),
    # Experimental, not adopted: queue filter for rule A + 0.1 s debounce. On the dev set it made
    # runtimes agree slightly more, at some recall cost.
    "v3": lambda motion, fps: detect_incidents(motion, fps, max_queue_neighbours=6, queue_filter_stops=True,
                                               min_fast_s=0.1),
}

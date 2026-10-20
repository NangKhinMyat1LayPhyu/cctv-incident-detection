"""Turn a clip's tracks into a readable incident report (Markdown + key-frame images).

For each detected incident the report says WHAT happened, WHEN, WHICH vehicles,
shows a key frame with the involved vehicles highlighted, and gives context
(scene summary, camera status, speeds in km/h only if the camera is calibrated).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .camera import main_view
from .clean import clean_tracks
from .events import EVENT_COLUMNS, VEHICLES, detect_incidents, motion_table
from .speed import track_speeds

RED, GREY, WHITE, BLACK = (0, 0, 255), (170, 170, 170), (255, 255, 255), (0, 0, 0)
TITLES = {"sudden_stop": "Sudden stop", "stopped_vehicles": "Stopped vehicles"}


def _ts(seconds: float) -> str:
    """12.34 -> '00:12.3'"""
    m = int(seconds // 60)
    return f"{m:02d}:{seconds - 60 * m:04.1f}"


def _key_frame(video_path, clean, event, fps, out_png) -> int:
    """Save an image of the event: involved vehicles in red (with their path over the
    last 2 s), all other vehicles in grey. Returns the frame number used."""
    ids = set(event["track_ids"])
    later = clean[clean["track_id"].isin(ids) & (clean["frame"] >= event["frame"])]
    # the earliest frame (from the event on) that shows the most involved vehicles
    frame = int(later.groupby("frame").size().idxmax()) if not later.empty else int(event["frame"])

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return frame

    for r in clean[clean["frame"] == frame].to_dict("records"):
        x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
        involved = r["track_id"] in ids
        cv2.rectangle(img, (x1, y1), (x2, y2), RED if involved else GREY, 2 if involved else 1)
        if involved:
            cv2.putText(img, f"#{r['track_id']} {r['class']}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)

    for tid in ids:   # path over the last 2 seconds
        path = clean[(clean["track_id"] == tid) & clean["frame"].between(frame - 2 * fps, frame)]
        pts = np.int32(np.c_[(path["x1"] + path["x2"]) / 2, path["y2"]])
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, RED, 1)

    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 24), BLACK, -1)
    cv2.putText(img, f"{TITLES.get(event['type'], event['type'])} @ {_ts(frame / fps)}",
                (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    cv2.imwrite(str(out_png), img)
    return frame


def _scene(motion, fps, min_track_s: float = 1.0) -> dict:
    """Scene numbers, all from the SAME set of vehicles the detector looks at:
    tracks of 1 s or more, inside the analysed camera view."""
    span = motion.groupby("track_id")["frame"].agg(["min", "max"])
    stable = span.index[(span["max"] - span["min"]) / fps >= min_track_s]
    m = motion[motion["track_id"].isin(stable)]
    veh = m[m["class"].isin(VEHICLES)]
    med = veh.groupby("track_id")["nspeed"].median()
    return {
        "classes": veh.groupby("track_id")["class"].first().value_counts().to_dict(),
        "moving": int((med >= 0.3).sum()),
        "slow": int(((med >= 0.15) & (med < 0.3)).sum()),
        "stopped": int((med < 0.15).sum()),
        "people": int(m.loc[m["class"] == "person", "track_id"].nunique()),
    }


def build_incident_report(video_path, tracks_csv, out_dir, H=None) -> Path:
    """Write <out_dir>/<clip>.md, key-frame images and <clip>_events.json. Returns the .md path.

    H: optional homography (speed.fit_homography) - if given, speeds are reported in km/h.
    """
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    raw = pd.read_csv(tracks_csv)
    if raw.empty:
        clean, events, scene, valid = raw, pd.DataFrame(columns=EVENT_COLUMNS), None, None
    else:
        valid = main_view(video_path)["valid"]
        clean = clean_tracks(raw, fps=fps)
        motion = motion_table(clean, fps=fps, valid=valid)
        events = detect_incidents(motion, fps=fps)
        scene = _scene(motion, fps)

    speeds = track_speeds(clean, H, fps, valid=valid) if (H is not None and not clean.empty) else None

    # ---- header
    verdict = f"**INCIDENT DETECTED** ({len(events)} event{'s' if len(events) != 1 else ''})" \
        if len(events) else "No incident detected"
    md = [f"# Incident report: {name}", "",
          f"**Verdict:** {verdict}  ", f"**Clip length:** {_ts(n_frames / fps)} ({n_frames} frames, {fps:.0f} fps)  "]
    if valid is not None:
        idx = np.nonzero(valid)[0]
        share = len(idx) / max(len(valid), 1)
        md.append(f"**Camera:** analysed view {_ts(idx[0] / fps)}-{_ts(idx[-1] / fps)} "
                  f"({share:.0%} of the clip)" + ("" if share > 0.95 else "; the camera moved, other parts ignored"))
    md.append("")

    # ---- events
    for i, ev in enumerate(events.to_dict("records"), start=1):
        png = f"{name}_event{i}.jpg"
        frame = _key_frame(video_path, clean, ev, fps, out_dir / png)
        md += [f"## {i}. {TITLES.get(ev['type'], ev['type'])}, from {_ts(ev['time_s'])}", "",
               f"{ev['detail']}.  "]
        if ev["type"] == "stopped_vehicles" and valid is not None and ev["frame"] <= np.nonzero(valid)[0][0] + fps:
            md.append("Already standing when the analysed view begins: the incident itself "
                      "happened earlier or out of view.  ")
        md += [
               "Vehicles involved: " + ", ".join(
                   f"{clean.loc[clean.track_id == t, 'class'].iloc[0]} #{t}" for t in ev["track_ids"]) + "  "]
        if speeds is not None:
            before = speeds[speeds["track_id"].isin(ev["track_ids"])
                            & speeds["frame"].between(ev["frame"] - 2 * fps, ev["frame"])]
            kmh = before.groupby("track_id")["speed_kmh"].median().dropna()
            if len(kmh):
                md.append("Speed in the 2 s before: " +
                          ", ".join(f"#{t} {v:.0f} km/h" for t, v in kmh.items()) + "  ")
        md += ["", f"![key frame at {_ts(frame / fps)}]({png})", ""]

    # ---- scene summary
    if scene is not None:
        n_veh = sum(scene["classes"].values())
        md += ["## Scene summary (analysed view, vehicles seen for 1 s or more)", "",
               f"- Vehicles: {n_veh} (" + ", ".join(f"{n} {c}" for c, n in scene["classes"].items()) + ")",
               f"- Moving: {scene['moving']}, slow: {scene['slow']}, standing still: {scene['stopped']}",
               f"- People: {scene['people']}", ""]

    # ---- limitations
    md += ["## Notes", "",
           "- Automatic analysis (detector-v1). Verify with the footage before acting.",
           "- Speeds in km/h: " + ("from a calibrated camera (lane-marking calibration)."
                                   if H is not None else "not available, camera not calibrated."),
           "- Times are from the start of the clip, not wall-clock time.", ""]

    out_md = out_dir / f"{name}.md"
    out_md.write_text("\n".join(md))
    events.to_json(out_dir / f"{name}_events.json", orient="records", indent=1)
    return out_md

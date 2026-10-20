"""Draw tracking results onto videos: counting line + counter, and live speeds."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

RED, GREEN, ORANGE, WHITE, BLACK, GREY = (0, 0, 255), (0, 255, 0), (0, 165, 255), (255, 255, 255), (0, 0, 0), (160, 160, 160)
FLASH_FRAMES = 5  # line turns green for this many frames after a crossing


def render_count_video(video_path, tracks: pd.DataFrame, line, events: pd.DataFrame, out_path,
                       valid=None) -> Path:
    """Write a video showing what the counter saw and counted.

    valid: optional bool array (camera.main_view); frames outside the main view
    get no line and a "camera moved" banner instead.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows_by_frame = {f: g.to_dict("records") for f, g in tracks.groupby("frame")}
    counted_at = dict(zip(events["track_id"], events["frame"]))   # track id -> frame it was counted
    crossing_frames = set(events["frame"])
    a, b = (tuple(int(v) for v in p) for p in line)
    totals: Counter = Counter()

    frame_idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break

        # Update the running totals with crossings that happen in this frame
        for ev in events[events["frame"] == frame_idx].to_dict("records"):
            totals[f"{ev['direction']} {ev['class']}"] += 1

        # The counting line: green just after a crossing, red otherwise.
        # Hidden while the camera is away from its main view (the line would be on the wrong road).
        in_view = valid is None or (frame_idx < len(valid) and valid[frame_idx])
        if in_view:
            flash = any(frame_idx - k in crossing_frames for k in range(FLASH_FRAMES))
            cv2.line(img, a, b, GREEN if flash else RED, 2)
        else:
            cv2.putText(img, "CAMERA MOVED - counting paused", (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        # Each tracked vehicle: box, label, and the bottom-centre point we count with
        for r in rows_by_frame.get(frame_idx, []):
            tid = int(r["track_id"])
            done = tid in counted_at and counted_at[tid] <= frame_idx
            colour = GREEN if done else ORANGE
            x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1)
            cv2.putText(img, f"{tid} {r['class']}", (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
            cv2.circle(img, ((x1 + x2) // 2, y2), 3, colour, -1)

        # Counter panel at the top
        text = "  ".join(f"{k}: {n}" for k, n in sorted(totals.items())) or "count: 0"
        cv2.rectangle(img, (0, 0), (w, 22), BLACK, -1)
        cv2.putText(img, f"total {sum(totals.values())} | {text}", (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        writer.write(img)
        frame_idx += 1

    cap.release()
    writer.release()
    return out_path


def render_speed_video(video_path, speeds: pd.DataFrame, summary: pd.DataFrame, out_path,
                       valid=None) -> Path:
    """Write a video with each vehicle's live speed.

    speeds:  output of speed.track_speeds (one row per vehicle per frame, with speed_kmh).
    summary: output of speed.speed_summary (decides which vehicles are stationary).
    Colours: green = moving, red = stationary, grey = no speed yet (first second of a track).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows_by_frame = {f: g.to_dict("records") for f, g in speeds.groupby("frame")}
    stationary = set(summary.index[summary["state"] == "stationary"])

    frame_idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if valid is not None and not (frame_idx < len(valid) and valid[frame_idx]):
            cv2.putText(img, "CAMERA MOVED - speed paused", (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        n_moving, n_stopped = 0, 0
        for r in rows_by_frame.get(frame_idx, []):
            tid, kmh = int(r["track_id"]), r["speed_kmh"]
            if tid in stationary:
                colour, label = RED, f"{tid} STOPPED"
                n_stopped += 1
            elif pd.isna(kmh):
                colour, label = GREY, f"{tid} {r['class']}"
            else:
                colour, label = GREEN, f"{tid} {kmh:.0f} km/h"
                n_moving += 1
            x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1)
            cv2.putText(img, label, (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

        cv2.rectangle(img, (0, 0), (w, 22), BLACK, -1)
        cv2.putText(img, f"moving {n_moving} | stopped {n_stopped}", (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        writer.write(img)
        frame_idx += 1

    cap.release()
    writer.release()
    return out_path

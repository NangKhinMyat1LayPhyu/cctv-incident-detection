"""Run the incident detector on a live stream, on a rolling window of recent tracks.

The offline detector (events.detect_incidents) looks at a finished clip. Live, there is
no "whole clip", so we keep the last `window_s` seconds of tracks and run the SAME
cleaning, motion cues and rules on that window every `check_every_s` seconds. Keeping
the rules identical means the held-out results describe the live detector's logic too.

Running the same rules again every second finds the same incident again, so:
- a sudden stop is alerted once per vehicle;
- "stopped vehicles" is alerted at most once every `repeat_after_s` seconds per camera
  (an aftermath scene stays an aftermath scene for many minutes).
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from .clean import clean_tracks
from .events import detect_incidents, motion_table

TRACK_COLUMNS = ["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2"]


class IncidentMonitor:
    def __init__(self, fps: float, window_s: float = 20.0, check_every_s: float = 1.0,
                 min_view_s: float = 3.0, repeat_after_s: float = 60.0,
                 max_queue_neighbours: float | None = 6):
        """max_queue_neighbours: 6 = detector v2 (queue filter), None = v1."""
        self.fps = fps
        self.window = int(window_s * fps)
        self.check_every = max(int(check_every_s * fps), 1)
        self.min_view = int(min_view_s * fps)
        self.repeat_after = int(repeat_after_s * fps)
        self.max_queue_neighbours = max_queue_neighbours
        self.rows: deque[tuple] = deque()
        self.tracks = pd.DataFrame(columns=TRACK_COLUMNS)   # cleaned window at the last check
        self.last_stopped_alert = None
        self.reset(0)

    def reset(self, frame: int) -> None:
        """Forget the tracks seen so far (camera moved, or the stream was reconnected).

        The "stopped vehicles" repeat limit is kept: operators steer the camera towards an
        incident, and the new view of the same aftermath should not raise a new alert."""
        self.rows.clear()
        self.view_start = frame
        self.next_check = frame
        self.ran = False
        self.alerted_tracks: set[int] = set()

    def add(self, frame: int, boxes) -> None:
        """boxes: (track_id, class, conf, x1, y1, x2, y2) for every tracked object in this frame."""
        self.rows.extend((frame, *b) for b in boxes)

    def check(self, frame: int) -> list[dict]:
        """Run the detector if it is time to. Returns NEW events only (absolute frame numbers).
        Afterwards, self.ran says whether the detector actually ran."""
        self.ran = False
        if frame < self.next_check:
            return []
        self.next_check = frame + self.check_every
        if frame - self.view_start < self.min_view:    # too little of this view seen yet
            return []

        while self.rows and self.rows[0][0] < frame - self.window:
            self.rows.popleft()
        if not self.rows:
            return []
        self.ran = True

        # The rules expect frames counted from the start of the clip: shift, then shift back.
        raw = pd.DataFrame(list(self.rows), columns=TRACK_COLUMNS)
        start = int(raw["frame"].min())
        raw["frame"] -= start
        clean = clean_tracks(raw, fps=self.fps)
        if clean.empty:
            return []
        events = detect_incidents(motion_table(clean, fps=self.fps), fps=self.fps,
                                  max_queue_neighbours=self.max_queue_neighbours)
        clean["frame"] += start
        self.tracks = clean

        new = []
        for ev in events.to_dict("records"):
            ev["frame"] += start
            ev["time_s"] = round(ev["frame"] / self.fps, 2)
            if ev["type"] == "sudden_stop":
                if ev["track_ids"][0] in self.alerted_tracks:
                    continue
                self.alerted_tracks.add(ev["track_ids"][0])
            elif ev["type"] == "stopped_vehicles":
                if self.last_stopped_alert is not None and frame - self.last_stopped_alert < self.repeat_after:
                    continue
                self.last_stopped_alert = frame
            new.append(ev)
        return new

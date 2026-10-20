"""Save evidence when the live detector raises an incident.

For every alert, in <out_dir>:
  <id>.mp4      clip from `pre_s` seconds before the event to `post_s` seconds after the alert
  <id>.jpg      key frame: involved vehicles in red, others in grey
  <id>.json     the event (also appended to events.jsonl, one line per alert)
and, if a webhook URL is given, the event is POSTed to it as JSON.

The clip needs frames from BEFORE the alert, so the last `buffer_s` seconds of video are
kept in memory. Frames are shrunk to at most `max_width` pixels wide to bound that memory
(10 s of 960x540 video at 25 fps is about 390 MB).
"""

from __future__ import annotations

import json
import threading
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

RED, GREY, WHITE, BLACK = (0, 0, 255), (170, 170, 170), (255, 255, 255), (0, 0, 0)
TITLES = {"sudden_stop": "Sudden stop", "stopped_vehicles": "Stopped vehicles"}


def _post(url: str, payload: dict) -> None:
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).close()
    except Exception as e:   # an unreachable webhook must not stop the camera
        print(f"  webhook failed: {e}")


class AlertWriter:
    def __init__(self, out_dir, fps: float, camera: str = "cam1", pre_s: float = 5.0,
                 post_s: float = 5.0, buffer_s: float = 10.0, max_width: int = 960,
                 webhook: str | None = None, run: str | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps, self.camera, self.webhook, self.max_width, self.run = fps, camera, webhook, max_width, run
        self.pre, self.post = int(pre_s * fps), int(post_s * fps)
        self.frames: deque[tuple[int, object, float]] = deque(maxlen=max(int(buffer_s * fps), self.pre))
        self.open_clips: list[dict] = []
        self.count = 0

    def add_frame(self, frame: int, img) -> None:
        """Call for every processed frame."""
        h, w = img.shape[:2]
        scale = min(1.0, self.max_width / w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        self.frames.append((frame, img, scale))
        for clip in self.open_clips:
            clip["writer"].write(img)
            clip["left"] -= 1
        for clip in [c for c in self.open_clips if c["left"] <= 0]:
            clip["writer"].release()
            self.open_clips.remove(clip)

    def alert(self, event: dict, tracks: pd.DataFrame, now_frame: int) -> dict:
        """Save clip, key frame and JSON for one event. Returns the stored record."""
        self.count += 1
        when = datetime.now(timezone.utc)
        alert_id = f"{self.camera}_{when:%Y%m%d_%H%M%S}_{self.count:03d}_{event['type']}"
        record = {"id": alert_id, "run": self.run, "camera": self.camera, "alert_utc": when.isoformat(timespec="seconds"),
                  "type": event["type"], "event_frame": int(event["frame"]),
                  "event_stream_s": round(event["frame"] / self.fps, 2),
                  "alert_stream_s": round(now_frame / self.fps, 2),
                  "track_ids": [int(t) for t in event["track_ids"]], "detail": event["detail"],
                  "clip": f"{alert_id}.mp4", "key_frame": f"{alert_id}.jpg"}

        if self.frames:
            self._key_frame(event, tracks, self.out_dir / record["key_frame"])
            self._start_clip(event["frame"], self.out_dir / record["clip"])

        (self.out_dir / f"{alert_id}.json").write_text(json.dumps(record, indent=2))
        with open(self.out_dir / "events.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  ALERT [{self.camera}] {record['type']} at {record['event_stream_s']} s "
              f"(raised at {record['alert_stream_s']} s): {record['detail']}")
        if self.webhook:
            threading.Thread(target=_post, args=(self.webhook, record), daemon=True).start()
        return record

    def close(self) -> None:
        for clip in self.open_clips:
            clip["writer"].release()
        self.open_clips.clear()

    def _start_clip(self, event_frame: int, path: Path) -> None:
        before = [img for f, img, _ in self.frames if f >= event_frame - self.pre]
        before = before or [self.frames[-1][1]]
        h, w = before[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for img in before:
            writer.write(img)
        self.open_clips.append({"writer": writer, "left": self.post})

    def _key_frame(self, event: dict, tracks: pd.DataFrame, path: Path) -> None:
        """The buffered frame closest to the event that shows the involved vehicles."""
        ids = set(event["track_ids"])
        buffered = {f: (img, s) for f, img, s in self.frames}
        shown = tracks[tracks["track_id"].isin(ids) & tracks["frame"].isin(buffered.keys())]
        if not shown.empty:
            later = shown[shown["frame"] >= event["frame"]]
            frame = int((later if not later.empty else shown)["frame"].iloc[0])
        else:
            frame = self.frames[-1][0]
        img, s = buffered[frame]
        img = img.copy()

        for r in tracks[tracks["frame"] == frame].to_dict("records"):
            x1, y1, x2, y2 = (int(r[k] * s) for k in ("x1", "y1", "x2", "y2"))
            involved = r["track_id"] in ids
            cv2.rectangle(img, (x1, y1), (x2, y2), RED if involved else GREY, 2 if involved else 1)
            if involved:
                cv2.putText(img, f"#{r['track_id']} {r['class']}", (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)

        w = img.shape[1]
        cv2.rectangle(img, (0, 0), (w, 24), BLACK, -1)
        cv2.putText(img, f"{self.camera}  {TITLES.get(event['type'], event['type'])} @ {frame / self.fps:.1f} s",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        cv2.imwrite(str(path), img)

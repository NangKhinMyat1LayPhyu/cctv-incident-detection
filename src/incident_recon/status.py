"""Live status for the dashboard: what the cameras are doing right now.

The live process writes, every `every_s` seconds, into its alerts folder:
  status.json           per camera: processing fps, frames, dropped frames, reconnects,
                        camera moves, alerts, seconds since the last frame; plus cycle latency
  latest/<camera>.jpg   the newest annotated frame of each camera
  main/<camera>.jpg     the newest frame of the camera's MAIN view (its longest steady view):
                        shown once the camera has stopped, or is looking somewhere else
The web service (service.py) only READS these files. The two processes share nothing else,
so either can restart without stopping the other.

Files are written to a temporary name and then renamed, so a reader never sees half a file.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class StatusWriter:
    def __init__(self, out_dir, run: str | None = None, every_s: float = 2.0):
        self.out_dir = Path(out_dir)
        self.run = run
        for sub in ("latest", "main"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)
        self.every_s = every_s
        self.started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.last_write = 0.0
        self.prev_frames: dict[str, int] = {}

    def _jpg(self, folder: str, name: str, img) -> str | None:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        write_atomic(self.out_dir / folder / f"{name}.jpg", buf.tobytes())
        return f"{folder}/{name}.jpg"

    def due(self) -> bool:
        return time.monotonic() - self.last_write >= self.every_s

    def write(self, cams, latest: dict, cycle_ms: list[float], running: bool = True) -> None:
        """cams: LiveCamera objects; latest: camera name -> newest frame; cycle_ms: recent cycle times."""
        now = time.monotonic()
        dt = now - self.last_write if self.last_write else None
        self.last_write = now
        cameras = []
        for c in cams:
            fps = (c.frames - self.prev_frames.get(c.name, 0)) / dt if dt else None
            self.prev_frames[c.name] = c.frames
            jpg = self._jpg("latest", c.name, c.draw(latest[c.name])) if c.name in latest else None
            main_jpg = self._jpg("main", c.name, c.draw(c.main_frame, c.main_boxes)) if c.main_frame is not None else None
            ended = c.reader.ended or not running
            cameras.append({
                "name": c.name, "live": c.reader.live, "stream_fps": round(c.fps, 1),
                "processed_fps": round(fps, 1) if fps is not None and not ended else None,
                "frames": c.frames, "dropped": c.reader.dropped, "reconnects": c.reader.reconnects,
                "camera_moves": c.camera_moves, "alerts": c.alerts.count,
                "ended": ended, "on_main_view": c.on_main_view, "latest_jpg": jpg, "main_jpg": main_jpg,
            })
        status = {
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "started_utc": self.started, "run": self.run, "running": running, "cameras": cameras,
            "cycle_ms_mean": round(float(np.mean(cycle_ms)), 1) if cycle_ms else None,
            "cycle_ms_p95": round(float(np.percentile(cycle_ms, 95)), 1) if cycle_ms else None,
        }
        write_atomic(self.out_dir / "status.json", json.dumps(status, indent=1).encode())

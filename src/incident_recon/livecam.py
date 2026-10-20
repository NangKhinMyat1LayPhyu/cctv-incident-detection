"""Everything one live camera needs, so that one process can serve several cameras.

The detector is SHARED: frames from all cameras are detected as one batch (one GPU call
instead of one per camera). Everything with memory is PER CAMERA: tracker, camera guard,
incident detector and alerts - track IDs and history must never mix between cameras.

Ultralytics' model.track() keeps one tracker per model, so tracking is done here with
our own ByteTrack per camera, fed with the boxes from the shared detector.
"""

from __future__ import annotations

import time

import cv2
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from .alerts import AlertWriter
from .camera import CameraWatch
from .online import IncidentMonitor
from .stream import FrameReader

GREEN, BLACK = (0, 200, 0), (0, 0, 0)


def bytetrack_config() -> IterableSimpleNamespace:
    return IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))


class LiveCamera:
    def __init__(self, name: str, source, out_dir, detector: str = "v2", webhook: str | None = None,
                 run: str | None = None, realtime: bool = False):
        self.name = name
        self.reader = FrameReader(source, realtime=realtime)
        self.fps = self.reader.fps
        self.guard = CameraWatch()
        self.monitor = IncidentMonitor(self.fps, max_queue_neighbours=6 if detector == "v2" else None)
        self.alerts = AlertWriter(out_dir, self.fps, camera=name, webhook=webhook, run=run)
        self._config = bytetrack_config()
        self.tracker = self._new_tracker()
        self.seen_reconnects = 0
        self.camera_moves = 0
        self.frames = 0
        self.check_ms: list[float] = []
        self.boxes: list[tuple] = []     # tracked boxes in the last frame, for drawing
        # The camera's MAIN view = its longest steady view so far (as camera.main_view does offline).
        # Its newest frame is kept, so the dashboard can show the road, not a view the operator
        # steered to (or a TV cut-away), once the camera has moved or stopped.
        self.main_frame, self.main_boxes, self.main_view_frames = None, [], 0
        self.on_main_view = False

    def _new_tracker(self) -> BYTETracker:
        # A NEW tracker, not tracker.reset(): reset() also restarts the track-ID counter,
        # which all trackers share, so another camera would start reusing its live IDs.
        return BYTETracker(self._config, frame_rate=round(self.fps))

    def update(self, frame_no: int, img, result, names: dict) -> list[dict]:
        """Process one frame given its detections. Returns the alerts it raised."""
        if self.reader.reconnects != self.seen_reconnects or self.guard.moved(img):
            if self.reader.reconnects == self.seen_reconnects:
                self.camera_moves += 1
            self.seen_reconnects = self.reader.reconnects
            self.monitor.reset(frame_no)        # old positions mean nothing in the new view
            self.tracker = self._new_tracker()

        tracks = self.tracker.update(result.boxes.cpu().numpy(), img)   # rows: x1 y1 x2 y2 id conf cls idx
        self.boxes = [(int(t[4]), names[int(t[6])], round(float(t[5]), 3), *(round(float(v), 1) for v in t[:4]))
                      for t in tracks]
        self.monitor.add(frame_no, self.boxes)
        view_frames = frame_no - self.monitor.view_start
        self.on_main_view = view_frames >= self.main_view_frames
        if self.on_main_view:
            self.main_frame, self.main_boxes, self.main_view_frames = img, self.boxes, view_frames
        self.alerts.add_frame(frame_no, img)

        t = time.perf_counter()
        events = self.monitor.check(frame_no)
        if self.monitor.ran:
            self.check_ms.append((time.perf_counter() - t) * 1000)
        self.frames += 1
        return [self.alerts.alert(ev, self.monitor.tracks, frame_no) for ev in events]

    def draw(self, img, boxes=None):
        """Copy of the frame with the tracked boxes (default: the last frame's) and IDs."""
        img = img.copy()
        for tid, cls, _, x1, y1, x2, y2 in self.boxes if boxes is None else boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 1)
            cv2.putText(img, f"#{tid} {cls}", (int(x1), max(int(y1) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
        cv2.rectangle(img, (0, 0), (img.shape[1], 20), BLACK, -1)
        cv2.putText(img, f"{self.name}  alerts: {self.alerts.count}", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
        return img

    def close(self) -> None:
        self.reader.close()
        self.alerts.close()

"""Read frames from a live camera (RTSP, webcam) or a video file, in a background thread.

Live sources: only the NEWEST frame is kept. If processing is slower than the camera,
older frames are dropped instead of queuing up, so the delay between the camera and
the detector stays bounded (a queue would grow without limit). If the connection is
lost, the reader reconnects. Frame numbers follow the clock (elapsed time x fps), so
dropped frames and outages show up as gaps and speeds stay correct.

RTSP is read over TCP, with open and read timeouts: without them a dead stream blocks
`read()` for minutes and the reconnect never happens. The reader thread owns the capture
object (only it reads and releases it), so stopping never collides with a blocked read.

Video files: every frame is handed over in order, nothing dropped, so runs repeat exactly.
With realtime=True a file is played at its own frame rate, like a camera (for demos).
"""

from __future__ import annotations

import os
import queue
import threading
import time

import cv2

TIMEOUT_S = 5.0     # give up opening / reading a live stream after this long


def is_live(source) -> bool:
    return isinstance(source, int) or str(source).lower().startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://"))


def open_capture(source) -> cv2.VideoCapture:
    """VideoCapture for a file, webcam or stream; streams get TCP and timeouts."""
    if isinstance(source, int) or not is_live(source):
        return cv2.VideoCapture(source)
    if str(source).lower().startswith(("rtsp://", "rtsps://")):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")   # UDP first is slower
    ms = int(TIMEOUT_S * 1000)
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, ms])


class FrameReader:
    def __init__(self, source, reconnect_s: float = 2.0, realtime: bool = False):
        self.source, self.reconnect_s, self.realtime = source, reconnect_s, realtime
        self.live = is_live(source)
        self.cap = open_capture(source)
        if not self.cap.isOpened():
            raise ConnectionError(f"Cannot open source: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not 1 <= self.fps <= 120:   # some streams report nonsense
            self.fps = 25.0
        self.reconnects = 0
        self.received = 0              # frames read from the source
        self.delivered = 0             # frames handed to the caller
        self.ended = False
        self._stop = threading.Event()
        self._latest = None
        self._cond = threading.Condition()
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._run_live if self.live else self._run_file, daemon=True)
        self._thread.start()

    @property
    def dropped(self) -> int:
        """Frames thrown away because processing was slower than the camera (live only)."""
        if not self.live:
            return 0   # files queue every frame
        return self.received - self.delivered - (self._latest is not None)

    def read(self, timeout: float = 10.0):
        """(frame_number, image), or None when the stream ended or nothing arrived in time."""
        if not self.live:
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                return None
            if item is not None:
                self.delivered += 1
            return item
        with self._cond:
            if not self._cond.wait_for(lambda: self._latest is not None or self._stop.is_set(), timeout):
                return None
            item, self._latest = self._latest, None
        if item is not None:
            self.delivered += 1
        return item

    def close(self) -> None:
        """Stop reading. The reader thread releases the capture itself, within one read timeout."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=TIMEOUT_S + 1)

    def _run_file(self) -> None:
        frame, t0 = 0, time.monotonic()
        while not self._stop.is_set():
            ok, img = self.cap.read()
            if not ok:
                break
            if self.realtime:                # wait until this frame's moment in the clip
                time.sleep(max(0.0, t0 + frame / self.fps - time.monotonic()))
            self.received += 1
            while not self._stop.is_set():   # wait for room: files are never dropped
                try:
                    self._queue.put((frame, img), timeout=0.5)
                    break
                except queue.Full:
                    pass
            frame += 1
        self.cap.release()
        self.ended = True
        self._queue.put(None)

    def _run_live(self) -> None:
        t0, last = time.monotonic(), -1
        while not self._stop.is_set():
            ok, img = self.cap.read()
            if not ok:
                if self._stop.is_set():
                    break
                print(f"Stream lost ({self.source}), reconnecting in {self.reconnect_s:.0f} s...", flush=True)
                self.cap.release()
                if self._stop.wait(self.reconnect_s):     # returns early when closing
                    break
                self.cap = open_capture(self.source)
                self.reconnects += 1
                continue
            self.received += 1
            last = max(last + 1, round((time.monotonic() - t0) * self.fps))
            with self._cond:
                self._latest = (last, img)    # overwrites an unprocessed older frame
                self._cond.notify()
        self.cap.release()
        self.ended = True

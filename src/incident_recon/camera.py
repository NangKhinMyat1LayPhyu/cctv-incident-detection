"""Detect when a CCTV camera moves (pan, tilt, zoom) away from its main view.

Operators often steer the camera towards a crash. Counting lines, speeds and
crash cues are only valid while the camera shows the view they were set up on,
so we find that "main view" and mark which frames show it.

Two ideas make this robust:
1. Compare each frame with the PREVIOUS frame (1/25 s earlier). In that short
   time traffic barely changes the picture, but a camera move changes all of it.
2. Cut the picture into a 3x3 grid of tiles and let the tiles VOTE. A big car
   moving changes one or two tiles; a camera move changes most of them.
"""

from __future__ import annotations

import cv2
import numpy as np

SMALL_W = 240  # frames are shrunk to this width: fast, and enough to see the whole scene
GRID = 3       # 3x3 tiles


def _small(img) -> np.ndarray:
    """One BGR frame as a small grayscale float image."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (SMALL_W, int(h * SMALL_W / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _small_frames(video_path) -> np.ndarray:
    """All frames as small grayscale float images."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(_small(img))
    cap.release()
    return np.array(frames)


def _tile_window(shape) -> np.ndarray:
    h, w = shape
    return np.outer(np.hanning(h // GRID), np.hanning(w // GRID))   # fades tile edges


def _phase_correlate(a, b, window) -> tuple[float, float, float]:
    """How far image b is slid relative to a: (dx, dy, response).

    Phase correlation: compare the two images in the frequency domain; the result
    is a map with one sharp peak at the shift between them. A high peak (response
    near 1) means the images match well once shifted; a low peak means the scene
    itself changed (zoom, or the camera looks somewhere else).
    """
    fa = np.fft.fft2((a - a.mean()) * window)
    fb = np.fft.fft2((b - b.mean()) * window)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.ifft2(cross).real
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape
    dy = dy - h if dy > h // 2 else dy   # shifts past halfway are negative shifts
    dx = dx - w if dx > w // 2 else dx
    return float(dx), float(dy), float(corr.max())


def _camera_moved(prev, cur, window, min_response) -> bool:
    """Tiles vote: did the camera move between two consecutive frames?"""
    h, w = prev.shape
    th, tw = h // GRID, w // GRID
    shifted, responses = 0, []
    for r in range(GRID):
        for c in range(GRID):
            tile = (slice(r * th, (r + 1) * th), slice(c * tw, (c + 1) * tw))
            dx, dy, response = _phase_correlate(prev[tile], cur[tile], window)
            shifted += dx != 0 or dy != 0
            responses.append(response)
    most_tiles_slid = shifted > GRID * GRID / 2              # pan / tilt / jump
    poor_match = np.median(responses) < min_response         # zoom / cut to another view
    return most_tiles_slid or poor_match


def main_view(video_path, min_response: float = 0.3) -> dict:
    """Find the camera's main view and which frames show it.

    Returns {"ref_frame": first frame of the main view,
             "valid": bool array, True where the frame shows the main view,
             "moving": bool array, True where the camera moved since the previous frame}.
    """
    frames = _small_frames(video_path)
    window = _tile_window(frames[0].shape)

    # 1. Did the camera move between each frame and the one before it?
    moving = np.zeros(len(frames), dtype=bool)
    for i in range(1, len(frames)):
        moving[i] = _camera_moved(frames[i - 1], frames[i], window, min_response)

    # 2. The main view is the longest stretch with no movement.
    best, start = (0, 0), 0
    for i in range(len(frames) + 1):
        if i == len(frames) or moving[i]:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = i + 1

    valid = np.zeros(len(frames), dtype=bool)
    valid[best[0]:best[1]] = True
    return {"ref_frame": best[0], "valid": valid, "moving": moving}

class CameraWatch:
    """Live version of the camera guard: frames arrive one at a time.

    main_view() looks at the whole clip and keeps the longest steady stretch. Live, we
    cannot look ahead, so the analysed view is simply the CURRENT one: moved() says when
    the camera moves, and the caller then throws away what it saw before.
    """

    def __init__(self, min_response: float = 0.3):
        self.min_response = min_response
        self.prev = None
        self.window = None

    def moved(self, img) -> bool:
        """Did the camera move since the previous frame?"""
        cur = _small(img)
        if self.prev is None or self.prev.shape != cur.shape:   # first frame, or new stream size
            self.prev, self.window = cur, _tile_window(cur.shape)
            return False
        moved = _camera_moved(self.prev, cur, self.window, self.min_response)
        self.prev = cur
        return moved

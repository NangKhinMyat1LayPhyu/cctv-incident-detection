"""Detect incidents on one or more LIVE sources, frame by frame, as they arrive.

A source can be an RTSP camera (rtsp://...), a webcam (0) or a video file (for testing),
optionally named: --source north=rtsp://... Repeat --source for more cameras.

Each cycle takes the newest frame from every camera, detects them as ONE batch on the
shared model, then each camera runs its own tracker, camera guard, rolling-window incident
detector and alerts (see incident_recon/livecam.py). Alerts from all cameras go to --out;
events.jsonl lists them all. status.json and latest/<camera>.jpg in --out feed the dashboard.

Live sources drop frames rather than fall behind, and reconnect if the stream is lost.
Video files are read frame by frame without drops, so test runs are repeatable.

Usage:
  python scripts/live.py --source rtsp://localhost:8554/cam1 --weights best.pt --seconds 0
  python scripts/live.py --source north=rtsp://.../cam1 --source south=rtsp://.../cam2 --weights best.pt
  python scripts/live.py --source a.mp4 --source b.mp4 --weights best.pt --sequential   # compare: no batching
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from incident_recon.device import get_device  # noqa: E402
from incident_recon.livecam import LiveCamera  # noqa: E402
from incident_recon.models import batch_one_only  # noqa: E402
from incident_recon.status import StatusWriter  # noqa: E402
from incident_recon.track import PERSON_NAMES, VEHICLE_NAMES  # noqa: E402


def parse_source(text: str, i: int) -> tuple[str, object]:
    """'north=rtsp://...' -> ('north', 'rtsp://...'); 'rtsp://...' -> ('cam1', ...); '0' -> webcam 0."""
    name, sep, src = text.partition("=")
    if not sep or "://" in name:
        name, src = f"cam{i + 1}", text
    return name, int(src) if src.isdigit() else src


def ms_stats(values) -> str:
    v = np.array(values)
    return f"mean {v.mean():.1f} ms, 95% under {np.percentile(v, 95):.1f} ms" if len(v) else "-"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="append", required=True,
                   help="[name=]rtsp://... URL, webcam number or video file; repeat for more cameras")
    p.add_argument("--weights", required=True)
    p.add_argument("--seconds", type=float, default=30, help="Stop after this many seconds (0 = run until stopped)")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--out", default="outputs/live", help="Folder for alert clips, key frames and events.jsonl")
    p.add_argument("--webhook", default=None, help="Optional: POST every alert as JSON to this URL")
    p.add_argument("--detector", choices=["v1", "v2"], default="v2")
    p.add_argument("--sequential", action="store_true", help="Detect one camera at a time (no batching), for comparison")
    p.add_argument("--save", default=None, help="Optional: folder for annotated videos, one per camera")
    p.add_argument("--realtime", action="store_true", help="Play video files at their own speed, like a camera (demos)")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # alerts reach a log file at once, not when the run ends

    model = YOLO(args.weights, task="detect")
    device = get_device()
    if batch_one_only(args.weights) and not args.sequential:
        print("This model takes one image per call (CoreML / fixed-shape ONNX): detecting cameras one at a time.")
        args.sequential = True
    classes = [i for i, name in model.names.items() if name in VEHICLE_NAMES | PERSON_NAMES]

    run = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")   # tags this run's alerts and status
    cams = []
    for i, text in enumerate(args.source):
        name, src = parse_source(text, i)
        try:
            cams.append(LiveCamera(name, src, args.out, detector=args.detector, webhook=args.webhook, run=run,
                                   realtime=args.realtime))
        except ConnectionError as e:
            for c in cams:
                c.close()
            sys.exit(str(e))
        c = cams[-1]
        print(f"[{name}] connected to {src} ({c.fps:.0f} fps, {'live' if c.reader.live else 'file'})")
    print(f"{len(cams)} camera(s), device {device}, detector {args.detector}, "
          f"{'sequential' if args.sequential else 'batched'} detection. Alerts -> {args.out}. "
          + (f"Running {args.seconds:.0f} s..." if args.seconds else "Running until Ctrl+C..."))

    writers = {}
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    detect_ms, cycle_ms, batch_sizes = [], [], []
    status, latest = StatusWriter(args.out, run=run), {}      # for the dashboard (service.py)
    active = list(cams)
    t_start = last_print = time.perf_counter()
    try:
        while active and (not args.seconds or time.perf_counter() - t_start < args.seconds):
            # 1. The newest frame from every camera that has one (files: the next frame, waiting for it)
            batch = []
            for cam in list(active):
                item = cam.reader.read(timeout=0 if cam.reader.live else 10)
                if item is None:
                    if cam.reader.ended:
                        print(f"[{cam.name}] stream ended.")
                        active.remove(cam)
                    continue
                batch.append((cam, *item))
            if not batch:
                time.sleep(0.002)    # live cameras: no new frame yet
                continue

            # 2. Detect all frames on the shared model, 3. then each camera tracks and checks
            t = time.perf_counter()
            imgs = [img for _, _, img in batch]
            kw = dict(classes=classes, conf=args.conf, device=device, verbose=False)
            results = [model.predict(img, **kw)[0] for img in imgs] if args.sequential else model.predict(imgs, **kw)
            detect_ms.append((time.perf_counter() - t) * 1000)
            for (cam, frame_no, img), result in zip(batch, results):
                cam.update(frame_no, img, result, model.names)
                latest[cam.name] = img
                if args.save:
                    out = cam.draw(img)
                    if cam.name not in writers:
                        h, w = out.shape[:2]
                        writers[cam.name] = cv2.VideoWriter(str(Path(args.save) / f"{cam.name}.mp4"),
                                                            cv2.VideoWriter_fourcc(*"mp4v"), cam.fps, (w, h))
                    writers[cam.name].write(out)
            cycle_ms.append((time.perf_counter() - t) * 1000)
            batch_sizes.append(len(batch))
            if status.due():
                status.write(cams, latest, cycle_ms[-100:])

            if time.perf_counter() - last_print >= 5:
                last_print = time.perf_counter()
                el = last_print - t_start
                print("  " + ", ".join(f"{c.name}: {c.frames / el:.1f} fps, {c.reader.dropped} dropped, "
                                       f"{c.alerts.count} alerts" for c in cams))
    except KeyboardInterrupt:
        print("\nStopped.")

    elapsed = time.perf_counter() - t_start
    status.write(cams, latest, cycle_ms[-100:], running=False)
    for cam in cams:
        cam.close()
    for w in writers.values():
        w.release()

    warmup = 5 if len(cycle_ms) > 20 else 0    # the first cycles include model start-up: not typical
    total = sum(c.frames for c in cams)
    print("\n=== live run summary ===")
    print(f"cameras          : {len(cams)}, {'sequential' if args.sequential else 'batched'} detection, "
          f"mean batch {np.mean(batch_sizes) if batch_sizes else 0:.1f} frames")
    print(f"throughput       : {total} frames in {elapsed:.1f} s -> {total / elapsed:.1f} frames/s in total")
    print(f"detection / cycle: {ms_stats(detect_ms[warmup:])}")
    print(f"whole cycle      : {ms_stats(cycle_ms[warmup:])}")
    if cycle_ms[warmup:]:
        budget = 1000 / max(c.fps for c in cams)
        p95 = np.percentile(cycle_ms[warmup:], 95)
        print(f"real-time budget : {budget:.1f} ms per cycle -> "
              f"{'KEEPS UP' if p95 < budget else 'TOO SLOW, live cameras will drop frames'}")
    for c in cams:
        print(f"[{c.name}] {c.frames} frames, {c.reader.dropped} dropped, {c.reader.reconnects} reconnects, "
              f"{c.camera_moves} camera moves, {c.alerts.count} alerts, incident check {ms_stats(c.check_ms)}")
    print(f"alerts           : see {Path(args.out) / 'events.jsonl'}")


if __name__ == "__main__":
    main()

"""Web service and dashboard.

  GET  /                     dashboard (live camera pictures, camera health, incidents, upload)
  GET  /api/status           live cameras right now (from the live process's status.json)
  GET  /api/events           live alerts, newest first (from events.jsonl); ?run=current for this run only
  GET  /api/events/hourly    alerts per hour, last 24 h
  POST /api/analyze          upload a clip -> job id; the clip is analysed in the background
  GET  /api/jobs[/{id}]      analysis jobs: status, events, report and key-frame links
  GET  /metrics              Prometheus metrics: camera fps, dropped frames, alerts, latency
  GET  /health               liveness check

The live process (scripts/live.py) and this service share only files in the live folder, so
either can be restarted without stopping the other. Uploaded clips are analysed one at a time
(one worker), so jobs never compete for the GPU. Start with `incident-recon serve`.
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
STALE_S = 10          # status older than this: the live process is not running


def run_analysis(video: Path, out_dir: Path, weights: str) -> dict:
    """Default analysis: tracking + incident report (same as `incident-recon analyze`)."""
    from .incident_report import build_incident_report
    from .track import track_video

    s = track_video(video, out_dir, weights=weights)
    report = build_incident_report(video, s["tracks_csv"], out_dir)
    events = json.loads((out_dir / f"{video.stem}_events.json").read_text())
    return {"events": events, "report": report.name, "video": Path(s["annotated_video"]).name,
            "key_frames": sorted(p.name for p in out_dir.glob(f"{video.stem}_event*.jpg")),
            "frames": s["frames"], "processing_fps": s["processing_fps"]}


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def create_app(live_dir="outputs/live", jobs_dir="outputs/jobs", weights: str | None = None,
               analyze: Callable[[Path, Path, str], dict] = run_analysis, max_upload_mb: int = 500) -> FastAPI:
    live_dir, jobs_dir = Path(live_dir), Path(jobs_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="incident-recon")
    worker = ThreadPoolExecutor(max_workers=1)
    lock = threading.Lock()
    jobs: dict[str, dict] = {}

    def save(job: dict) -> None:
        with lock:
            jobs[job["id"]] = job
            (jobs_dir / job["id"] / "job.json").write_text(json.dumps(job, indent=1))

    # Jobs from earlier runs: keep finished ones, mark interrupted ones as failed.
    for f in jobs_dir.glob("*/job.json"):
        job = _read_json(f, None)
        if job:
            if job["status"] in ("queued", "running"):
                job.update(status="failed", error="service restarted before the job finished")
                f.write_text(json.dumps(job, indent=1))
            jobs[job["id"]] = job

    def process(job: dict) -> None:
        save({**job, "status": "running"})
        folder = jobs_dir / job["id"]
        try:
            result = analyze(folder / job["input"], folder, weights)
            save({**job, "status": "done", "result": result,
                  "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        except Exception as e:   # a broken clip must not stop the worker
            save({**job, "status": "failed", "error": f"{type(e).__name__}: {e}"})

    def live_status() -> dict:
        status = _read_json(live_dir / "status.json", {"running": False, "cameras": []})
        updated = status.get("updated_utc")
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(updated)).total_seconds() if updated else None
        status["age_s"] = round(age, 1) if age is not None else None
        status["live"] = bool(status.get("running")) and age is not None and age < STALE_S
        return status

    def live_events() -> list[dict]:
        events = []
        try:
            with open(live_dir / "events.jsonl") as f:
                for line in f:
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass   # a line being written right now
        except FileNotFoundError:
            pass
        return events

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return (Path(__file__).parent / "dashboard.html").read_text()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/status")
    def status() -> dict:
        return live_status()

    @app.get("/api/events")
    def events(limit: int = 100, camera: str | None = None, run: str = "all") -> list[dict]:
        """run: "all", "current" (the live process's latest run) or a run id."""
        if run == "current":
            run = live_status().get("run") or ""
        ev = [e for e in live_events()
              if (camera is None or e.get("camera") == camera) and (run == "all" or e.get("run") == run)]
        return ev[::-1][:limit]

    @app.get("/api/events/hourly")
    def hourly() -> list[dict]:
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        hours = [now - timedelta(hours=h) for h in range(23, -1, -1)]
        count = Counter(datetime.fromisoformat(e["alert_utc"]).replace(minute=0, second=0, microsecond=0)
                        for e in live_events() if "alert_utc" in e)
        return [{"hour_utc": h.isoformat(timespec="minutes"), "alerts": count.get(h, 0)} for h in hours]

    @app.post("/api/analyze", status_code=202)
    def submit(file: Annotated[UploadFile, File()]) -> dict:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(400, f"expected a video ({', '.join(sorted(VIDEO_SUFFIXES))})")
        if weights is None and analyze is run_analysis:
            raise HTTPException(503, "no detector weights configured (start with --weights)")
        job_id = uuid.uuid4().hex[:12]
        folder = jobs_dir / job_id
        folder.mkdir()
        target = folder / f"input{suffix}"       # never use the uploaded name as a path
        size, limit = 0, max_upload_mb * 1024 * 1024
        with open(target, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    out.close()
                    shutil.rmtree(folder)
                    raise HTTPException(413, f"file larger than {max_upload_mb} MB")
                out.write(chunk)
        job = {"id": job_id, "status": "queued", "filename": Path(file.filename).name, "input": target.name,
               "size_mb": round(size / 1e6, 1), "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        save(job)
        worker.submit(process, job)
        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        with lock:
            return sorted(jobs.values(), key=lambda j: j["submitted_utc"], reverse=True)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        with lock:
            if job_id not in jobs:
                raise HTTPException(404, "no such job")
            return jobs[job_id]

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        s, ev = live_status(), live_events()
        lines = ["# HELP incident_recon_live_up 1 if the live process wrote its status in the last 10 s",
                 "# TYPE incident_recon_live_up gauge", f"incident_recon_live_up {int(s['live'])}"]
        per_camera = [("processed_fps", "gauge", "frames processed per second"),
                      ("frames", "counter", "frames processed"), ("dropped", "counter", "frames dropped (too slow)"),
                      ("reconnects", "counter", "stream reconnects"), ("camera_moves", "counter", "camera moves"),
                      ("alerts", "counter", "alerts raised")]
        for key, kind, about in per_camera:
            name = f"incident_recon_camera_{key.removeprefix('camera_')}" + ("_total" if kind == "counter" else "")
            lines += [f"# HELP {name} {about}", f"# TYPE {name} {kind}"]
            lines += [f'{name}{{camera="{c["name"]}"}} {c[key]}' for c in s["cameras"] if c.get(key) is not None]
        if s.get("cycle_ms_p95") is not None:
            lines += ["# HELP incident_recon_cycle_ms_p95 95th percentile processing time per cycle (ms)",
                      "# TYPE incident_recon_cycle_ms_p95 gauge", f"incident_recon_cycle_ms_p95 {s['cycle_ms_p95']}"]
        lines += ["# HELP incident_recon_events_total alerts in events.jsonl by type",
                  "# TYPE incident_recon_events_total counter"]
        lines += [f'incident_recon_events_total{{type="{t}"}} {n}' for t, n in Counter(e["type"] for e in ev).items()]
        with lock:
            states = Counter(j["status"] for j in jobs.values())
        lines += ["# HELP incident_recon_jobs analysis jobs by status", "# TYPE incident_recon_jobs gauge"]
        lines += [f'incident_recon_jobs{{status="{k}"}} {n}' for k, n in states.items()]
        return "\n".join(lines) + "\n"

    app.mount("/live", StaticFiles(directory=live_dir), name="live")
    app.mount("/jobs", StaticFiles(directory=jobs_dir), name="jobs")
    app.state.worker = worker
    return app

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from incident_recon.service import create_app


def now(minus_s=0):
    return (datetime.now(timezone.utc) - timedelta(seconds=minus_s)).isoformat(timespec="seconds")


def event(camera, kind="sudden_stop", minus_s=0, key="k.jpg"):
    return {"id": f"{camera}_{kind}_{minus_s}", "camera": camera, "alert_utc": now(minus_s), "type": kind,
            "event_stream_s": 9.4, "detail": "car #13 went to standstill", "clip": "c.mp4", "key_frame": key}


@pytest.fixture
def live(tmp_path):
    d = tmp_path / "live"
    d.mkdir()
    (d / "status.json").write_text(json.dumps({
        "updated_utc": now(), "running": True, "cycle_ms_p95": 18.0,
        "cameras": [{"name": "north", "processed_fps": 25.0, "frames": 500, "dropped": 3, "reconnects": 1,
                     "camera_moves": 0, "alerts": 2, "latest_jpg": "latest/north.jpg"}]}))
    lines = [json.dumps(event("north", minus_s=7200)), json.dumps(event("south", "stopped_vehicles")),
             '{"half a line being writ']
    (d / "events.jsonl").write_text("\n".join(lines))
    (d / "k.jpg").write_bytes(b"jpeg")
    return d


def fake_analysis(video, out_dir, weights):
    (out_dir / "report.md").write_text("# report")
    return {"events": [{"type": "sudden_stop", "time_s": 9.37, "detail": "car"}], "report": "report.md",
            "video": "v.mp4", "key_frames": [], "frames": 10, "processing_fps": 50}


def client(live, tmp_path, **kw):
    kw.setdefault("analyze", fake_analysis)
    return TestClient(create_app(live, tmp_path / "jobs", **kw))


def wait_done(c, job_id):
    c.app.state.worker.shutdown(wait=True)       # let the background worker finish
    return c.get(f"/api/jobs/{job_id}").json()


def test_status_is_live_when_fresh(live, tmp_path):
    s = client(live, tmp_path).get("/api/status").json()
    assert s["live"] is True and s["cameras"][0]["name"] == "north"


def test_status_goes_stale_when_live_process_stops_writing(live, tmp_path):
    status = json.loads((live / "status.json").read_text())
    (live / "status.json").write_text(json.dumps({**status, "updated_utc": now(60)}))
    assert client(live, tmp_path).get("/api/status").json()["live"] is False


def test_no_live_process_at_all(tmp_path):
    s = client(tmp_path / "empty", tmp_path).get("/api/status").json()
    assert s["live"] is False and s["cameras"] == []


def test_events_newest_first_filtered_and_half_written_line_skipped(live, tmp_path):
    c = client(live, tmp_path)
    assert [e["camera"] for e in c.get("/api/events").json()] == ["south", "north"]
    assert [e["camera"] for e in c.get("/api/events?camera=north").json()] == ["north"]


def test_hourly_counts_cover_24_hours(live, tmp_path):
    hours = client(live, tmp_path).get("/api/events/hourly").json()
    assert len(hours) == 24
    assert hours[-1]["alerts"] == 1 and hours[-3]["alerts"] == 1 and sum(h["alerts"] for h in hours) == 2


def test_metrics_in_prometheus_format(live, tmp_path):
    m = client(live, tmp_path).get("/metrics").text
    assert "incident_recon_live_up 1" in m
    assert 'incident_recon_camera_dropped_total{camera="north"} 3' in m
    assert 'incident_recon_events_total{type="sudden_stop"} 1' in m
    assert "incident_recon_cycle_ms_p95 18.0" in m


def test_key_frames_are_served(live, tmp_path):
    assert client(live, tmp_path).get("/live/k.jpg").content == b"jpeg"


def test_dashboard_page(live, tmp_path):
    assert "Incident Monitor" in client(live, tmp_path).get("/").text


def test_upload_is_analysed_in_the_background(live, tmp_path):
    c = client(live, tmp_path)
    r = c.post("/api/analyze", files={"file": ("../../etc/clip.mp4", b"video bytes")})
    assert r.status_code == 202
    job = wait_done(c, r.json()["job_id"])
    assert job["status"] == "done" and job["result"]["events"][0]["time_s"] == 9.37
    assert job["filename"] == "clip.mp4"                                  # path parts stripped
    assert (tmp_path / "jobs" / job["id"] / "input.mp4").read_bytes() == b"video bytes"
    assert c.get(f"/jobs/{job['id']}/report.md").text == "# report"


def test_upload_rejects_non_video(live, tmp_path):
    assert client(live, tmp_path).post("/api/analyze", files={"file": ("x.txt", b"hi")}).status_code == 400


def test_upload_too_large_is_refused_and_cleaned_up(live, tmp_path):
    c = client(live, tmp_path, max_upload_mb=1)
    r = c.post("/api/analyze", files={"file": ("big.mp4", b"0" * (2 * 1024 * 1024))})
    assert r.status_code == 413
    assert list((tmp_path / "jobs").iterdir()) == []


def test_upload_refused_without_weights(live, tmp_path):
    c = TestClient(create_app(live, tmp_path / "jobs"))                   # real analysis, no weights
    assert c.post("/api/analyze", files={"file": ("a.mp4", b"v")}).status_code == 503


def test_failed_analysis_is_reported_and_worker_survives(live, tmp_path):
    def broken(video, out_dir, weights):
        raise ValueError("cannot decode video")
    c = client(live, tmp_path, analyze=broken)
    job_id = c.post("/api/analyze", files={"file": ("a.mp4", b"v")}).json()["job_id"]
    job = wait_done(c, job_id)
    assert job["status"] == "failed" and "cannot decode video" in job["error"]


def test_restart_marks_interrupted_jobs_failed_and_keeps_finished_ones(live, tmp_path):
    jobs = tmp_path / "jobs"
    for jid, state in [("a1", "running"), ("b2", "done")]:
        (jobs / jid).mkdir(parents=True)
        (jobs / jid / "job.json").write_text(json.dumps({"id": jid, "status": state, "submitted_utc": now()}))
    c = client(live, tmp_path)
    assert c.get("/api/jobs/a1").json()["status"] == "failed"
    assert c.get("/api/jobs/b2").json()["status"] == "done"
    assert c.get("/api/jobs/zz").status_code == 404


def test_events_of_the_current_run_only(live, tmp_path):
    status = json.loads((live / "status.json").read_text())
    (live / "status.json").write_text(json.dumps({**status, "run": "r2"}))
    lines = [json.dumps({**event("north", minus_s=60), "run": "r1"}), json.dumps({**event("north"), "run": "r2"})]
    (live / "events.jsonl").write_text("\n".join(lines))
    c = client(live, tmp_path)
    assert [e["run"] for e in c.get("/api/events?run=current").json()] == ["r2"]
    assert len(c.get("/api/events").json()) == 2
    assert [e["run"] for e in c.get("/api/events?run=r1").json()] == ["r1"]

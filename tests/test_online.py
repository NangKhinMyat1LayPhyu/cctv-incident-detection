from conftest import track, tracks

from incident_recon.online import IncidentMonitor

FPS = 10.0


def feed(monitor, t, start=0):
    """Feed a tracks table frame by frame, as a live stream would. Returns all alerts raised."""
    alerts = []
    for f, g in t.groupby("frame"):
        frame = int(f) + start
        boxes = g[["track_id", "class", "conf", "x1", "y1", "x2", "y2"]].itertuples(index=False, name=None)
        monitor.add(frame, list(boxes))
        alerts += monitor.check(frame)
    return alerts


def stopping_car(n=80):
    return track(1, range(n), 100, [100 + 8.0 * min(i, 29) for i in range(n)])


def parked_scene(n):
    return tracks(*[track(i, range(n), 100 + 300 * i, 300.0) for i in range(3)],
                  track(9, range(n), 900, [8.0 * (i % 40) for i in range(n)]))


def test_live_sudden_stop_matches_offline_frame_in_absolute_numbers():
    m = IncidentMonitor(FPS)
    m.reset(1000)                                   # stream already running: frame numbers are large
    alerts = feed(m, stopping_car(), start=1000)
    assert [a["type"] for a in alerts] == ["sudden_stop"]
    assert alerts[0]["frame"] == 1034               # offline detector: frame 34 of the clip


def test_same_sudden_stop_alerted_only_once():
    m = IncidentMonitor(FPS)
    alerts = feed(m, stopping_car(n=200))           # checked again every second for 17 s
    assert len(alerts) == 1


def test_stopped_vehicles_repeat_at_most_once_a_minute_even_after_camera_move():
    m = IncidentMonitor(FPS, repeat_after_s=60)
    first = feed(m, parked_scene(200))
    m.reset(200)                                    # camera moved: tracks forgotten, repeat limit kept
    second = feed(m, parked_scene(200), start=200)
    assert [a["type"] for a in first] == ["stopped_vehicles"]
    assert second == []


def test_stopped_vehicles_alerted_again_after_the_repeat_interval():
    m = IncidentMonitor(FPS, repeat_after_s=60)
    alerts = feed(m, parked_scene(700))             # 70 s of the same scene
    assert len(alerts) == 2


def test_no_detection_until_view_seen_for_min_view_s():
    m = IncidentMonitor(FPS, min_view_s=3)
    feed(m, stopping_car(n=25))                     # 2.5 s: too short to judge
    assert not m.ran

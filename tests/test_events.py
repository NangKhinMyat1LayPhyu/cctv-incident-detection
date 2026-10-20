import pytest
from conftest import track, tracks

from incident_recon.events import detect_incidents, motion_table, sudden_stops

FPS = 10.0


def moving(tid, x=500, frames=range(30)):
    """2 vehicle sizes per second: 8 px per frame for a 40 px box at 10 fps."""
    return track(tid, frames, x, [8.0 * i for i in range(len(frames))])


def parked(tid, x, y=300.0, frames=range(30)):
    return track(tid, frames, x, y)


def stopping_car(tid=1, stop_at=29, n=60):
    """Drives at 2 sizes/s until `stop_at`, then stands still."""
    return track(tid, range(n), 100, [100 + 8.0 * min(i, stop_at) for i in range(n)])


def test_speed_is_measured_in_vehicle_sizes_per_second():
    m = motion_table(moving(1), fps=FPS)
    assert m["nspeed"].dropna().to_numpy() == pytest.approx(2.0)


def test_speed_does_not_depend_on_distance_from_camera():
    near = track(1, range(30), 100, [16.0 * i for i in range(30)], size=80)   # big box, 16 px/frame
    far = track(2, range(30), 300, [4.0 * i for i in range(30)], size=20)     # small box, 4 px/frame
    m = motion_table(tracks(near, far), fps=FPS).dropna(subset=["nspeed"])
    assert m.groupby("track_id")["nspeed"].median().to_list() == pytest.approx([2.0, 2.0])


def test_sudden_stop_found_at_the_first_standstill_frame():
    stops = sudden_stops(motion_table(stopping_car(), fps=FPS), fps=FPS)
    assert len(stops) == 1
    assert stops.iloc[0]["frame"] == 34      # 0.5 s window: first frame with no movement since frame 29


def test_brief_pause_is_not_a_sudden_stop():
    # stops for 0.5 s, then drives on: must stand still for at least 1 s to count
    ys = [8.0 * i for i in range(30)] + [8.0 * 29] * 5 + [8.0 * (i - 5) for i in range(35, 60)]
    m = motion_table(track(1, range(60), 100, ys), fps=FPS)
    assert sudden_stops(m, fps=FPS).empty


def test_constant_traffic_raises_nothing():
    m = motion_table(tracks(moving(1, 100), moving(2, 300), moving(3, 500)), fps=FPS)
    assert detect_incidents(m, fps=FPS).empty


def test_scattered_stopped_vehicles_are_an_incident():
    scene = tracks(parked(1, 100), parked(2, 400), parked(3, 700), moving(4), moving(5, 900))
    events = detect_incidents(motion_table(scene, fps=FPS), fps=FPS, max_queue_neighbours=6)
    assert list(events["type"]) == ["stopped_vehicles"]


def test_queue_is_an_incident_for_v1_but_not_v2():
    queue = tracks(*[parked(i, 100 + 20 * i) for i in range(10)], moving(20), moving(21, 900))
    m = motion_table(queue, fps=FPS)
    assert list(detect_incidents(m, fps=FPS)["type"]) == ["stopped_vehicles"]       # v1
    assert detect_incidents(m, fps=FPS, max_queue_neighbours=6).empty                 # v2: a queue, not a crash


def test_one_stopped_vehicle_is_not_enough():
    scene = tracks(parked(1, 100), moving(2), moving(3, 900))
    assert detect_incidents(motion_table(scene, fps=FPS), fps=FPS).empty


def test_debounce_needs_the_vehicle_fast_for_long_enough():
    # creeps at 0.25 sizes/s, jumps 20 px in one frame, then stands still. Speed is measured over
    # 0.5 s, so the jump reads as "fast" (>= 1 size/s) for 5 frames = 0.5 s.
    ys = [float(i) for i in range(20)] + [39.0] * 31
    m = motion_table(track(1, range(51), 100, ys), fps=FPS)
    assert len(sudden_stops(m, fps=FPS)) == 1                  # v1/v2: any fast moment is enough
    assert len(sudden_stops(m, fps=FPS, min_fast_s=0.4)) == 1  # fast for 0.5 s >= 0.4 s
    assert sudden_stops(m, fps=FPS, min_fast_s=0.6).empty      # fast for 0.5 s < 0.6 s


def test_queue_filter_for_sudden_stops():
    queue = [parked(i, 100 + 20 * i, y=332.0, frames=range(60)) for i in range(2, 10)]
    joining = stopping_car(tid=1)                               # stops at y = 332, beside the queue
    m = motion_table(tracks(joining, *queue), fps=FPS)
    v2 = detect_incidents(m, fps=FPS, max_queue_neighbours=6, min_stopped_share=1.1)    # rule B off
    v3 = detect_incidents(m, fps=FPS, max_queue_neighbours=6, min_stopped_share=1.1, queue_filter_stops=True)
    assert list(v2["type"]) == ["sudden_stop"]
    assert v3.empty

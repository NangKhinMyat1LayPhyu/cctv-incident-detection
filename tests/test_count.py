import numpy as np
from conftest import track, tracks

from incident_recon.count import count_crossings

LINE = [(0, 100), (200, 100)]   # horizontal line at y = 100, from x = 0 to x = 200


def test_vehicle_crossing_once_is_counted_once_with_direction():
    t = track(1, range(20), 100, [50 + 5 * i for i in range(20)])   # y 50 -> 145, moving down
    out = count_crossings(t, LINE)
    assert len(out) == 1
    assert out.iloc[0]["direction"] == "down"


def test_upward_crossing_is_named_up():
    t = track(1, range(20), 100, [150 - 5 * i for i in range(20)])
    assert count_crossings(t, LINE).iloc[0]["direction"] == "up"


def test_vehicle_going_back_and_forth_is_still_counted_once():
    ys = [50 + 10 * i for i in range(11)] + [150 - 10 * i for i in range(11)]   # down past the line, then back
    out = count_crossings(track(1, range(22), 100, ys), LINE)
    assert len(out) == 1


def test_jitter_inside_dead_band_is_not_counted():
    # box height 40, margin 0.15 -> dead band of 6 px on each side; wobble of +-4 px stays inside it
    ys = [100 + (4 if i % 2 else -4) for i in range(30)]
    assert count_crossings(track(1, range(30), 100, ys), LINE).empty


def test_same_wobble_without_dead_band_would_be_counted():
    ys = [100 + (4 if i % 2 else -4) for i in range(30)]
    assert len(count_crossings(track(1, range(30), 100, ys), LINE, margin=0)) == 1


def test_crossing_beside_the_line_segment_is_not_counted():
    t = track(1, range(20), 300, [50 + 5 * i for i in range(20)])   # x = 300: past the line's right end
    assert count_crossings(t, LINE).empty


def test_crossing_while_camera_moved_is_not_counted():
    t = track(1, range(20), 100, [50 + 5 * i for i in range(20)])
    valid = np.ones(20, dtype=bool)
    valid[8:13] = False            # camera moved while the vehicle was at the line
    assert count_crossings(t, LINE, valid=valid).empty


def test_each_vehicle_counted_separately():
    down = [50 + 5 * i for i in range(20)]
    t = tracks(track(1, range(20), 50, down), track(2, range(20), 150, down), track(3, range(20), 100, 40))
    assert sorted(count_crossings(t, LINE)["track_id"]) == [1, 2]

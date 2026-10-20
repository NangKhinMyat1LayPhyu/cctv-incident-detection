from conftest import track, tracks

from incident_recon.clean import clean_tracks


def test_class_flicker_resolved_by_total_confidence():
    # 6 frames 'car' at 0.9 (total 5.4) vs 8 frames 'truck' at 0.5 (total 4.0): car wins although rarer
    t = track(1, range(14), 100, 100, cls=["car"] * 6 + ["truck"] * 8, conf=[0.9] * 6 + [0.5] * 8)
    assert set(clean_tracks(t, fps=10)["class"]) == {"car"}


def test_short_tracks_are_dropped():
    t = tracks(track(1, range(10), 100, 100), track(2, range(4), 200, 100))   # 1.0 s and 0.4 s at 10 fps
    out = clean_tracks(t, fps=10, min_seconds=0.5)
    assert set(out["track_id"]) == {1}


def test_input_is_not_modified():
    t = track(1, range(10), 100, 100, cls=["car", "truck"] * 5)
    before = t.copy()
    clean_tracks(t, fps=10)
    assert t.equals(before)

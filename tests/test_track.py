import pytest

pytest.importorskip("ultralytics")   # CI installs only the light dependencies

from incident_recon.track import detector_conf, tracker_settings  # noqa: E402


def test_original_setup_is_unchanged():
    cfg = tracker_settings(0.3, None)
    assert detector_conf(0.3, None) == 0.3                       # detector drops boxes under 0.3
    assert (cfg["track_high_thresh"], cfg["new_track_thresh"]) == (0.25, 0.25)   # Ultralytics defaults


def test_low_conf_boxes_can_only_continue_tracks():
    cfg = tracker_settings(0.3, 0.1)
    assert detector_conf(0.3, 0.1) == 0.1                        # weak boxes now reach the tracker...
    assert cfg["track_low_thresh"] == 0.1
    assert cfg["track_high_thresh"] == 0.3 and cfg["new_track_thresh"] == 0.3   # ...but cannot start a track

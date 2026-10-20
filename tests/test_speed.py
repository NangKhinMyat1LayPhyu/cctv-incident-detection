import numpy as np
import pytest
from conftest import track, tracks

from incident_recon.speed import fit_homography, speed_summary, to_road, track_speeds

# Image trapezoid (perspective: the road narrows with distance) -> a 10 m x 40 m road rectangle
IMAGE = [(100, 400), (300, 400), (220, 100), (180, 100)]
ROAD = [(0, 0), (10, 0), (10, 40), (0, 40)]


def test_homography_maps_the_four_points_exactly():
    H = fit_homography(IMAGE, ROAD)
    xs, ys = zip(*IMAGE)
    np.testing.assert_allclose(to_road(H, xs, ys), ROAD, atol=1e-3)


def test_constant_speed_measured_in_kmh():
    H = fit_homography([(0, 0), (100, 0), (100, 100), (0, 100)], [(0, 0), (10, 0), (10, 10), (0, 10)])  # 10 px = 1 m
    fps = 10
    t = track(1, range(30), 50, [5.0 * i for i in range(30)], fps=fps)   # 5 px/frame = 0.5 m/frame = 5 m/s
    s = track_speeds(t, H, fps=fps)
    assert s["speed_kmh"].dropna().to_numpy() == pytest.approx(18.0)     # 5 m/s = 18 km/h
    assert s["speed_kmh"].isna().sum() == 10                              # no speed during the first 1 s window


def test_summary_labels_stationary_and_direction():
    H = fit_homography([(0, 0), (100, 0), (100, 100), (0, 100)], [(0, 0), (10, 0), (10, 10), (0, 10)])
    moving = track(1, range(30), 50, [5.0 * i for i in range(30)])
    parked = track(2, range(30), 80, 60.0)
    s = speed_summary(track_speeds(tracks(moving, parked), H, fps=10))
    assert s.loc[1, "state"] == "moving" and s.loc[1, "direction"] == "towards"
    assert s.loc[2, "state"] == "stationary" and s.loc[2, "direction"] == "-"

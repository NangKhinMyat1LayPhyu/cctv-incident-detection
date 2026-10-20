import numpy as np

from incident_recon.camera import CameraWatch


def scene(seed=0, h=240, w=320):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    return np.kron(small, np.ones((8, 8, 1), dtype=np.uint8))      # blocky texture, like a real scene


def test_still_camera_with_sensor_noise_is_not_a_move():
    watch, img = CameraWatch(), scene()
    rng = np.random.default_rng(1)
    watch.moved(img)
    for _ in range(5):
        noisy = np.clip(img.astype(int) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
        assert not watch.moved(noisy)


def test_pan_is_a_move():
    watch, img = CameraWatch(), scene()
    watch.moved(img)
    assert watch.moved(np.roll(img, 24, axis=1))        # whole picture slides sideways


def test_cut_to_another_view_is_a_move():
    watch = CameraWatch()
    watch.moved(scene(0))
    assert watch.moved(scene(5))


def test_one_vehicle_moving_is_not_a_camera_move():
    watch, img = CameraWatch(), scene()
    watch.moved(img)
    moved_car = img.copy()
    moved_car[100:140, 40:80] = moved_car[100:140, 0:40]   # one small region changes
    assert not watch.moved(moved_car)

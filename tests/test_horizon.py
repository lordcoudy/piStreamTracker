"""Horizon tilt from MoveNet shoulder/hip pairs, plus warp/level helpers."""

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pistream.horizon import (
    estimate_roll_deg,
    fill_crop_scale,
    level_affine,
    step_angle,
    tilt_degrees,
    transform_points,
    warp_level,
)


def _kp(pairs_xy, conf=0.9):
    """17x3 keypoints; pairs_xy maps index -> (x, y)."""
    kp = np.zeros((17, 3), dtype=np.float32)
    for i, (x, y) in pairs_xy.items():
        kp[i] = (x, y, conf)
    return kp


class TiltDegreesTests(unittest.TestCase):
    def test_level_shoulders_return_near_zero(self):
        kp = [[0, 0, 0]] * 17
        kp[5] = [10.0, 20.0, 0.9]
        kp[6] = [40.0, 20.0, 0.9]
        self.assertAlmostEqual(tilt_degrees(kp, 0.3), 0.0, places=5)

    def test_right_shoulder_lower_is_positive_tilt(self):
        kp = [[0, 0, 0]] * 17
        kp[5] = [10.0, 10.0, 0.9]
        kp[6] = [40.0, 40.0, 0.9]
        angle = tilt_degrees(kp, 0.3)
        self.assertGreater(angle, 40.0)
        self.assertLess(angle, 50.0)

    def test_ignores_low_confidence_points(self):
        kp = [[0, 0, 0]] * 17
        kp[5] = [10.0, 10.0, 0.1]
        kp[6] = [40.0, 40.0, 0.1]
        self.assertIsNone(tilt_degrees(kp, 0.3))

    def test_averages_shoulder_and_hip_pairs(self):
        kp = [[0, 0, 0]] * 17
        kp[5] = [0.0, 0.0, 0.9]
        kp[6] = [10.0, 10.0, 0.9]
        kp[11] = [0.0, 0.0, 0.9]
        kp[12] = [10.0, 0.0, 0.9]  # hips level
        angle = tilt_degrees(kp, 0.3)
        shoulder = math.degrees(math.atan2(10.0, 10.0))
        self.assertAlmostEqual(angle, shoulder / 2.0, places=5)

    def test_short_keypoint_list_returns_none(self):
        self.assertIsNone(tilt_degrees([[1, 2, 0.9]] * 5, 0.3))

    def test_estimate_roll_deg_is_tilt_degrees(self):
        kp = _kp({5: (40, 50), 6: (80, 50)})
        self.assertEqual(estimate_roll_deg(kp, 0.3), tilt_degrees(kp, 0.3))


class EstimateRollTests(unittest.TestCase):
    def test_level_shoulders_zero_roll(self):
        kp = _kp({5: (40, 50), 6: (80, 50)})
        self.assertAlmostEqual(estimate_roll_deg(kp, 0.3), 0.0, places=2)

    def test_tilted_shoulders_positive_clockwise(self):
        kp = _kp({5: (0, 0), 6: (100, 10)})
        self.assertAlmostEqual(
            estimate_roll_deg(kp, 0.3),
            math.degrees(math.atan2(10, 100)),
            places=2,
        )

    def test_average_shoulder_and_hip(self):
        kp = _kp({
            5: (0, 0), 6: (100, 10),
            11: (0, 50), 12: (100, 60),
        })
        expected = math.degrees(math.atan2(10, 100))
        self.assertAlmostEqual(estimate_roll_deg(kp, 0.3), expected, places=2)

    def test_low_confidence_returns_none(self):
        kp = _kp({5: (0, 0), 6: (100, 10)}, conf=0.1)
        self.assertIsNone(estimate_roll_deg(kp, 0.3))

    def test_near_vertical_pair_ignored(self):
        kp = _kp({5: (50, 0), 6: (52, 80)})
        self.assertIsNone(estimate_roll_deg(kp, 0.3))


class StepAngleTests(unittest.TestCase):
    def test_step_holds_on_none(self):
        ema = step_angle(True, 8.0, 0.0, alpha=1.0, max_angle=20.0)
        held = step_angle(True, None, ema, alpha=1.0, max_angle=20.0)
        self.assertAlmostEqual(held, ema)

    def test_step_disabled_resets_to_zero(self):
        self.assertEqual(step_angle(False, 12.0, 12.0, alpha=0.15, max_angle=20.0), 0.0)

    def test_step_rejects_outlier_beyond_1_5x_max(self):
        out = step_angle(True, 40.0, 4.0, alpha=1.0, max_angle=20.0)
        self.assertAlmostEqual(out, 4.0)

    def test_step_clamps_within_max(self):
        out = step_angle(True, 19.0, 0.0, alpha=1.0, max_angle=20.0)
        self.assertAlmostEqual(out, 19.0)


class WarpLevelTests(unittest.TestCase):
    def test_fill_crop_identity_at_zero(self):
        self.assertAlmostEqual(fill_crop_scale(0.0, 1280, 960), 1.0, places=6)

    def test_fill_crop_grows_with_angle(self):
        self.assertGreater(fill_crop_scale(10.0, 1280, 960), 1.1)

    def test_level_affine_center_fixed(self):
        M = level_affine(200, 100, 15.0, fill_crop=False)
        pts = transform_points(np.array([[100.0, 50.0]]), M)
        self.assertAlmostEqual(pts[0, 0], 100.0, delta=0.5)
        self.assertAlmostEqual(pts[0, 1], 50.0, delta=0.5)

    def test_warp_level_noop_small_angle(self):
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        frame[20, 40] = (0, 0, 255)
        out = warp_level(frame, 0.2, min_apply=0.5)
        self.assertIs(out, frame)

    def test_warp_level_same_shape(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        out = warp_level(frame, 8.0, fill_crop=True)
        self.assertEqual(out.shape, frame.shape)


class HorizonConfigDefaultsTests(unittest.TestCase):
    def test_load_config_horizon_defaults(self):
        from pistream.config import load_config

        with TemporaryDirectory() as tmp:
            cfg = load_config(str(Path(tmp) / 'missing.yaml'))
        h = cfg['tracker']['horizon']
        self.assertFalse(h['enabled'])
        self.assertEqual(h['max_angle'], 20)
        self.assertTrue(h['fill_crop'])


if __name__ == '__main__':
    unittest.main()

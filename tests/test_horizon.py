"""Horizon tilt from MoveNet shoulder/hip pairs."""

import math
import unittest

from pistream.horizon import tilt_degrees


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


if __name__ == '__main__':
    unittest.main()

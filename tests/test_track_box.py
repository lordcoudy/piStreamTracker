"""Tracker re-init gate and bbox IoU."""

import unittest
from types import SimpleNamespace

import numpy as np

from pistream.track import DETECTION_MISS_LIMIT, HumanTracker, bbox_iou, should_reinit_tracker


class BBoxIouTests(unittest.TestCase):
    def test_identical_boxes_are_one(self):
        box = (10, 10, 40, 40)
        self.assertAlmostEqual(bbox_iou(box, box), 1.0)

    def test_disjoint_boxes_are_zero(self):
        self.assertEqual(bbox_iou((0, 0, 10, 10), (20, 20, 10, 10)), 0.0)


class ReinitGateTests(unittest.TestCase):
    def test_reinit_when_no_active_tracker(self):
        self.assertTrue(should_reinit_tracker(False, 0.9))

    def test_keep_tracker_when_iou_high(self):
        self.assertFalse(should_reinit_tracker(True, 0.7))

    def test_reinit_when_iou_low(self):
        self.assertTrue(should_reinit_tracker(True, 0.1))


class DetectionLossTests(unittest.TestCase):
    @staticmethod
    def tracker_shell():
        human = HumanTracker.__new__(HumanTracker)
        human.frame_count = 0
        human.detection_interval = 10
        human._detection_misses = 0
        human._last_keypoints = None
        human._last_confidence = 0.5
        human._shift_logger = None
        human.capture = SimpleNamespace(width=64, height=48)
        human.motors = SimpleNamespace(update=lambda *_: None, stop=lambda: None)
        human.detector = SimpleNamespace(keypoint_threshold=0.3)
        human.horizon_correction = False
        human._horizon_cfg = {
            'max_angle': 20.0,
            'ema_alpha': 0.15,
            'min_apply': 0.5,
            'fill_crop': True,
        }
        human._horizon_ema = 0.0
        human._horizon_M = None
        human._leveled_frame = None
        human._draw = lambda frame, _detection, _angle=0.0: frame
        human._apply_zoom = lambda frame: frame
        return human

    def test_lost_object_tracker_clears_stale_status_detection(self):
        human = self.tracker_shell()
        human._async_detector = None
        human.current_detection = {'bbox': (1, 2, 10, 20)}
        human.tracker = SimpleNamespace(active=True, update=lambda _frame: None)

        human.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))

        self.assertIsNone(human.current_detection)

    def test_repeated_negative_pose_results_drop_stale_tracker(self):
        class ObjectTracker:
            active = True

            def __init__(self):
                self.resets = 0

            def update(self, _frame):
                return (1, 2, 10, 20) if self.active else None

            def reset(self):
                self.active = False
                self.resets += 1

        human = self.tracker_shell()
        human._async_detector = SimpleNamespace(
            poll_result=lambda: (True, None),
            busy=True,
        )
        human.current_detection = {'bbox': (1, 2, 10, 20)}
        human.tracker = ObjectTracker()
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        for _ in range(DETECTION_MISS_LIMIT):
            human.process_frame(frame)

        self.assertEqual(human.tracker.resets, 1)
        self.assertIsNone(human.current_detection)

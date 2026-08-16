"""Tracker re-init gate and bbox IoU."""

import unittest

from pistream.track import bbox_iou, should_reinit_tracker


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

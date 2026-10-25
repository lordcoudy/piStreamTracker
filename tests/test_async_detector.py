"""Async detector result-channel semantics."""

import unittest
import time
from threading import Event, Lock
from unittest.mock import patch

import numpy as np

from pistream.detect import AsyncDetector


class AsyncDetectorResultTests(unittest.TestCase):
    def test_new_negative_result_is_distinct_from_no_new_result(self):
        worker = AsyncDetector.__new__(AsyncDetector)
        worker._result_lock = Lock()
        worker._result = None
        worker._new_result = True

        self.assertEqual(worker.poll_result(), (True, None))
        self.assertEqual(worker.poll_result(), (False, None))


class AsyncDetectorFreshnessTests(unittest.TestCase):
    def test_reset_discards_inflight_result_but_accepts_next_generation(self):
        entered, release, finished = Event(), Event(), Event()

        class Detector:
            def detect(self, _frame):
                entered.set()
                release.wait(1)
                finished.set()
                return {'bbox': (1, 2, 10, 20), 'keypoints': None}

        worker = AsyncDetector(Detector(), 1.0, 64, 48)
        worker.start()
        try:
            worker.submit(np.zeros((48, 64, 3), dtype=np.uint8))
            self.assertTrue(entered.wait(1))
            worker.invalidate()
            release.set()
            self.assertTrue(finished.wait(1))
            deadline = time.monotonic() + 1
            while worker.busy and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertFalse(worker.busy)
            self.assertEqual(worker.poll_result(), (False, None))
            entered.clear()
            finished.clear()
            worker.submit(np.zeros((48, 64, 3), dtype=np.uint8))
            self.assertTrue(entered.wait(1))
            self.assertTrue(finished.wait(1))
            deadline = time.monotonic() + 1
            while worker.busy and time.monotonic() < deadline:
                time.sleep(.001)
            ready, result = worker.poll_result()
            self.assertTrue(ready)
            self.assertEqual(result['bbox'], (1, 2, 10, 20))
        finally:
            release.set()
            worker.stop()

    def test_age_is_measured_from_submission_not_inference_completion(self):
        worker = AsyncDetector(None, 1.0, 64, 48, max_result_age=0.5)
        worker._new_result = True
        worker._result = {'bbox': (1, 2, 10, 20)}
        worker._result_at = 10.0
        with patch('pistream.detect.time.monotonic', return_value=11.0):
            self.assertEqual(worker.poll_result(), (False, None))

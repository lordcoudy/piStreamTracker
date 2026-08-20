"""Async detector result-channel semantics."""

import unittest
from threading import Lock

from pistream.detect import AsyncDetector


class AsyncDetectorResultTests(unittest.TestCase):
    def test_new_negative_result_is_distinct_from_no_new_result(self):
        worker = AsyncDetector.__new__(AsyncDetector)
        worker._result_lock = Lock()
        worker._result = None
        worker._new_result = True

        self.assertEqual(worker.poll_result(), (True, None))
        self.assertEqual(worker.poll_result(), (False, None))

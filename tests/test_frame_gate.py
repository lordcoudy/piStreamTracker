"""Skip processing when VideoCapture has not published a new frame."""

import unittest

from pistream.preview import accept_new_frame


class AcceptNewFrameTests(unittest.TestCase):
    def test_same_object_is_not_new(self):
        frame = object()
        got, is_new = accept_new_frame(frame, True, frame)
        self.assertFalse(is_new)
        self.assertIs(got, frame)

    def test_new_object_is_accepted(self):
        prev = object()
        nxt = object()
        got, is_new = accept_new_frame(prev, True, nxt)
        self.assertTrue(is_new)
        self.assertIs(got, nxt)

    def test_failed_read_is_not_new(self):
        prev = object()
        got, is_new = accept_new_frame(prev, False, None)
        self.assertFalse(is_new)
        self.assertIs(got, prev)


if __name__ == '__main__':
    unittest.main()

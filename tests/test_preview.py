"""Web preview gating, scale, recording fps cap, camera stream URL."""

import unittest

from pistream.preview import (
    camera_stream_url,
    capped_recording_fps,
    preview_gate,
    preview_target_size,
)


class PreviewTargetSizeTests(unittest.TestCase):
    def test_shrinks_long_side_to_max_edge(self):
        w, h = preview_target_size(1280, 960, 640)
        self.assertEqual(max(w, h), 640)
        self.assertAlmostEqual(w / h, 1280 / 960, places=2)

    def test_leaves_small_frames_unchanged(self):
        self.assertEqual(preview_target_size(320, 240, 640), (320, 240))

    def test_non_positive_max_edge_is_a_no_op(self):
        self.assertEqual(preview_target_size(1280, 960, 0), (1280, 960))


class PreviewGateTests(unittest.TestCase):
    def test_skips_same_sequence(self):
        self.assertFalse(preview_gate(last_seq=3, current_seq=3, now=1.0,
                                      last_emit_at=0.0, min_interval=0.05))

    def test_emits_new_sequence_after_interval(self):
        self.assertTrue(preview_gate(last_seq=3, current_seq=4, now=1.0,
                                     last_emit_at=0.9, min_interval=0.05))

    def test_holds_new_sequence_until_interval_elapsed(self):
        self.assertFalse(preview_gate(last_seq=3, current_seq=4, now=0.92,
                                      last_emit_at=0.90, min_interval=0.05))


class RecordingFpsCapTests(unittest.TestCase):
    def test_does_not_exceed_source_fps(self):
        self.assertEqual(capped_recording_fps(60, 30), 30.0)

    def test_keeps_requested_when_below_source(self):
        self.assertEqual(capped_recording_fps(15, 30), 15.0)

    def test_invalid_source_falls_back_to_30(self):
        self.assertEqual(capped_recording_fps(60, 0), 30.0)


class CameraStreamUrlTests(unittest.TestCase):
    def test_builds_mjpeg_url_from_config(self):
        cfg = {
            'network': {'camera_ip': '192.168.100.1'},
            'camera': {'port': 8000},
        }
        self.assertEqual(camera_stream_url(cfg), 'http://192.168.100.1:8000/stream')


if __name__ == '__main__':
    unittest.main()

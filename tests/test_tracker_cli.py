"""Desktop controls during capture loss and manual recording-stop recovery."""

import unittest
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import cv2

from pistream.track import HumanTracker


def tracker_shell(reads=()):
    tracker = HumanTracker.__new__(HumanTracker)
    tracker.capture = SimpleNamespace(read=Mock(side_effect=reads), width=64, height=48)
    tracker.connect = Mock(return_value=True)
    tracker.cleanup = Mock()
    tracker.process_frame = Mock(return_value=(np.zeros((48, 64, 3), dtype=np.uint8), None))
    tracker.recording_frame = Mock(return_value=None)
    tracker.write_frame = Mock()
    tracker.update_fps = Mock()
    tracker._record_lock = RLock()
    tracker._recording = False
    tracker._recording_backend = None
    tracker._rec_thread = None
    tracker._async_detector = None
    tracker.remote_stop_pending = False
    tracker.motors = Mock(connected=False)
    tracker.tracker = SimpleNamespace(tracker_type=None)
    tracker._fps = 0.0
    tracker.screenshot = Mock()
    tracker.reset_tracking = Mock()
    tracker.stop_recording = Mock(return_value=True)
    tracker.start_recording = Mock(return_value=True)
    return tracker


class TrackerControlTests(unittest.TestCase):
    def test_quit_is_processed_during_missing_and_repeated_frames(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        for missing in ((False, None), (True, frame)):
            with self.subTest(missing=missing[0]):
                tracker = tracker_shell([(True, frame), missing])
                with (
                    patch('pistream.track.cv2.imshow'),
                    patch('pistream.track.cv2.waitKey', side_effect=[-1, ord('q')]),
                    patch('pistream.track.cv2.destroyAllWindows'),
                ):
                    self.assertEqual(tracker.run(), 0)
                self.assertEqual(tracker.process_frame.call_count, 1)
                self.assertEqual(tracker.write_frame.call_count, 1)
                self.assertEqual(tracker.update_fps.call_count, 1)
                tracker.cleanup.assert_called_once()

    def test_outage_controls_and_screenshot_use_last_displayed_frame(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        tracker = tracker_shell([(True, frame)] + [(False, None)] * 5)
        tracker._recording = True
        tracker._recording_backend = 'camera'
        with (
            patch('pistream.track.cv2.imshow'),
            patch('pistream.track.cv2.waitKey', side_effect=[-1] + list(map(ord, 'rdesq'))),
            patch('pistream.track.cv2.destroyAllWindows'),
        ):
            tracker.run()
        tracker.stop_recording.assert_called_once()
        tracker.reset_tracking.assert_called_once_with(home=True)
        tracker.motors.connect.assert_called_once()
        self.assertIs(tracker.screenshot.call_args.args[0], tracker.process_frame.return_value[0])

    def test_screenshot_before_first_frame_is_skipped(self):
        tracker = tracker_shell([(False, None), (False, None)])
        with (
            patch('pistream.track.cv2.waitKey', side_effect=list(map(ord, 'sq'))),
            patch('pistream.track.cv2.destroyAllWindows'),
        ):
            self.assertEqual(tracker.run(), 0)
        tracker.screenshot.assert_not_called()


class CameraStopRecoveryTests(unittest.TestCase):
    def test_gui_errors_fall_back_to_terminal_without_automatic_stop_attempts(self):
        for operation in ('putText', 'resize', 'imshow', 'waitKey'):
            with self.subTest(operation=operation):
                tracker = tracker_shell()
                tracker.remote_stop_pending = True

                def retry():
                    tracker.remote_stop_pending = False
                    return True

                tracker.stop_recording.side_effect = retry
                with (
                    patch('pistream.track.cv2.imshow'),
                    patch('pistream.track.cv2.waitKey', return_value=-1),
                    patch(f'pistream.track.cv2.{operation}', side_effect=cv2.error('GUI unavailable')),
                    patch('pistream.track.sys.stdin.isatty', return_value=True),
                    patch('builtins.input', side_effect=['anything', 'retry']) as prompt,
                ):
                    self.assertEqual(
                        tracker._recover_camera_stop(display=True, frame=np.zeros((48, 64, 3), dtype=np.uint8)),
                        0,
                    )
                tracker.stop_recording.assert_called_once()
                self.assertEqual(prompt.call_count, 2)
                self.assertFalse(tracker.remote_stop_pending)

    def test_gui_error_without_terminal_preserves_pending_and_exits_nonzero(self):
        tracker = tracker_shell()
        tracker.remote_stop_pending = True
        with (
            patch('pistream.track.cv2.imshow', side_effect=cv2.error('GUI unavailable')),
            patch('pistream.track.sys.stdin.isatty', return_value=False),
            patch('builtins.input') as prompt,
        ):
            self.assertEqual(tracker._recover_camera_stop(display=True), 1)
        self.assertTrue(tracker.remote_stop_pending)
        tracker.stop_recording.assert_not_called()
        prompt.assert_not_called()

    def test_tracking_error_still_exits_nonzero_after_successful_terminal_fallback(self):
        tracker = tracker_shell([RuntimeError('capture failed')])

        def cleanup():
            tracker.remote_stop_pending = True
            tracker.capture = None

        def retry():
            tracker.remote_stop_pending = False
            return True

        tracker.cleanup.side_effect = cleanup
        tracker.stop_recording.side_effect = retry
        with (
            patch('pistream.track.cv2.imshow', side_effect=cv2.error('GUI unavailable')),
            patch('pistream.track.sys.stdin.isatty', return_value=True),
            patch('builtins.input', return_value='retry'),
            patch('pistream.track.cv2.destroyAllWindows'),
        ):
            self.assertEqual(tracker.run(display=True), 1)
        tracker.stop_recording.assert_called_once()
        self.assertFalse(tracker.remote_stop_pending)

    def test_terminal_retries_only_on_explicit_input_until_confirmation(self):
        tracker = tracker_shell()
        tracker.remote_stop_pending = True

        def retry():
            if tracker.stop_recording.call_count == 2:
                tracker.remote_stop_pending = False
                return True
            return False

        tracker.stop_recording.side_effect = retry
        with (
            patch('pistream.track.sys.stdin.isatty', return_value=True),
            patch('builtins.input', side_effect=['anything', 'retry', 'retry']),
        ):
            self.assertEqual(tracker._recover_camera_stop(display=False), 0)
        self.assertEqual(tracker.stop_recording.call_count, 2)

    def test_terminal_force_quit_and_unavailable_input_exit_nonzero(self):
        for interactive, response in ((True, 'quit'), (True, EOFError()), (False, None)):
            with self.subTest(interactive=interactive, response=response):
                tracker = tracker_shell()
                tracker.remote_stop_pending = True
                with (
                    patch('pistream.track.sys.stdin.isatty', return_value=interactive),
                    patch('builtins.input', side_effect=[response]),
                ):
                    self.assertEqual(tracker._recover_camera_stop(display=False), 1)
                tracker.stop_recording.assert_not_called()

    def test_gui_retry_interruption_preserves_pending_stop_until_force_quit(self):
        tracker = tracker_shell()
        tracker.remote_stop_pending = True
        tracker.stop_recording.side_effect = KeyboardInterrupt()
        with (
            patch('pistream.track.cv2.imshow'),
            patch('pistream.track.cv2.waitKey', side_effect=list(map(ord, 'rq'))),
        ):
            self.assertEqual(tracker._recover_camera_stop(display=True), 1)
        self.assertTrue(tracker.remote_stop_pending)
        tracker.stop_recording.assert_called_once()

    def test_run_enters_recovery_after_local_cleanup_and_closes_windows(self):
        tracker = tracker_shell([KeyboardInterrupt()])

        def cleanup():
            tracker.remote_stop_pending = True
            tracker.capture = None

        tracker.cleanup.side_effect = cleanup
        tracker._recover_camera_stop = Mock(return_value=1)
        with patch('pistream.track.cv2.destroyAllWindows') as destroy:
            self.assertEqual(tracker.run(display=False), 1)
        tracker.cleanup.assert_called_once()
        tracker._recover_camera_stop.assert_called_once_with(display=False, frame=None)
        destroy.assert_called_once()


if __name__ == '__main__':
    unittest.main()

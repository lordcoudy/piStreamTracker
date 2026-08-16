"""Recording file listing and path safety."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pistream.recordings import (
    RECORDING_SUFFIXES,
    is_recording_file,
    list_recording_files,
    safe_recording_path,
)


class RecordingSuffixTests(unittest.TestCase):
    def test_h264_mp4_is_a_recording(self):
        self.assertIn('.mp4', RECORDING_SUFFIXES)

    def test_mjpg_fallback_avi_is_a_recording(self):
        self.assertIn('.avi', RECORDING_SUFFIXES)

    def test_is_recording_file_rejects_other_suffixes(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'notes.md'
            p.write_text('x')
            self.assertFalse(is_recording_file(p))

    def test_is_recording_file_accepts_mp4(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'rec_20260101.mp4'
            p.write_bytes(b'x')
            self.assertTrue(is_recording_file(p))

    def test_list_recording_files_includes_mp4(self):
        with TemporaryDirectory() as tmp:
            rec = Path(tmp)
            (rec / 'talk.mp4').write_bytes(b'x')
            (rec / 'notes.md').write_text('no')
            names = {item['name'] for item in list_recording_files(rec)}
        self.assertIn('talk.mp4', names)
        self.assertNotIn('notes.md', names)


class SafeRecordingPathTests(unittest.TestCase):
    def test_returns_path_inside_output_dir(self):
        with TemporaryDirectory() as tmp:
            rec = Path(tmp) / 'recordings'
            rec.mkdir()
            target = rec / 'clip.mp4'
            target.write_bytes(b'x')
            got = safe_recording_path(rec, 'clip.mp4')
            self.assertEqual(got, target.resolve())

    def test_rejects_parent_directory_escape(self):
        with TemporaryDirectory() as tmp:
            rec = Path(tmp) / 'recordings'
            rec.mkdir()
            with self.assertRaises(ValueError):
                safe_recording_path(rec, '../secret.txt')

    def test_rejects_prefix_sibling_directory(self):
        """/recordings must not match /recordings_evil via startswith."""
        with TemporaryDirectory() as tmp:
            rec = Path(tmp) / 'recordings'
            rec.mkdir()
            evil = Path(tmp) / 'recordings_evil'
            evil.mkdir()
            (evil / 'x.mp4').write_bytes(b'x')
            with self.assertRaises(ValueError):
                safe_recording_path(rec, '../recordings_evil/x.mp4')

    def test_rejects_absolute_path(self):
        with TemporaryDirectory() as tmp:
            rec = Path(tmp) / 'recordings'
            rec.mkdir()
            with self.assertRaises(ValueError):
                safe_recording_path(rec, '/etc/passwd')


if __name__ == '__main__':
    unittest.main()

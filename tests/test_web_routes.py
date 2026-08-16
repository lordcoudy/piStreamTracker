"""Drive shipped Flask listing/path routes (same handlers the UI calls)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote


class WebRecordingsRouteTests(unittest.TestCase):
    def test_list_includes_mp4_and_avi_not_other_suffixes(self):
        from pistream import web_app

        with TemporaryDirectory() as tmp:
            rec = Path(tmp)
            (rec / 'talk.mp4').write_bytes(b'x')
            (rec / 'legacy.avi').write_bytes(b'y')
            (rec / 'notes.md').write_text('no')
            web_app._config = {'tracker': {'output_dir': tmp}}
            client = web_app.app.test_client()
            payload = client.get('/api/recordings').get_json()
            names = {item['name'] for item in payload['files']}
        self.assertIn('talk.mp4', names)
        self.assertIn('legacy.avi', names)
        self.assertNotIn('notes.md', names)

    def test_download_rejects_parent_escape(self):
        from pistream import web_app

        with TemporaryDirectory() as tmp:
            web_app._config = {'tracker': {'output_dir': tmp}}
            client = web_app.app.test_client()
            response = client.get('/api/recordings/' + quote('../secret.txt', safe=''))
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json().get('status'), 'error')

    def test_delete_rejects_absolute_path(self):
        from pistream import web_app

        with TemporaryDirectory() as tmp:
            web_app._config = {'tracker': {'output_dir': tmp}}
            with web_app.app.app_context():
                body, status = web_app.api_recordings_delete('/etc/passwd')
        self.assertEqual(status, 403)
        self.assertEqual(body.get_json().get('status'), 'error')

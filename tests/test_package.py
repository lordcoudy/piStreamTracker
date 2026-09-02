"""Package layout after the Phase 4 split."""

import unittest
from pathlib import Path


class PackageLayoutTests(unittest.TestCase):
    def setUp(self):
        self.pkg = Path(__file__).resolve().parents[1] / 'pistream'

    def test_template_has_overlay_toggle_and_static_links(self):
        html = (self.pkg / 'templates' / 'index.html').read_text()
        self.assertIn('overlay-toggle', html)
        self.assertIn("url_for('static', filename='style.css')", html)
        self.assertIn("url_for('static', filename='app.js')", html)
        self.assertNotIn('<style>', html)
        self.assertNotIn('<script>', html)

    def test_static_assets_exist(self):
        self.assertTrue((self.pkg / 'static' / 'style.css').is_file())
        self.assertTrue((self.pkg / 'static' / 'app.js').is_file())
        js = (self.pkg / 'static' / 'app.js').read_text()
        self.assertIn('function toggleOverlay', js)

    def test_api_helper_parses_json_defensively(self):
        js = (self.pkg / 'static' / 'app.js').read_text()
        self.assertIn('JSON.parse', js)
        self.assertIn('Invalid JSON', js)
        self.assertIn("typeof res.running !== 'boolean'", js)
        self.assertIn("typeof res.fps !== 'number'", js)

    def test_project_root_finds_config(self):
        from pistream.config import project_root
        root = project_root()
        self.assertTrue((root / 'config.yaml').is_file())
        self.assertTrue((root / 'tracker.py').is_file())

    def test_process_frame_aims_then_levels_then_draws(self):
        src = (self.pkg / 'track.py').read_text()
        aim = src.index('self._update_aim(')
        level = src.index('leveled = self._level_frame(')
        draw = src.index('annotated = self._draw(')
        self.assertLess(aim, level)
        self.assertLess(level, draw)

    def test_index_serves_extracted_template(self):
        from pistream.web_app import app
        client = app.test_client()
        response = client.get('/')
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn('overlay-toggle', body)
        self.assertIn('style.css', body)
        self.assertIn('app.js', body)


if __name__ == '__main__':
    unittest.main()

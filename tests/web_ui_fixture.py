"""Hardware-free Flask fixture for the standalone browser regression checks."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import jsonify  # noqa: E402
from pistream import web_app  # noqa: E402


class PendingCameraLifecycle:
    stopping = False

    def __init__(self):
        self.stop_calls = 0
        self.tracker = SimpleNamespace(
            running=False,
            recording=True,
            current_detection=None,
            capture=None,
            motors=SimpleNamespace(connected=False),
            fps=0.0,
            zoom_level=1.0,
            horizon_correction=False,
            recording_error=None,
            remote_stop_pending=True,
            cleanup_pending=True,
        )

    def start(self):
        if self.tracker:
            return {'status': 'recording_stop_pending'}
        return {'status': 'ok'}

    def stop(self):
        self.stop_calls += 1
        if self.stop_calls == 1:
            return {'status': 'recording_stop_pending'}
        self.tracker = None
        return {'status': 'ok'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5089)
    args = parser.parse_args()
    lifecycle = PendingCameraLifecycle()
    web_app._lifecycle = lifecycle
    web_app._config = {'camera': {'resolution': {'width': 1280, 'height': 960}}}
    web_app.app.config['TEMPLATES_AUTO_RELOAD'] = True
    web_app.app.view_functions['video_feed'] = lambda: web_app.Response(
        web_app._placeholder_jpeg(), content_type='image/jpeg'
    )

    @web_app.app.get('/__fixture/state')
    def fixture_state():
        return jsonify({'stop_calls': lifecycle.stop_calls})

    web_app.app.run(host='127.0.0.1', port=args.port, threaded=True, debug=False)


if __name__ == '__main__':
    main()

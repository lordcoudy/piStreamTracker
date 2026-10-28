"""Bounded settings requests to the configured camera host."""

import json
import urllib.request

from pistream.jsonutil import load_json_object


def request_camera_settings(base_url: str, token: str = '', settings: dict | None = None):
    headers = {'Accept': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    data = None
    if settings is not None:
        data = json.dumps(settings, allow_nan=False).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(f'{base_url.rstrip("/")}/camera/settings',
                                 data=data, headers=headers)
    # A configuration rebuild can take several frames. Never retry a mutation:
    # timeout does not tell us whether the camera already saved the change.
    with urllib.request.urlopen(req, timeout=12) as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise ValueError('Camera settings response is too large')
    payload = load_json_object(raw)
    settings = payload.get('settings')
    if not isinstance(settings, dict) or not isinstance(settings.get('resolution'), dict):
        raise ValueError('Invalid camera settings response')
    resolution = settings['resolution']
    if not all(isinstance(resolution.get(key), (int, float)) for key in ('width', 'height')):
        raise ValueError('Invalid camera resolution response')
    if not isinstance(settings.get('framerate'), (int, float)):
        raise ValueError('Invalid camera framerate response')
    return payload

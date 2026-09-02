"""Parse JSON object bodies from camera/tracker HTTP APIs."""

from __future__ import annotations

import json
from typing import Union


def load_json_object(raw: Union[bytes, str]) -> dict:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('response is not valid JSON') from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError('response is not valid JSON') from exc
    if not isinstance(data, dict):
        raise ValueError('response is not a JSON object')
    return data

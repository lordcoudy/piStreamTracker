"""Parse JSON object bodies from camera/tracker HTTP APIs."""

from __future__ import annotations

import json
from typing import Union


def load_json_object(raw: Union[bytes, str]) -> dict:
    try:
        data = json.loads(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('response is not valid JSON') from exc
    if not isinstance(data, dict):
        raise ValueError('response is not a JSON object')
    return data

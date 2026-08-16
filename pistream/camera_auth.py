"""Optional shared secret for the camera record API."""

from __future__ import annotations

import hmac
from typing import Optional


def extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, rest = authorization.partition(' ')
    if scheme.lower() != 'bearer' or not rest:
        return None
    return rest.strip()


def auth_headers(token: Optional[str]) -> dict:
    if token:
        return {'Authorization': f'Bearer {token}'}
    return {}


def token_ok(provided: Optional[str], expected: Optional[str]) -> bool:
    if not expected:
        return True
    if provided is None:
        return False
    return hmac.compare_digest(provided, expected)

"""Horizon tilt from MoveNet left/right shoulder and hip pairs."""

from __future__ import annotations

import math
from typing import Optional, Sequence


def tilt_degrees(keypoints: Sequence[Sequence[float]], threshold: float) -> Optional[float]:
    """Return mean tilt in degrees, or None if no pair is confident enough.

    Positive = clockwise (right side lower). Uses keypoints 5/6 (shoulders)
    and 11/12 (hips).
    """
    if keypoints is None or len(keypoints) < 13:
        return None

    angles = []
    for li, ri in ((5, 6), (11, 12)):
        lx, ly, lc = keypoints[li]
        rx, ry, rc = keypoints[ri]
        if lc >= threshold and rc >= threshold:
            dx = float(rx - lx)
            dy = float(ry - ly)
            if abs(dx) > 5:
                angles.append(math.degrees(math.atan2(dy, dx)))

    if not angles:
        return None
    return sum(angles) / len(angles)

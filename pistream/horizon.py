"""Software horizon leveling from pose keypoints (no IMU)."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np

SHOULDER_PAIR = (5, 6)
HIP_PAIR = (11, 12)
_MIN_DX = 5.0
_OUTLIER_FACTOR = 1.5


def estimate_roll_deg(
    keypoints: Optional[Sequence[Sequence[float]]], threshold: float
) -> Optional[float]:
    """Return mean tilt in degrees, or None if no pair is confident enough.

    Positive = clockwise (right side lower). Uses keypoints 5/6 (shoulders)
    and 11/12 (hips).
    """
    if keypoints is None or len(keypoints) < 13:
        return None

    angles = []
    for li, ri in (SHOULDER_PAIR, HIP_PAIR):
        lx, ly, lc = keypoints[li]
        rx, ry, rc = keypoints[ri]
        if lc < threshold or rc < threshold:
            continue
        dx = float(rx - lx)
        dy = float(ry - ly)
        if abs(dx) <= _MIN_DX:
            continue
        angles.append(math.degrees(math.atan2(dy, dx)))

    if not angles:
        return None
    return float(sum(angles) / len(angles))


def tilt_degrees(
    keypoints: Optional[Sequence[Sequence[float]]], threshold: float
) -> Optional[float]:
    """Alias for estimate_roll_deg (kept for existing call sites and tests)."""
    return estimate_roll_deg(keypoints, threshold)


def step_angle(
    enabled: bool,
    raw_deg: Optional[float],
    ema: float,
    *,
    alpha: float,
    max_angle: float,
) -> float:
    if not enabled:
        return 0.0
    if raw_deg is None:
        return ema
    if abs(raw_deg) > max_angle * _OUTLIER_FACTOR:
        return ema
    raw_deg = max(-max_angle, min(max_angle, raw_deg))
    return alpha * raw_deg + (1.0 - alpha) * ema


def fill_crop_scale(angle_deg: float, width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 1.0
    a = math.radians(angle_deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return max(c + s * height / width, c + s * width / height)


def level_affine(
    width: int, height: int, angle_deg: float, fill_crop: bool = True
) -> np.ndarray:
    cx, cy = width / 2.0, height / 2.0
    scale = fill_crop_scale(angle_deg, width, height) if fill_crop else 1.0
    return cv2.getRotationMatrix2D((cx, cy), -angle_deg, scale)


def transform_points(points_xy: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (M @ np.hstack([pts, ones]).T).T


def warp_level(
    frame: np.ndarray,
    angle_deg: float,
    *,
    fill_crop: bool = True,
    min_apply: float = 0.5,
) -> np.ndarray:
    if abs(angle_deg) < min_apply:
        return frame
    h, w = frame.shape[:2]
    M = level_affine(w, h, angle_deg, fill_crop=fill_crop)
    return cv2.warpAffine(
        frame, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

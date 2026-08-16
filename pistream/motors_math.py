"""Pure motor command math (no USB / ev3-dc)."""

from typing import Optional, Tuple


def compute_axis_command(
    shift: int,
    deadzone: int,
    invert: bool,
    frame_dim: int,
    scale: float,
    speed_factor: float,
    max_speed: int,
) -> Optional[Tuple[int, int]]:
    """Return (degrees, speed) or None if the axis should stop."""
    if abs(shift) < deadzone:
        return None
    if frame_dim <= 0 or scale <= 0:
        return None
    speed = int(min(abs(shift) / 100.0 * speed_factor, max_speed))
    degrees = int((-shift if invert else shift) / (frame_dim / scale))
    if speed <= 0:
        return None
    return degrees, speed


def motors_held(now: float, hold_until: float) -> bool:
    return now < hold_until

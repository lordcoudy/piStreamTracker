"""Preview sizing, encode gating, and recording fps limits (no OpenCV)."""

from __future__ import annotations

from urllib.parse import urlsplit


def accept_new_frame(prev, ret: bool, frame):
    """Return (frame_or_prev, is_new). Same capture buffer is not a new frame."""
    if not ret or frame is None:
        return prev, False
    if prev is frame:
        return prev, False
    return frame, True


def preview_target_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
    """Scale so the long side is at most max_edge. Keep aspect ratio."""
    if width <= 0 or height <= 0 or max_edge <= 0:
        return width, height
    long_side = max(width, height)
    if long_side <= max_edge:
        return width, height
    scale = max_edge / float(long_side)
    return max(1, int(width * scale)), max(1, int(height * scale))


def preview_gate(
    last_seq: int,
    current_seq: int,
    now: float,
    last_emit_at: float,
    min_interval: float,
) -> bool:
    """True when a new frame should be JPEG-encoded for the web preview."""
    if current_seq == last_seq:
        return False
    if now - last_emit_at < min_interval:
        return False
    return True


def capped_recording_fps(requested: float, source_fps: float) -> float:
    """Never write recordings faster than the camera produces frames."""
    src = float(source_fps) if source_fps and source_fps > 0 else 30.0
    req = float(requested) if requested and requested > 0 else src
    return max(1.0, min(req, src))


def camera_stream_url(config: dict) -> str:
    """One configured source for tracking and the overlay-free preview."""
    custom = (config.get('tracker') or {}).get('stream_url')
    if custom:
        return custom
    return f'{camera_base_url(config)}/stream'


def camera_base_url(config: dict) -> str:
    """Camera controls/files use the stream origin, including localhost overrides."""
    custom = (config.get('tracker') or {}).get('stream_url')
    if custom:
        parsed = urlsplit(custom)
        if parsed.scheme in {'http', 'https'} and parsed.netloc:
            return f'{parsed.scheme}://{parsed.netloc}'
    net = config.get('network') or {}
    cam = config.get('camera') or {}
    ip = net.get('camera_ip') or '192.168.100.1'
    port = cam.get('port') or 8000
    return f"http://{ip}:{port}"

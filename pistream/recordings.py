"""Recording file listing and path safety."""

import json
import urllib.request
from datetime import datetime
from pathlib import Path

from pistream.camera_auth import auth_headers

RECORDING_SUFFIXES = {'.avi', '.jpg', '.png', '.txt', '.mp4'}


def is_recording_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in RECORDING_SUFFIXES


def file_entry(path: Path) -> dict:
    stat = path.stat()
    size = stat.st_size
    if size < 1024:
        size_str = f"{size} B"
    elif size < 1024 * 1024:
        size_str = f"{size / 1024:.1f} KB"
    else:
        size_str = f"{size / (1024 * 1024):.1f} MB"
    return {
        'name': path.name,
        'size': size_str,
        'bytes': stat.st_size,
        'date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
        'type': path.suffix.lower().lstrip('.'),
    }


def list_recording_files(rec_path: Path) -> list:
    if not rec_path.exists():
        return []
    files = []
    for path in sorted(rec_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if is_recording_file(path):
            files.append(file_entry(path))
    return files


def fetch_remote_recordings(base_url: str, token: str = '') -> list:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/record/list",
        headers=auth_headers(token or None),
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    files = data.get('files') if isinstance(data, dict) else None
    return files if isinstance(files, list) else []


def safe_recording_path(output_dir: Path, filename: str) -> Path:
    """Resolve filename under output_dir or raise ValueError."""
    name = Path(filename)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Invalid path')

    rec_path = Path(output_dir).resolve()
    file_path = (rec_path / name).resolve()
    if not file_path.is_relative_to(rec_path):
        raise ValueError('Invalid path')
    return file_path

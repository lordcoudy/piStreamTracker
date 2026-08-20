"""Recording file listing and path safety."""

import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

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
        'mtime': stat.st_mtime,
        'date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
        'type': path.suffix.lower().lstrip('.'),
    }


def list_recording_files(rec_path: Path) -> list:
    if not rec_path.exists():
        return []
    files = []
    for path in rec_path.iterdir():
        try:
            if is_recording_file(path):
                files.append(file_entry(path))
        except FileNotFoundError:
            continue  # A recorder or another request removed it mid-listing.
    return sorted(files, key=lambda item: item['mtime'], reverse=True)


def fetch_remote_recordings(base_url: str, token: str = '') -> list:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/record/list",
        headers=auth_headers(token or None),
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    files = data.get('files') if isinstance(data, dict) else None
    return files if isinstance(files, list) else []


def open_remote_recording(base_url: str, filename: str, token: str = ''):
    """Open a camera-hosted recording for streaming to the web client."""
    encoded = quote(filename, safe='')
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/record/files/{encoded}",
        headers=auth_headers(token or None),
    )
    return urllib.request.urlopen(req, timeout=10)


def delete_remote_recording(base_url: str, filename: str, token: str = '') -> None:
    """Delete one camera-hosted recording."""
    encoded = quote(filename, safe='')
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/record/files/{encoded}",
        method='DELETE',
        headers=auth_headers(token or None),
    )
    with urllib.request.urlopen(req, timeout=10):
        return


def safe_recording_path(output_dir: Path, filename: str) -> Path:
    """Resolve filename under output_dir or raise ValueError."""
    name = Path(filename)
    if name.is_absolute() or len(name.parts) != 1 or '..' in name.parts:
        raise ValueError('Invalid path')

    rec_path = Path(output_dir).resolve()
    file_path = (rec_path / name).resolve()
    if not file_path.is_relative_to(rec_path):
        raise ValueError('Invalid path')
    return file_path

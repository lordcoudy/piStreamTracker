"""Recording file listing and path safety."""

from pathlib import Path

RECORDING_SUFFIXES = {'.avi', '.jpg', '.png', '.txt', '.mp4'}


def is_recording_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in RECORDING_SUFFIXES


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

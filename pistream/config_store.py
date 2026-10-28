"""Coordinate YAML saves between the camera and tracker processes."""

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def config_file_lock(path):
    """Lock a stable sibling file: replacing the YAML must not replace its lock."""
    import fcntl

    path = Path(path).resolve()
    lock_path = path.with_name(f'.{path.name}.lock')
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

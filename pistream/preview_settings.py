"""Validation and atomic tracker-host persistence for browser preview settings."""

import os
import stat
import tempfile
from pathlib import Path

import yaml

from pistream.config_store import config_file_lock


def preview_candidate(current, patch):
    if not isinstance(patch, dict) or patch.keys() - {'max_edge', 'quality'}:
        raise ValueError('Expected only max_edge and quality preview settings')
    candidate = {**current, **patch}
    for name, minimum, maximum in (('max_edge', 0, 16384), ('quality', 1, 100)):
        value = candidate[name]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f'{name} must be an integer between {minimum} and {maximum}')
    return candidate


def save_preview_settings(config_path, settings):
    """Preserve latest unrelated YAML fields and replace only after a durable write."""
    path = Path(config_path)
    with config_file_lock(path):
        _save_locked(path, settings)


def _save_locked(path, settings):
    if path.exists():
        with path.open(encoding='utf-8') as handle:
            document = yaml.safe_load(handle)
        if document is None:
            document = {}
    else:
        document = {}
    if not isinstance(document, dict):
        raise ValueError('Configuration root must be a mapping')
    web = document.setdefault('web', {})
    if not isinstance(web, dict):
        raise ValueError('web must be a mapping')
    web.update(preview_max_edge=settings['max_edge'], preview_quality=settings['quality'])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', encoding='utf-8', delete=False) as handle:
            temporary = Path(handle.name)
            yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=True)
            if path.exists():
                os.fchmod(handle.fileno(), stat.S_IMODE(path.stat().st_mode))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

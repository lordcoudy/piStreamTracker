"""Camera and web processes must not overwrite each other's saved YAML fields."""

import multiprocessing

import yaml

from pistream.config_store import config_file_lock


def _write_field(path, key, ready, entered, release):
    ready.set()
    with config_file_lock(path):
        entered.set()
        assert release.wait(3)
        with open(path) as stream:
            data = yaml.safe_load(stream)
        data[key] = True
        with open(path, 'w') as stream:
            yaml.safe_dump(data, stream)


def test_separate_process_saves_keep_both_configuration_sections(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    path = tmp_path / 'config.yaml'
    path.write_text('unrelated: preserved\n')
    first_ready, second_ready = ctx.Event(), ctx.Event()
    first_entered, second_entered = ctx.Event(), ctx.Event()
    release_first, release_second = ctx.Event(), ctx.Event()
    release_second.set()
    first = ctx.Process(target=_write_field, args=(path, 'camera', first_ready, first_entered, release_first))
    second = ctx.Process(target=_write_field, args=(path, 'web', second_ready, second_entered, release_second))
    first.start()
    try:
        assert first_entered.wait(3)
        second.start()
        assert second_ready.wait(3)
        assert not second_entered.wait(0.1), 'The second process must wait before reading YAML'
        release_first.set()
        first.join(3)
        second.join(3)
        assert first.exitcode == second.exitcode == 0
        assert yaml.safe_load(path.read_text()) == {'unrelated': 'preserved', 'camera': True, 'web': True}
    finally:
        release_first.set()
        for process in (first, second):
            if process.is_alive():
                process.terminate()
                process.join(3)

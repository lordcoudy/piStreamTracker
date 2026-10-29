"""Exercise launchers using tiny child processes, without camera or GPIO imports."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


@pytest.fixture
def launcher(tmp_path):
    if os.geteuid() == 0:
        pytest.skip('Launchers intentionally reject root')
    root = tmp_path / 'checkout with spaces'
    root.mkdir()
    for script in ('setup.sh', 'run_standalone.sh'):
        shutil.copy(ROOT / script, root / script)
    config = root / 'custom pi4.yaml'
    config.write_text('camera: {}\n')
    events = root / 'events.jsonl'
    fake_python = f'''#!{sys.executable}
import json, os, pathlib, signal, sys, time
if sys.argv[1] == '-c':
    code = sys.argv.pop(2)
    sys.argv.pop(1)
    exec(code)
    raise SystemExit
with open(os.environ['LAUNCH_EVENTS'], 'a') as out:
    out.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == '-m':
    raise SystemExit
if os.environ.get('LAUNCH_ENV_EVENTS'):
    with open(os.environ['LAUNCH_ENV_EVENTS'], 'a') as out:
        out.write(json.dumps({{name: os.environ.get(name) for name in
            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}) + '\\n')
def stop(*args):
    with open(os.environ['LAUNCH_EVENTS'], 'a') as out:
        out.write(json.dumps(['stopped', pathlib.Path(sys.argv[1]).name]) + '\\n')
    raise SystemExit
signal.signal(signal.SIGTERM, stop)
if pathlib.Path(sys.argv[1]).name == 'web.py' and os.environ.get('WEB_EXIT'):
    raise SystemExit(int(os.environ['WEB_EXIT']))
while True:
    time.sleep(.02)
'''
    executable(root / '.venv/bin/python', fake_python)
    executable(root / 'bin/systemctl', '#!/bin/sh\necho "${FAKE_SERVICE_STATE:-inactive}"\n')
    executable(root / 'bin/python3', f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    env = dict(os.environ, PATH=f'{root / "bin"}:{os.environ["PATH"]}',
               LAUNCH_EVENTS=str(events))
    return root, config, events, env


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_child_failure_stops_other_child_and_shares_absolute_config(launcher):
    root, config, events, env = launcher
    result = subprocess.run(['bash', str(root / 'run_standalone.sh'), '--config', str(config)],
                            env=dict(env, WEB_EXIT='7'), capture_output=True, text=True, timeout=5)
    assert result.returncode == 7, result.stderr
    rows = read_events(events)
    children = [row for row in rows if row[0] != 'stopped']
    assert sorted(Path(row[0]).name for row in children) == ['camera.py', 'web.py']
    assert all(row[1:] == ['--config', str(config.resolve())] for row in children)
    assert ['stopped', 'camera.py'] in rows


def test_signal_shuts_down_both_children(launcher):
    root, config, events, env = launcher
    child = subprocess.Popen(['bash', str(root / 'run_standalone.sh'), '--config', str(config)],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 3
        while (not events.exists() or len(read_events(events)) < 2) and time.monotonic() < deadline:
            time.sleep(.02)
        assert events.exists() and len(read_events(events)) == 2
        child.terminate()
        child.communicate(timeout=5)
        assert child.returncode == 143
        assert sorted(row[1] for row in read_events(events) if row[0] == 'stopped') == [
            'camera.py', 'web.py',
        ]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_active_service_prevents_second_camera(launcher):
    root, config, events, env = launcher
    result = subprocess.run(['bash', str(root / 'run_standalone.sh'), '--config', str(config)],
                            env=dict(env, FAKE_SERVICE_STATE='active'), capture_output=True,
                            text=True, timeout=3)
    assert result.returncode == 1
    assert 'already running' in result.stderr
    assert not events.exists()


def test_setup_status_and_dry_run_do_not_bootstrap_and_preserve_custom_config(launcher):
    root, config, events, env = launcher
    for command in (['status'], ['install', '--dry-run']):
        result = subprocess.run(['bash', str(root / 'setup.sh'), '--standalone', '--config',
                                 str(config), *command], env=env, capture_output=True,
                                text=True, timeout=3)
        assert result.returncode == 0, result.stderr
        assert not (root / 'venv').exists()
        args = read_events(events)[-1]
        assert args[:2] == ['-m', 'pistream.install']
        assert args[args.index('--config') + 1] == str(config)
        if command[0] == 'install':
            assert args[args.index('--role') + 1] == 'standalone'


@pytest.mark.parametrize('exit_code', [0, 7])
def test_setup_model_check_uses_runtime_loader_and_propagates_failure(tmp_path, exit_code):
    source = (ROOT / 'setup.sh').read_text()
    # Run the real bootstrap function alone, without apt, pip or network access.
    function = 'download_model() {' + source.split('download_model() {', 1)[1].split(
        '\narray_contains()', 1,
    )[0]
    executable(tmp_path / 'venv/bin/python', f'''#!{sys.executable}
import pathlib, sys
pathlib.Path('loader-request').write_text(sys.stdin.read())
pathlib.Path('loader-args').write_text(' '.join(sys.argv[1:]))
raise SystemExit({exit_code})
''')
    executable(tmp_path / 'bin/wget', '#!/bin/sh\nexit 99\n')
    result = subprocess.run(['bash', '-c', 'set -e\nprint_step() { echo "$1"; }\n'
                             + function + '\ndownload_model standalone\n'], cwd=tmp_path,
                            env=dict(os.environ, PATH=f'{tmp_path / "bin"}:{os.environ["PATH"]}'),
                            capture_output=True, text=True, timeout=3)
    assert (tmp_path / 'loader-request').exists(), result.stderr
    request = (tmp_path / 'loader-request').read_text()
    assert 'PoseDetector' in request and '.ready' in request
    assert 'standalone' in (tmp_path / 'loader-args').read_text()
    assert result.returncode == exit_code
    assert ('Model ready' in result.stdout) is (exit_code == 0)


def bootstrap_function(name):
    source = (ROOT / 'setup.sh').read_text()
    return name + '() {' + source.split(name + '() {', 1)[1].split('\n}\n', 1)[0] + '\n}\n'


@pytest.mark.parametrize('backend', ['ev3', 'gpio_servo', 'none'])
def test_standalone_packages_follow_configured_motor_backend(tmp_path, backend):
    commands = tmp_path / 'apt.jsonl'
    executable(tmp_path / 'bin/sudo', f'''#!{sys.executable}
import json, pathlib, sys
with pathlib.Path({str(commands)!r}).open('a') as out:
    out.write(json.dumps(sys.argv[1:]) + '\\n')
''')
    functions = bootstrap_function('install_system_deps') + bootstrap_function('install_motor_system_deps')
    result = subprocess.run(['bash', '-c', 'set -e\nprint_step() { :; }\n' + functions
                             + f'\ninstall_system_deps standalone\ninstall_motor_system_deps {backend}\n'],
                            cwd=tmp_path, env=dict(os.environ, PATH=f'{tmp_path / "bin"}:{os.environ["PATH"]}'),
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
    args = [argument for command in read_events(commands) for argument in command]
    assert 'python3-picamera2' in args and 'rpicam-apps' in args
    assert ('pigpiod' in args) is (backend == 'gpio_servo')
    assert ('python3-pigpio' in args) is (backend == 'gpio_servo')
    assert 'network-manager' not in args


@pytest.mark.parametrize('backend', ['ev3', 'gpio_servo', 'none'])
def test_standalone_python_extra_and_import_smoke_follow_active_config(tmp_path, backend):
    config = tmp_path / 'active config.yaml'
    config.write_text(f'motors: {{backend: {backend}}}\n')
    commands = tmp_path / 'python-deps.jsonl'
    executable(tmp_path / 'bin/pip', f'''#!{sys.executable}
import json, pathlib, sys
with pathlib.Path({str(commands)!r}).open('a') as out:
    out.write(json.dumps(['pip', *sys.argv[1:]]) + '\\n')
''')
    executable(tmp_path / 'bin/python', f'''#!{sys.executable}
import json, pathlib, sys
code = sys.argv[2]
with pathlib.Path({str(commands)!r}).open('a') as out:
    out.write(json.dumps(['python', code]) + '\\n')
if 'motor_backend' in code:
    sys.argv = ['-c', *sys.argv[3:]]
    exec(code)
''')
    activate = tmp_path / 'venv/bin/activate'
    activate.parent.mkdir(parents=True)
    activate.write_text('')
    functions = bootstrap_function('configured_motor_backend') + bootstrap_function('install_python_deps')
    result = subprocess.run(['bash', '-c', 'set -e\nprint_step() { :; }\n'
                             'check_platform() { return 1; }\n' + functions
                             + '\ninstall_python_deps standalone "$1"\n', 'test', str(config)],
                            cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(ROOT),
                                                  PATH=f'{tmp_path / "bin"}:{os.environ["PATH"]}'),
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
    rows = read_events(commands)
    packages = [arg for row in rows if row[0] == 'pip' for arg in row]
    smoke = '\n'.join(row[1] for row in rows if row[0] == 'python' and 'motor_backend' not in row[1])
    assert ('.[standalone]' in packages) is (backend == 'gpio_servo')
    assert ('.' in packages) is (backend != 'gpio_servo')
    assert ('import pigpio' in smoke) is (backend == 'gpio_servo')
    assert ('import ev3_dc' in smoke) is (backend == 'ev3')
    assert all(name in smoke for name in ('picamera2', 'libcamera', 'cv2', 'numpy', 'tflite_runtime'))


def test_model_preload_uses_selected_config_thread_budget_and_model_path(tmp_path):
    config = tmp_path / 'pi4 ev3.yaml'
    config.write_text('tracker:\n  movenet: {threads: 1, model_path: /tmp/custom-model.tflite}\n')
    executable(tmp_path / 'venv/bin/python', f'''#!{sys.executable}
import json, pathlib, sys, types
detector_module = types.ModuleType('pistream.detect')
class PoseDetector:
    ready = True
    def __init__(self, **kwargs):
        pathlib.Path('detector-options.json').write_text(json.dumps(kwargs))
detector_module.PoseDetector = PoseDetector
sys.modules['pistream.detect'] = detector_module
sys.argv = sys.argv[1:]
exec(sys.stdin.read())
''')
    result = subprocess.run(['bash', '-c', 'set -e\nprint_step() { :; }\n'
                             + bootstrap_function('download_model')
                             + '\ndownload_model standalone "$1"\n', 'test', str(config)],
                            cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(ROOT)),
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / 'detector-options.json').read_text()) == {
        'threads': 1, 'model_path': '/tmp/custom-model.tflite',
    }


@pytest.mark.parametrize('override,expected', [(None, '2'), ('3', '3')])
def test_ev3_launcher_needs_no_pigpiod_and_preserves_explicit_thread_limits(launcher, override, expected):
    root, config, events, env = launcher
    config.write_text('motors: {backend: ev3}\ntracker:\n  movenet: {threads: 1}\n')
    calls = root / 'systemctl-calls'
    executable(root / 'bin/systemctl', f'#!/bin/sh\necho "$*" >> "{calls}"\necho inactive\n')
    env = dict(env, WEB_EXIT='7', LAUNCH_ENV_EVENTS=str(root / 'thread-env.jsonl'))
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env.pop(name, None)
        if override is not None:
            env[name] = override
    result = subprocess.run(['bash', str(root / 'run_standalone.sh'), '--config', str(config)],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 7, result.stderr
    assert 'pigpiod' not in calls.read_text()
    assert all(set(row.values()) == {expected} for row in read_events(root / 'thread-env.jsonl'))

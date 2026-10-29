"""The one-Pi EV3 profile keeps camera traffic local and the Pi 4 workload bounded."""

from pathlib import Path

from pistream.config import camera_bind_host, load_config, motor_backend, web_bind_host
from pistream.preview import camera_base_url, camera_stream_url


def profile():
    return load_config(str(Path(__file__).resolve().parents[1] / 'config.pi4-ev3.yaml'))


def test_camera_capture_controls_and_recordings_share_the_local_origin():
    config = profile()
    assert camera_bind_host(config) == '127.0.0.1'
    assert camera_stream_url(config) == 'http://127.0.0.1:8000/stream'
    assert camera_base_url(config) == 'http://127.0.0.1:8000'
    assert web_bind_host(config) == '0.0.0.0'
    assert config['tracker']['recording_mode'] == 'camera'
    assert Path(config['camera']['recording_dir']).resolve() != Path(config['tracker']['output_dir']).resolve()


def test_ev3_profile_keeps_hd_capture_and_a_single_inference_thread():
    config = profile()
    assert motor_backend(config) == 'ev3'
    assert config['ev3']['enabled'] is True
    assert config['ev3']['keep_alive_interval'] == 60
    assert config['camera']['resolution']['width'] >= 1280
    assert config['camera']['resolution']['height'] >= 720
    assert config['camera']['framerate'] >= 25
    assert config['tracker']['recording_fps'] == config['camera']['framerate']
    assert config['tracker']['movenet']['threads'] == 1
    assert config['web']['preview_max_fps'] <= 10
    assert config['web']['autostart_delay'] == 120

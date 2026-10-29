"""Preview saves are local, atomic, and visible to already connected viewers."""

import copy
import io
import stat
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
import yaml

from pistream import web_app


@pytest.fixture
def preview(monkeypatch, tmp_path):
    path = tmp_path / 'active.yaml'
    path.write_text('web:\n  preview_quality: 70\n  preview_max_edge: 640\n  overlay: false\n'
                    'camera:\n  untouched: 42\n', encoding='utf-8')
    path.chmod(0o640)
    config = {'web': {'preview_quality': 70, 'preview_max_edge': 640, 'overlay': False}}
    lifecycle = SimpleNamespace(tracker=None, start=Mock(), stop=Mock(),
                                cancel_scheduled_start=Mock())
    monkeypatch.setattr(web_app, '_config', config)
    monkeypatch.setattr(web_app, '_config_path', path, raising=False)
    monkeypatch.setattr(web_app, '_lifecycle', lifecycle)
    monkeypatch.setattr(web_app, '_overlay_enabled', False)
    return web_app.app.test_client(), path, config, lifecycle


def test_get_and_partial_save_preserve_latest_yaml_and_file_mode(preview):
    client, path, config, _lifecycle = preview
    assert client.get('/api/preview/settings').get_json()['settings'] == {
        'max_edge': 640, 'quality': 70,
    }
    document = yaml.safe_load(path.read_text())
    document['web']['external_change'] = 'preserve me'
    path.write_text(yaml.safe_dump(document))
    response = client.post('/api/preview/settings', json={'quality': 90})
    assert response.status_code == 200
    assert response.get_json()['settings'] == {'max_edge': 640, 'quality': 90}
    assert response.get_json()['saved'] is True
    saved = yaml.safe_load(path.read_text())
    assert saved['web'] == {
        'preview_quality': 90, 'preview_max_edge': 640,
        'overlay': False, 'external_change': 'preserve me',
    }
    assert saved['camera'] == {'untouched': 42}
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert config['web']['preview_quality'] == 90
    assert not list(path.parent.glob('.*.tmp'))


@pytest.mark.parametrize('data', [
    {'max_edge': True}, {'quality': False}, {'quality': 0}, {'quality': 101},
    {'max_edge': -1}, {'max_edge': 16385}, {'max_edge': 12.5}, {'quality': 70.5},
    {'quality': float('nan')}, {'max_edge': float('inf')}, {'quality': '80'},
    {'quality': None}, {'other': 1}, {'preview_quality': 80}, [], None,
])
def test_invalid_preview_settings_do_not_change_disk_or_memory(preview, data):
    client, path, config, _lifecycle = preview
    before_disk, before_config = path.read_bytes(), copy.deepcopy(config)
    response = client.post('/api/preview/settings', json=data)
    assert response.status_code == 400
    assert path.read_bytes() == before_disk
    assert config == before_config


@pytest.mark.parametrize('failure', ['replace', 'fsync'])
def test_failed_save_leaves_memory_and_original_file_unchanged(preview, failure):
    client, path, config, _lifecycle = preview
    before_disk, before_config = path.read_bytes(), copy.deepcopy(config)
    with patch(f'os.{failure}', side_effect=OSError('disk error')):
        response = client.post('/api/preview/settings', json={'max_edge': 960})
    assert response.status_code == 500
    assert response.get_json()['status'] == 'error'
    assert path.read_bytes() == before_disk
    assert config == before_config
    assert not list(path.parent.glob('.*.tmp'))


def test_saving_while_hidden_does_not_open_viewer_or_touch_lifecycle(preview, monkeypatch):
    client, _path, _config, lifecycle = preview
    opened = Mock()
    monkeypatch.setattr(web_app.urllib.request, 'urlopen', opened)
    response = client.post('/api/preview/settings', json={'max_edge': 0, 'quality': 100})
    assert response.status_code == 200
    assert not web_app._overlay_enabled
    lifecycle.start.assert_not_called()
    lifecycle.stop.assert_not_called()
    lifecycle.cancel_scheduled_start.assert_not_called()
    opened.assert_not_called()


@pytest.mark.parametrize('document', ['[]\n', 'web: []\n', 'web: [unterminated'])
def test_invalid_latest_yaml_is_not_overwritten(preview, document):
    client, path, config, _lifecycle = preview
    path.write_text(document)
    previous = copy.deepcopy(config)
    assert client.post('/api/preview/settings', json={'quality': 80}).status_code == 500
    assert path.read_text() == document
    assert config == previous


def test_preview_save_rejects_cross_origin(preview):
    client, path, _config, _lifecycle = preview
    original = path.read_bytes()
    response = client.post('/api/preview/settings', json={'quality': 80},
                           headers={'Origin': 'https://other.example'})
    assert response.status_code == 403
    assert path.read_bytes() == original


def decode_part(part):
    image = part.split(b'\r\n\r\n', 1)[1][:-2]
    return cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_overlay_viewer_applies_settings_without_reopening(preview, monkeypatch):
    client, _path, _config, _lifecycle = preview
    frame = np.random.default_rng(7).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(web_app, '_latest_frame', frame)
    monkeypatch.setattr(web_app, '_latest_seq', 1)
    monkeypatch.setattr(web_app.time, 'monotonic', Mock(side_effect=[1.0, 2.0]))
    stream = web_app._generate_overlay_preview()
    try:
        assert decode_part(next(stream)).shape[:2] == (120, 160)  # Never upscale.
        assert client.post('/api/preview/settings', json={'max_edge': 80}).status_code == 200
        monkeypatch.setattr(web_app, '_latest_seq', 2)
        assert decode_part(next(stream)).shape[:2] == (60, 80)
    finally:
        stream.close()


def test_camera_only_viewer_applies_size_and_quality_on_same_upstream(preview, monkeypatch):
    client, _path, _config, _lifecycle = preview
    frame = np.random.default_rng(7).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', frame)
    assert ok
    jpeg = encoded.tobytes()
    part = b'--source\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n'
    upstream = io.BytesIO(part * 2 + b'--source--\r\n')
    upstream.headers = {'Content-Type': 'multipart/x-mixed-replace; boundary=source'}
    opened = Mock(return_value=upstream)
    monkeypatch.setattr(web_app.urllib.request, 'urlopen', opened)
    monkeypatch.setattr(web_app.time, 'monotonic', Mock(side_effect=[1.0, 2.0]))
    response = web_app._proxy_camera_stream()
    stream = iter(response.response)
    real_encode = cv2.imencode
    try:
        next(stream)  # Initial No Signal allows closure before opening a source.
        assert decode_part(next(stream)).shape[:2] == (120, 160)
        assert client.post('/api/preview/settings', json={'max_edge': 80, 'quality': 95}).status_code == 200
        with patch.object(web_app.cv2, 'imencode', wraps=real_encode) as encoder:
            assert decode_part(next(stream)).shape[:2] == (60, 80)
        assert encoder.call_args.args[2] == [cv2.IMWRITE_JPEG_QUALITY, 95]
        opened.assert_called_once()
    finally:
        response.close()
    assert upstream.closed


def test_raw_preview_fps_gate_drops_early_frames_before_decoding(preview, monkeypatch):
    wire = bytearray()
    for brightness in (20, 100, 200):
        frame = np.full((24, 32, 3), brightness, dtype=np.uint8)
        ok, encoded = cv2.imencode('.jpg', frame)
        assert ok
        jpeg = encoded.tobytes()
        wire.extend(b'--source\r\nContent-Length: ' + str(len(jpeg)).encode()
                    + b'\r\n\r\n' + jpeg + b'\r\n')
    upstream = io.BytesIO(wire + b'--source--\r\n')
    upstream.headers = {'Content-Type': 'multipart/x-mixed-replace; boundary=source'}
    opened = Mock(return_value=upstream)
    monkeypatch.setattr(web_app.urllib.request, 'urlopen', opened)
    monkeypatch.setattr(web_app.time, 'monotonic', Mock(side_effect=[1.0, 1.01, 1.1]))
    response = web_app._proxy_camera_stream()
    stream = iter(response.response)
    try:
        next(stream)
        with patch.object(web_app, '_decode_camera_jpeg', wraps=web_app._decode_camera_jpeg) as decoder:
            assert decode_part(next(stream)).mean() == pytest.approx(20, abs=1)
            assert decode_part(next(stream)).mean() == pytest.approx(200, abs=1)
        assert decoder.call_count == 2
        opened.assert_called_once()
    finally:
        response.close()
    assert upstream.closed


@pytest.mark.parametrize('width,height', [(65535, 65535), (16385, 1), (8192, 8192)])
def test_oversized_jpeg_is_rejected_before_opencv_allocation(width, height):
    # A tiny compressed input can advertise enormous decoded dimensions.
    sof = (b'\xff\xc0\x00\x0b\x08' + height.to_bytes(2, 'big')
           + width.to_bytes(2, 'big') + b'\x01\x01\x11\x00')
    image = b'\xff\xd8' + sof + b'\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xff\xd9'
    with patch.object(web_app.cv2, 'imdecode') as decoder:
        with pytest.raises(ValueError):
            web_app._decode_camera_jpeg(image)
    decoder.assert_not_called()


@pytest.mark.parametrize('progressive', [False, True])
def test_bounded_decoder_accepts_real_jpeg_with_marker_like_metadata(progressive):
    frame = np.full((32, 48, 3), 123, dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_PROGRESSIVE, int(progressive)])
    assert ok
    image = encoded.tobytes()
    payload = b'EXIF data containing fake SOF \xff\xc0\xff\xff and EOI \xff\xd9'
    app = b'\xff\xe1' + (len(payload) + 2).to_bytes(2, 'big') + payload
    decoded = web_app._decode_camera_jpeg(image[:2] + app + image[2:])
    np.testing.assert_array_equal(decoded, frame)


@pytest.mark.parametrize('image', [b'', b'\xff\xd8\xff\xd9', b'\xff\xd8\xff\xe1\xff\xff'])
def test_malformed_header_is_rejected_without_decoding(image):
    with patch.object(web_app.cv2, 'imdecode') as decoder:
        with pytest.raises(ValueError):
            web_app._decode_camera_jpeg(image)
    decoder.assert_not_called()


def test_run_web_uses_active_config_path(preview, monkeypatch):
    client, path, config, _lifecycle = preview
    other = path.parent / 'selected.yaml'
    lifecycle = Mock()
    monkeypatch.setattr(web_app, 'TrackerLifecycle', Mock(return_value=lifecycle))

    def serve(**_kwargs):
        assert client.post('/api/preview/settings', json={'quality': 82}).status_code == 200

    monkeypatch.setattr(web_app.app, 'run', serve)
    web_app.run_web(config, config_path=other)
    assert yaml.safe_load(other.read_text())['web']['preview_quality'] == 82
    assert yaml.safe_load(path.read_text())['web']['preview_quality'] == 70


def test_main_passes_selected_yaml_to_web(preview, monkeypatch):
    _client, path, config, _lifecycle = preview
    monkeypatch.setattr('sys.argv', ['web.py', '--config', str(path)])
    monkeypatch.setattr(web_app, 'load_config', Mock(return_value=config))
    monkeypatch.setattr(web_app, 'apply_cli_overrides', Mock())
    monkeypatch.setattr(web_app, 'configure_logging', Mock())
    monkeypatch.setattr(web_app, 'web_bind_host', Mock(return_value='127.0.0.1'))
    server = Mock()
    monkeypatch.setattr(web_app, 'run_web', server)
    web_app.main()
    server.assert_called_once_with(config, '127.0.0.1', 5000, config_path=str(path))

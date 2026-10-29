"""Raw camera viewing survives upstream startup delays and broken connections."""

from functools import lru_cache
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from pistream import web_app


@pytest.fixture(autouse=True)
def idle_tracker(monkeypatch):
    monkeypatch.setattr(web_app, '_lifecycle', None)
    monkeypatch.setattr(web_app, '_config', {})


class CameraResponse:
    def __init__(self, chunks, *, boundary='upstream', events=None, name='camera'):
        self.headers = {'Content-Type': f'multipart/x-mixed-replace; boundary={boundary}'}
        self.chunks = iter(chunks)
        self.buffer = bytearray()
        self.closed = False
        self.events = events
        self.name = name

    def read(self, size):
        if not self.buffer:
            chunk = next(self.chunks, b'')
            if isinstance(chunk, Exception):
                raise chunk
            self.buffer.extend(chunk)
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def readline(self, limit):
        line = bytearray()
        while len(line) < limit:
            byte = self.read(1)
            if not byte:
                break
            line.extend(byte)
            if byte == b'\n':
                break
        return bytes(line)

    def close(self):
        self.closed = True
        if self.events is not None:
            self.events.append(('close', self.name))


@lru_cache
def jpeg(content):
    pixels = np.full((16, 20, 3), sum(content) % 256, dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', pixels)
    assert ok
    return encoded.tobytes()


def multipart_part(image, boundary='upstream', *, content_length=True):
    headers = b'--' + boundary.encode() + b'\r\nContent-Type: image/jpeg\r\n'
    if content_length:
        headers += f'Content-Length: {len(image)}\r\n'.encode()
    return headers + b'\r\n' + image + b'\r\n'


def multipart(image, boundary='upstream', *, content_length=True):
    return (multipart_part(image, boundary, content_length=content_length)
            + b'--' + boundary.encode() + b'--\r\n')


def assert_part(part, image):
    assert part.startswith(b'--frame\r\n')
    assert b'Content-Type: image/jpeg\r\n' in part
    assert part.endswith(b'\r\n')
    assert same_image(part, image)


def same_image(part, image):
    """Preview re-encoding preserves visible pixels, not source JPEG metadata."""
    actual = part.split(b'\r\n\r\n', 1)[1][:-2]
    actual = cv2.imdecode(np.frombuffer(actual, dtype=np.uint8), cv2.IMREAD_COLOR)
    expected = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    return (actual is not None and expected is not None and actual.shape == expected.shape
            and np.allclose(actual, expected, atol=2))


def next_image(stream, image, placeholder):
    # The generator may send a placeholder both after failure and before its
    # next connection attempt so a closed viewer can be detected before open.
    for _ in range(8):
        part = next(stream)
        if same_image(part, image):
            assert_part(part, image)
            return
        assert_part(part, placeholder)
    pytest.fail('Camera preview did not resume within the retry sequence')


def test_camera_jpegs_handles_fragmented_headers_markers_and_multiple_frames():
    first = jpeg(b'first')
    second = jpeg(b'second')
    wire = multipart_part(first, 'other') + multipart(second, 'other')
    start = wire.index(b'\xff\xd8')
    upstream = CameraResponse([wire[:7], wire[7:start + 1], wire[start + 1:start + len(first) - 1],
                               wire[start + len(first) - 1:]], boundary='other')
    assert list(web_app._camera_jpegs(upstream)) == [first, second]


@pytest.mark.parametrize('content_length', [True, False])
def test_camera_jpegs_preserves_valid_app_marker_payload_byte_for_byte(content_length):
    original = jpeg(b'valid JPEG with application data')
    payload = b'Application data containing JPEG markers: \xff\xd9 and \xff\xd8'
    segment = b'\xff\xe1' + (len(payload) + 2).to_bytes(2, 'big') + payload
    with_app = original[:2] + segment + original[2:]
    decoded = cv2.imdecode(np.frombuffer(with_app, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None, 'The regression must exercise a valid JPEG'
    np.testing.assert_array_equal(decoded, cv2.imdecode(np.frombuffer(original, dtype=np.uint8),
                                                      cv2.IMREAD_COLOR))
    wire = multipart(with_app, boundary='CUSTOM', content_length=content_length)
    upstream = CameraResponse([wire[:21], wire[21:70], wire[70:]], boundary='CUSTOM')
    assert list(web_app._camera_jpegs(upstream)) == [with_app]


def test_camera_jpegs_content_length_allows_a_boundary_like_app_payload():
    original = jpeg(b'valid JPEG with a boundary-like application payload')
    payload = b'Application data\r\n--CUSTOM\r\ninside the JPEG'
    segment = b'\xff\xe1' + (len(payload) + 2).to_bytes(2, 'big') + payload
    with_app = original[:2] + segment + original[2:]
    wire = multipart(with_app, boundary='CUSTOM')
    assert list(web_app._camera_jpegs(CameraResponse([wire], boundary='CUSTOM'))) == [with_app]


def test_camera_jpegs_lengthless_parts_preserve_fragmented_boundaries():
    first = jpeg(b'first lengthless frame')
    second = jpeg(b'second lengthless frame')
    wire = (multipart_part(first, 'CUSTOM', content_length=False)
            + multipart(second, 'CUSTOM', content_length=False))
    chunks = [wire[index:index + 7] for index in range(0, len(wire), 7)]
    upstream = CameraResponse(chunks, boundary='CUSTOM')
    upstream.headers['Content-Type'] = 'multipart/x-mixed-replace; boundary="CUSTOM"'
    assert list(web_app._camera_jpegs(upstream)) == [first, second]


def test_camera_jpegs_discards_a_frame_cut_short_by_eof():
    complete = jpeg(b'complete')
    incomplete = jpeg(b'incomplete')
    partial_part = multipart_part(incomplete)[:-12]
    upstream = CameraResponse([multipart_part(complete) + partial_part])
    assert list(web_app._camera_jpegs(upstream)) == [complete]


@pytest.mark.parametrize('content_length', [True, False])
@pytest.mark.parametrize('complete', [True, False])
def test_camera_jpegs_bounds_incomplete_and_complete_frames(content_length, complete):
    image = jpeg(b'oversized')
    wire = multipart(image, content_length=content_length)
    if not complete:
        wire = multipart_part(image, content_length=content_length)[:-12]
    chunks = [wire[:23], wire[23:]]
    with pytest.raises(ValueError):
        list(web_app._camera_jpegs(CameraResponse(chunks), max_frame_bytes=16))


def test_raw_preview_recovers_from_initial_offline_and_later_eof(monkeypatch):
    placeholder = jpeg(b'no signal')
    first = jpeg(b'first connection')
    recovered = jpeg(b'recovered connection')
    events = []
    eof = CameraResponse([multipart(first, 'FIRST')], boundary='FIRST', events=events, name='first')
    healthy = CameraResponse([multipart(recovered, 'SECOND')], boundary='SECOND', events=events,
                             name='second')
    sources = iter([OSError('camera is offline'), eof, healthy])

    def open_camera(*_args, **_kwargs):
        events.append(('open', None))
        source = next(sources)
        if isinstance(source, Exception):
            raise source
        return source

    placeholder_factory = Mock(return_value=placeholder)
    delays = []
    monkeypatch.setattr(web_app.urllib.request, 'urlopen', open_camera)
    monkeypatch.setattr(web_app, 'camera_stream_url', lambda _config: 'http://camera:8000/stream')
    monkeypatch.setattr(web_app, '_placeholder_jpeg', placeholder_factory)
    monkeypatch.setattr(web_app.time, 'sleep', delays.append)
    factory = Mock()
    monkeypatch.setattr(web_app, 'HumanTracker', factory)

    response = web_app._proxy_camera_stream()
    assert response is not None
    assert response.content_type == 'multipart/x-mixed-replace; boundary=frame'
    stream = iter(response.response)
    try:
        assert_part(next(stream), placeholder)
        assert events == [], 'The first placeholder must permit closure before opening a camera'
        next_image(stream, first, placeholder)
        assert_part(next(stream), placeholder)
        assert eof.closed
        next_image(stream, recovered, placeholder)
        opens = [index for index, event in enumerate(events) if event[0] == 'open']
        assert events.index(('close', 'first')) < opens[2]
        assert len(opens) == 3
        assert delays and all(0 < delay <= 3 for delay in delays)
        placeholder_factory.assert_called_once_with()
        factory.assert_not_called()
    finally:
        response.close()
    assert healthy.closed
    assert len([event for event in events if event[0] == 'open']) == 3


def test_raw_preview_discards_partial_frame_before_reconnecting(monkeypatch):
    placeholder = jpeg(b'no signal')
    recovered = jpeg(b'recovered')
    broken = CameraResponse([multipart_part(jpeg(b'incomplete'), 'OLD')[:-12]], boundary='OLD')
    healthy = CameraResponse([multipart(recovered, 'NEW')], boundary='NEW')
    opened = Mock(side_effect=[broken, healthy])
    monkeypatch.setattr(web_app.urllib.request, 'urlopen', opened)
    monkeypatch.setattr(web_app, 'camera_stream_url', lambda _config: 'http://camera:8000/stream')
    monkeypatch.setattr(web_app, '_placeholder_jpeg', lambda: placeholder)
    monkeypatch.setattr(web_app.time, 'sleep', lambda _delay: None)
    response = web_app._proxy_camera_stream()
    stream = iter(response.response)
    try:
        assert_part(next(stream), placeholder)
        next_image(stream, recovered, placeholder)
        assert broken.closed
    finally:
        response.close()
    assert healthy.closed
    assert opened.call_count == 2


def test_raw_preview_closes_a_reset_connection_before_retry(monkeypatch):
    first = jpeg(b'first')
    recovered = jpeg(b'recovered')
    placeholder = jpeg(b'no signal')
    reset = CameraResponse([multipart_part(first), ConnectionResetError('camera rebooted')])
    healthy = CameraResponse([multipart(recovered)])

    def open_camera(*_args, **_kwargs):
        if opened.call_count == 1:
            return reset
        assert reset.closed, 'An old upstream must close before opening its replacement'
        return healthy

    opened = Mock(side_effect=open_camera)
    monkeypatch.setattr(web_app.urllib.request, 'urlopen', opened)
    monkeypatch.setattr(web_app, 'camera_stream_url', lambda _config: 'http://camera:8000/stream')
    monkeypatch.setattr(web_app, '_placeholder_jpeg', lambda: placeholder)
    monkeypatch.setattr(web_app.time, 'sleep', lambda _delay: None)
    response = web_app._proxy_camera_stream()
    stream = iter(response.response)
    try:
        next_image(stream, first, placeholder)
        assert_part(next(stream), placeholder)
        next_image(stream, recovered, placeholder)
    finally:
        response.close()
    assert reset.closed and healthy.closed
    assert opened.call_count == 2


def test_repeated_offline_retries_use_a_bounded_delay(monkeypatch):
    opened = Mock(side_effect=OSError('camera offline'))
    delays = []
    monkeypatch.setattr(web_app.urllib.request, 'urlopen', opened)
    monkeypatch.setattr(web_app, 'camera_stream_url', lambda _config: 'http://camera:8000/stream')
    monkeypatch.setattr(web_app, '_placeholder_jpeg', lambda: jpeg(b'no signal'))
    monkeypatch.setattr(web_app.time, 'sleep', delays.append)
    response = web_app._proxy_camera_stream()
    stream = iter(response.response)
    try:
        for _ in range(20):
            assert_part(next(stream), jpeg(b'no signal'))
            if opened.call_count == 6:
                break
    finally:
        response.close()
    assert opened.call_count == 6
    assert len(delays) == 5
    assert all(0 < delay <= 3 for delay in delays)


def test_closing_preview_during_retry_prevents_another_camera_open(monkeypatch):
    opened = Mock(side_effect=OSError('camera offline'))
    sleep = Mock()
    monkeypatch.setattr(web_app.urllib.request, 'urlopen', opened)
    monkeypatch.setattr(web_app, 'camera_stream_url', lambda _config: 'http://camera:8000/stream')
    monkeypatch.setattr(web_app, '_placeholder_jpeg', lambda: jpeg(b'no signal'))
    monkeypatch.setattr(web_app.time, 'sleep', sleep)
    response = web_app._proxy_camera_stream()
    stream = iter(response.response)
    next(stream)
    assert opened.call_count == 0
    next(stream)
    response.close()
    with pytest.raises(StopIteration):
        next(stream)
    assert opened.call_count == 1
    sleep.assert_not_called()


def test_closing_preview_closes_its_current_upstream(monkeypatch):
    upstream = CameraResponse([multipart(jpeg(b'live'))])
    opened = Mock(return_value=upstream)
    monkeypatch.setattr(web_app.urllib.request, 'urlopen', opened)
    monkeypatch.setattr(web_app, 'camera_stream_url', lambda _config: 'http://camera:8000/stream')
    response = web_app._proxy_camera_stream()
    stream = iter(response.response)
    next(stream)  # The initial placeholder does not open the upstream.
    next(stream)  # A live frame owns the upstream until this response closes.
    response.close()
    assert upstream.closed
    assert opened.call_count == 1


@pytest.mark.parametrize('consume_placeholder', [False, True])
def test_closing_running_preview_releases_its_eager_upstream(monkeypatch, consume_placeholder):
    tracker = SimpleNamespace(running=True, capture=object())
    monkeypatch.setattr(web_app, '_lifecycle', SimpleNamespace(tracker=tracker))
    upstream = CameraResponse([multipart(jpeg(b'live'))])
    opened = Mock(return_value=upstream)
    monkeypatch.setattr(web_app.urllib.request, 'urlopen', opened)
    monkeypatch.setattr(web_app, 'camera_stream_url', lambda _config: 'http://camera:8000/stream')
    response = web_app._proxy_camera_stream()
    assert opened.call_count == 1
    assert not upstream.closed
    if consume_placeholder:
        next(iter(response.response))
    response.close()
    assert upstream.closed
    assert opened.call_count == 1


def test_running_raw_preview_falls_back_to_processed_frames_if_camera_open_fails(monkeypatch):
    tracker = SimpleNamespace(running=True, capture=object())
    monkeypatch.setattr(web_app, '_lifecycle', SimpleNamespace(tracker=tracker))
    monkeypatch.setattr(web_app, '_overlay_enabled', False)
    opened = Mock(side_effect=OSError('camera offline'))
    monkeypatch.setattr(web_app.urllib.request, 'urlopen', opened)
    monkeypatch.setattr(web_app, 'camera_stream_url', lambda _config: 'http://camera:8000/stream')
    monkeypatch.setattr(web_app, '_generate_overlay_preview', lambda: iter([b'processed frame']))
    response = web_app.app.test_client().get('/video_feed', buffered=False)
    try:
        assert next(iter(response.response)) == b'processed frame'
    finally:
        response.close()
    opened.assert_called_once()
    assert tracker.running
    assert web_app._overlay_enabled is False

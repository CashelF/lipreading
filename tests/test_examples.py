"""Explicit sample recording validates input and never saves partial sessions."""
import asyncio
import importlib
from io import BytesIO
import json
import stat
import struct

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from starlette.websockets import WebSocketDisconnect

from backend import examples

ORIGIN = {'origin': 'http://localhost:5173'}


def jpeg(size=(128, 128)):
    output = BytesIO()
    Image.new('RGB', size, (64, 64, 64)).save(output, 'JPEG')
    return output.getvalue()


def packet(timestamp, payload=None):
    return struct.pack('<d', float(timestamp)) + (jpeg() if payload is None else payload)


@pytest.fixture
def client(tmp_path, monkeypatch):
    local_store = examples.ExampleStore(tmp_path / 'samples')
    monkeypatch.setattr(examples, 'store', local_store)
    lock = asyncio.Lock()
    app = FastAPI()
    app.include_router(examples.create_examples_router(lambda origin, host: origin == ORIGIN['origin'], lambda: lock))
    with TestClient(app) as connection:
        yield connection, local_store, lock


def start(ws, label='quiet_still', take=1):
    ws.send_json({'type': 'start', 'label': label, 'take': take})
    assert ws.receive_json() == {'type': 'ready'}


def send_frames(ws, count=201, interval=40):
    image = jpeg()
    for index in range(count):
        ws.send_bytes(packet(index * interval, image))


def complete_buffer(label='speech_subtle', take=2):
    buffer = examples.ExampleBuffer(label, take)
    image = jpeg()
    for index in range(201):
        buffer.append(packet(1000 + index * 40, image))
    return buffer


@pytest.mark.parametrize('size', [(640, 360), (1280, 720), (1280, 960), (540, 960)])
def test_examples_preserve_the_same_high_resolution_jpegs_as_live_reading(size):
    buffer = examples.ExampleBuffer('speech_normal', 1)
    payload = jpeg(size)
    buffer.append(packet(0, payload))
    assert buffer.dimensions == size
    assert buffer.frames[0].jpeg == payload


def test_save_exact_camera_frames_manifest_and_private_permissions(client):
    connection, store, lock = client
    with connection.websocket_connect('/api/examples', headers=ORIGIN) as ws:
        start(ws, 'speech_subtle', 2)
        send_frames(ws)
        ws.send_json({'type': 'stop'})
        saved = ws.receive_json()
        assert saved['type'] == 'saved'
        assert saved['label'] == 'speech_subtle'
        assert saved['take'] == 2
        assert saved['frames'] == 201
        assert saved['duration_seconds'] == 8
    assert not lock.locked()
    path = store.root / saved['id']
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['timestamps_ms'] == [index * 40.0 for index in range(201)]
    assert manifest['frame_files'] == [f'{index:06d}.jpg' for index in range(201)]
    assert manifest['capture_fps'] == 25
    assert (manifest['width'], manifest['height']) == (128, 128)
    assert (path / '000001.jpg').read_bytes() == jpeg()
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert stat.S_IMODE((path / '000001.jpg').stat().st_mode) == 0o600
    assert stat.S_IMODE((path / 'manifest.json').stat().st_mode) == 0o600
    assert connection.get('/api/examples').json() == [{key: saved[key] for key in
                                                     ('id', 'label', 'take', 'frames', 'duration_seconds')}]
    assert not list(store.root.glob('.pending-*'))


@pytest.mark.parametrize('headers', [{}, {'origin': 'https://untrusted.example'}])
def test_reject_untrusted_origin(client, headers):
    connection, store, lock = client
    with pytest.raises(WebSocketDisconnect) as exc:
        with connection.websocket_connect('/api/examples', headers=headers):
            pass
    assert exc.value.code == 1008
    assert not lock.locked()
    assert store.list() == []


@pytest.mark.parametrize('finish', ['cancel', 'disconnect'])
def test_cancel_or_disconnect_does_not_save(client, finish):
    connection, store, lock = client
    with connection.websocket_connect('/api/examples', headers=ORIGIN) as ws:
        start(ws)
        send_frames(ws, 30)
        if finish == 'cancel':
            ws.send_json({'type': 'cancel'})
            assert ws.receive_json() == {'type': 'canceled'}
    assert not lock.locked()
    assert not store.root.exists()


@pytest.mark.parametrize('command', [
    [], {'type': 'unknown'}, {'type': 'stop'},
    {'type': 'start', 'label': '../../secret', 'take': 1},
    {'type': 'start', 'label': 'quiet_still', 'take': True},
    {'type': 'start', 'label': 'quiet_still', 'take': 3},
    {'type': 'start', 'label': ['quiet_still'], 'take': 1},
    {'type': 'start', 'label': 'quiet_still', 'take': 1, 'padding': 'x' * 1024},
])
def test_invalid_control_discards_sample(client, command):
    connection, store, lock = client
    with connection.websocket_connect('/api/examples', headers=ORIGIN) as ws:
        ws.send_json(command)
        assert ws.receive_json()['type'] == 'error'
    assert not lock.locked()
    assert store.list() == []


def test_second_session_cannot_capture_while_lock_held(client):
    connection, store, lock = client
    with connection.websocket_connect('/api/examples', headers=ORIGIN) as first:
        start(first)
        with connection.websocket_connect('/api/examples', headers=ORIGIN) as second:
            assert second.receive_json()['type'] == 'error'
        assert lock.locked()
        first.send_json({'type': 'cancel'})
        assert first.receive_json()['type'] == 'canceled'
    assert not lock.locked()


def test_frames_before_start_are_rejected(client):
    connection, store, lock = client
    with connection.websocket_connect('/api/examples', headers=ORIGIN) as ws:
        ws.send_bytes(packet(0))
        assert ws.receive_json()['type'] == 'error'
    assert not lock.locked()


@pytest.mark.parametrize('payload', [b'\xff\xd8broken', jpeg((1281, 960)), jpeg((1280, 961)), jpeg((95, 100))])
def test_invalid_jpeg_or_dimensions_rejected(client, payload):
    connection, store, lock = client
    with connection.websocket_connect('/api/examples', headers=ORIGIN) as ws:
        start(ws)
        ws.send_bytes(packet(0, payload))
        assert ws.receive_json()['type'] == 'error'
    assert not lock.locked()
    assert store.list() == []


def test_camera_gap_rejected_and_no_partial_files(client):
    connection, store, lock = client
    with connection.websocket_connect('/api/examples', headers=ORIGIN) as ws:
        start(ws)
        ws.send_bytes(packet(0))
        ws.send_bytes(packet(251))
        assert 'interrupted' in ws.receive_json()['detail']
    assert not store.root.exists()


def test_short_recording_cannot_be_saved(client):
    connection, store, lock = client
    with connection.websocket_connect('/api/examples', headers=ORIGIN) as ws:
        start(ws)
        send_frames(ws, 100)
        ws.send_json({'type': 'stop'})
        assert ws.receive_json()['type'] == 'error'
    assert not store.root.exists()


def test_low_frame_rate_cannot_be_saved(client):
    connection, store, lock = client
    with connection.websocket_connect('/api/examples', headers=ORIGIN) as ws:
        start(ws)
        send_frames(ws, 81, 100)
        ws.send_json({'type': 'stop'})
        assert 'slowly' in ws.receive_json()['detail']
    assert not store.root.exists()


@pytest.mark.parametrize('limit', ['MAX_BYTES', 'MAX_FRAMES'])
def test_frame_and_memory_limits_are_bounded(monkeypatch, limit):
    monkeypatch.setattr(examples, limit, 1)
    buffer = examples.ExampleBuffer('quiet_motion', 1)
    with pytest.raises(ValueError, match='too large'):
        buffer.append(packet(0))
        buffer.append(packet(40))
    assert len(buffer.frames) <= 1


def test_duplicate_timestamps_and_changing_resolution_rejected():
    buffer = examples.ExampleBuffer('quiet_motion', 1)
    buffer.append(packet(0))
    with pytest.raises(ValueError, match='increase'):
        buffer.append(packet(0))
    with pytest.raises(ValueError, match='resolution changed'):
        buffer.append(packet(40, jpeg((160, 128))))


def test_delete_requires_origin_and_only_removes_requested_sample(client):
    connection, store, lock = client
    first = store.save(complete_buffer())
    second = store.save(complete_buffer())
    url = f"/api/examples/{first['id']}"
    assert connection.delete(url).status_code == 403
    assert connection.delete(url, headers={'origin': 'https://untrusted.example'}).status_code == 403
    assert connection.delete('/api/examples/not-a-sample', headers=ORIGIN).status_code == 400
    assert connection.delete(url, headers=ORIGIN).json() == {'deleted': first['id']}
    assert store.list() == [second]
    assert connection.delete(url, headers=ORIGIN).status_code == 404


def test_list_rejects_untrusted_browser_origin(client):
    connection, _, _ = client
    assert connection.get('/api/examples', headers={'origin': 'https://untrusted.example'}).status_code == 403


def test_store_does_not_follow_symlink_sample_directory(tmp_path):
    store = examples.ExampleStore(tmp_path / 'samples')
    store.root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep.txt').write_text('keep')
    sample_id = 'a' * 32
    (store.root / sample_id).symlink_to(outside, target_is_directory=True)
    assert store.list() == []
    with pytest.raises(ValueError, match='path'):
        store.delete(sample_id)
    assert (outside / 'keep.txt').read_text() == 'keep'


def test_failed_save_removes_staging_files(tmp_path, monkeypatch):
    store = examples.ExampleStore(tmp_path / 'samples')
    def fail_write(*args):
        raise OSError('disk full')
    monkeypatch.setattr(store, '_write_private', fail_write)
    with pytest.raises(OSError):
        store.save(complete_buffer())
    assert list(store.root.iterdir()) == []


def test_total_sample_limit(tmp_path, monkeypatch):
    store = examples.ExampleStore(tmp_path / 'samples')
    monkeypatch.setattr(examples, 'MAX_SAMPLES', 1)
    store.save(complete_buffer())
    with pytest.raises(ValueError, match='collection is full'):
        store.save(complete_buffer())
    assert len(store.list()) == 1


def test_idle_session_times_out_and_releases_lock(client, monkeypatch):
    connection, store, lock = client
    monkeypatch.setattr(examples, 'SESSION_TIMEOUT_SECONDS', .02)
    with connection.websocket_connect('/api/examples', headers=ORIGIN) as ws:
        start(ws)
        assert 'timed out' in ws.receive_json()['detail']
    assert not lock.locked()
    assert not store.root.exists()


@pytest.mark.parametrize('recording_first', [True, False])
def test_capture_and_reading_share_app_lock(tmp_path, monkeypatch, recording_first):
    server = importlib.import_module('backend.app')
    class ModelStub:
        def load(self):
            pass
        def status(self):
            # Recording remains available even if model loading failed.
            return {'ready': not recording_first, 'detail': 'test'}
    monkeypatch.setattr(server, 'model', ModelStub())
    monkeypatch.setattr(server, 'session_lock', asyncio.Lock())
    monkeypatch.setattr(examples, 'store', examples.ExampleStore(tmp_path / 'samples'))
    first_path = '/api/examples' if recording_first else '/api/stream'
    second_path = '/api/stream' if recording_first else '/api/examples'
    with TestClient(server.app) as connection:
        with connection.websocket_connect(first_path, headers=ORIGIN) as first:
            if recording_first:
                start(first)
            else:
                first.send_json({'type': 'start'})
                assert first.receive_json()['state'] == 'capturing'
            assert connection.get('/api/status').json()['busy'] is True
            with connection.websocket_connect(second_path, headers=ORIGIN) as second:
                assert second.receive_json()['type'] == 'error'
            first.send_json({'type': 'cancel' if recording_first else 'stop'})
            assert first.receive_json()['type'] == ('canceled' if recording_first else 'stopped')
        assert not server.session_lock.locked()

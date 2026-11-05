"""Transport tests use an explicit in-process model double, never a production demo mode."""
import asyncio
import importlib
from io import BytesIO
import struct
import threading

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from starlette.websockets import WebSocketDisconnect


class StubModel:
    def __init__(self):
        self.ready = True
        self.calls = []
        self.error = None

    def load(self):
        pass

    def status(self):
        return {'ready': self.ready, 'model': 'test model', 'detail': 'Weights unavailable' if not self.ready else 'Ready'}

    def transcribe(self, frames, fps, *, cache=None, speech_gate=True):
        self.calls.append((len(frames), fps))
        if self.error:
            raise self.error
        return {'text': 'test output'}


# Model imports are lazy; tests patch the instance without loading checkpoints.
server = importlib.import_module('backend.app')

ORIGIN = {'origin': 'http://localhost:5173'}


@pytest.fixture
def client(monkeypatch):
    stub = StubModel()
    monkeypatch.setattr(server, 'model', stub)
    monkeypatch.setattr(server, 'session_lock', asyncio.Lock())
    monkeypatch.setattr(server, 'loading', False)
    with TestClient(server.app) as test_client:
        yield test_client, stub


def frames(ws, count=12):
    out = BytesIO()
    Image.new('RGB', (128, 128), (64, 64, 64)).save(out, format='JPEG')
    for i in range(count):
        ws.send_bytes(struct.pack('<d', i * 40.0) + out.getvalue())


def drain_until_stopped(ws):
    received = []
    for _ in range(20):
        item = ws.receive_json()
        received.append(item)
        if item['type'] == 'stopped':
            return received
    pytest.fail('Stream did not acknowledge stop within 20 messages')


def test_status_exposes_actual_model_readiness(client):
    test_client, stub = client
    stub.ready = False
    result = test_client.get('/api/status').json()
    assert result['ready'] is False
    assert result['detail'] == 'Weights unavailable'
    assert result['fps'] == 25
    assert result['segment_seconds'] == 6
    assert result['partial_seconds'] == 1
    assert result['initial_seconds'] == 2
    assert result['continuous_revisions'] is True


def test_timing_probe_round_trip_does_not_start_inference(client):
    test_client, stub = client
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start'})
        assert ws.receive_json()['state'] == 'capturing'
        ws.send_json({'type': 'ping', 'client_time_ms': 1234.5})
        assert ws.receive_json() == {'type': 'pong', 'client_time_ms': 1234.5}
        ws.send_json({'type': 'stop'})
        drain_until_stopped(ws)
    assert not stub.calls


@pytest.mark.parametrize('headers', [{}, {'origin': 'https://malicious.example'}, {'origin': 'http://localhost:5173.malicious.example'}])
def test_rejects_untrusted_websocket_origins(client, headers):
    test_client, _ = client
    with pytest.raises(WebSocketDisconnect) as exc:
        with test_client.websocket_connect('/api/stream', headers=headers):
            pass
    assert exc.value.code == 1008
    assert not server.session_lock.locked()


def test_stop_flushes_final_clip_and_releases_session(client):
    test_client, stub = client
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start'})
        assert ws.receive_json()['state'] == 'capturing'
        frames(ws)
        ws.send_json({'type': 'stop'})
        messages = drain_until_stopped(ws)
        transcript = [m for m in messages if m['type'] == 'transcript']
        assert len(transcript) == 1
        assert transcript[0]['final'] is True
        assert transcript[0]['segment_id'] == 0
        assert transcript[0]['text'] == 'test output'
    assert stub.calls == [(12, 25)]
    assert not server.session_lock.locked()


def test_empty_session_stops_without_fabricated_transcript(client):
    test_client, stub = client
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start'})
        ws.receive_json()
        ws.send_json({'type': 'stop'})
        messages = drain_until_stopped(ws)
    assert [m for m in messages if m['type'] == 'transcript'] == []
    assert stub.calls == []


def test_model_unavailable_rejects_start_and_releases_session(client):
    test_client, stub = client
    stub.ready = False
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start'})
        error = ws.receive_json()
        assert error['type'] == 'error'
        assert error['detail'] == 'Weights unavailable'
    assert not server.session_lock.locked()


def test_second_client_cannot_overlap_model_inference(client):
    test_client, _ = client
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as first:
        first.send_json({'type': 'start'})
        first.receive_json()
        with test_client.websocket_connect('/api/stream', headers=ORIGIN) as second:
            assert second.receive_json() == {'type': 'error', 'detail': 'Another reading session is active.'}
        assert server.session_lock.locked()
        first.send_json({'type': 'stop'})
        drain_until_stopped(first)
    assert not server.session_lock.locked()


@pytest.mark.parametrize('command', [[], {'type': 'unknown'}, {'type': 'start', 'padding': 'x' * 2000}])
def test_invalid_controls_release_session(client, command):
    test_client, _ = client
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json(command)
        assert ws.receive_json()['type'] == 'error'
    assert not server.session_lock.locked()


def test_binary_frame_before_start_is_rejected(client):
    test_client, stub = client
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        frames(ws, 1)
        assert ws.receive_json()['type'] == 'error'
    assert not server.session_lock.locked()
    assert not stub.calls


def test_bad_frames_clear_final_instead_of_fabricating_words(client):
    test_client, stub = client
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start'})
        ws.receive_json()
        for i in range(12):
            ws.send_bytes(struct.pack('<d', i * 40.0) + b'\xff\xd8corrupt')
        ws.send_json({'type': 'stop'})
        messages = drain_until_stopped(ws)
        assert any(m['type'] == 'quality' and 'decode' in m['detail'] for m in messages)
        assert any(m['type'] == 'transcript' and m['text'] == '' and m['final'] for m in messages)
    assert stub.calls == []


def test_inference_exception_reports_error_and_releases_session(client):
    test_client, stub = client
    stub.error = RuntimeError('GPU failure')
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start'})
        ws.receive_json()
        frames(ws)
        ws.send_json({'type': 'stop'})
        messages = drain_until_stopped(ws)
        assert any(m['type'] == 'error' for m in messages)
        assert all(m['text'] == '' and m['final'] for m in messages if m['type'] == 'transcript')
    assert not server.session_lock.locked()


def test_reload_rejects_cross_site_requests(client):
    test_client, stub = client
    stub.ready = False
    result = test_client.post('/api/model/load', headers={'origin': 'https://malicious.example'})
    assert result.status_code == 403


def test_reload_does_not_reload_ready_model(client):
    test_client, stub = client
    def forbidden_load():
        pytest.fail('Ready model should not be reloaded')
    stub.load = forbidden_load
    assert test_client.post('/api/model/load', headers=ORIGIN).json() == {'loading': False, 'ready': True}


def test_start_rejected_while_loading(client, monkeypatch):
    test_client, stub = client
    # Wait for the fixture's no-op startup load, then emulate a reload.
    import time
    deadline = time.monotonic() + 2
    while server.loading and time.monotonic() < deadline:
        time.sleep(0.005)
    monkeypatch.setattr(server, 'loading', True)
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start'})
        assert ws.receive_json()['type'] == 'error'
    assert not server.session_lock.locked()


def test_disconnect_holds_gpu_ownership_until_running_inference_finishes(client):
    test_client, stub = client
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    def slow_transcribe(frame_batch, fps, *, cache=None, speech_gate=True):
        entered.set()
        try:
            assert release.wait(3), 'Test failed to release inference'
            return {'text': 'test output'}
        finally:
            finished.set()
    stub.transcribe = slow_transcribe
    try:
        with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
            ws.send_json({'type': 'start'})
            ws.receive_json()
            frames(ws, 51)  # First two-second partial starts the worker.
            assert entered.wait(2)
            ws.close()
            with test_client.websocket_connect('/api/stream', headers=ORIGIN) as second:
                assert second.receive_json()['type'] == 'error'
            assert server.session_lock.locked()
            release.set()
            assert finished.wait(2)
            import time
            deadline = time.monotonic() + 2
            while server.session_lock.locked() and time.monotonic() < deadline:
                time.sleep(0.005)
            assert not server.session_lock.locked()
    finally:
        release.set()


def test_cancellation_after_stop_releases_lock_only_after_cuda_thread_finishes(monkeypatch):
    class Socket:
        headers = {**ORIGIN, 'host': 'localhost:8000'}

        def __init__(self):
            self.queue = asyncio.Queue()
            self.messages = []

        async def accept(self):
            pass

        async def receive(self):
            return await self.queue.get()

        async def send_json(self, message):
            self.messages.append(message)

        async def close(self, **kwargs):
            pass

    entered = threading.Event()
    release = threading.Event()
    stub = StubModel()
    def slow_transcribe(frames, fps, *, cache=None, speech_gate=True):
        entered.set()
        assert release.wait(3)
        return {'text': 'test output'}
    stub.transcribe = slow_transcribe
    monkeypatch.setattr(server, 'model', stub)
    monkeypatch.setattr(server, 'loading', False)

    async def scenario():
        monkeypatch.setattr(server, 'session_lock', asyncio.Lock())
        ws = Socket()
        out = BytesIO()
        Image.new('RGB', (128, 128)).save(out, format='JPEG')
        ws.queue.put_nowait({'type': 'websocket.receive', 'text': '{"type":"start"}'})
        for i in range(12):
            ws.queue.put_nowait({'type': 'websocket.receive', 'bytes': struct.pack('<d', i * 40.0) + out.getvalue()})
        ws.queue.put_nowait({'type': 'websocket.receive', 'text': '{"type":"stop"}'})
        task = asyncio.create_task(server.stream(ws))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert server.session_lock.locked(), 'CUDA work is still running; exclusive ownership must remain'
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert not server.session_lock.locked(), 'Cancelled stream leaked the exclusive session lock'

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_slow_inference_stops_stream_instead_of_accumulating_unbounded_video(client):
    test_client, stub = client
    entered = threading.Event()
    release = threading.Event()
    def slow_transcribe(frame_batch, fps, *, cache=None, speech_gate=True):
        entered.set()
        assert release.wait(5)
        return {'text': 'test output'}
    stub.transcribe = slow_transcribe
    out = BytesIO()
    Image.new('RGB', (128, 128)).save(out, format='JPEG')
    image = out.getvalue()
    try:
        with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
            ws.send_json({'type': 'start'})
            ws.receive_json()
            frames(ws, 51)
            assert entered.wait(2)
            # New candidates replace stale pending work, but after eight seconds
            # their six-second video no longer overlaps the in-flight 0–2s window.
            for i in range(51, 202):
                ws.send_bytes(struct.pack('<d', i * 40.0) + image)
            messages = []
            for _ in range(5):
                message = ws.receive_json()
                messages.append(message)
                if message['type'] == 'error':
                    break
            assert any(m['type'] == 'error' and 'preserve overlapping video' in m['detail'] for m in messages)
            assert server.session_lock.locked()
            release.set()
            import time
            deadline = time.monotonic() + 2
            while server.session_lock.locked() and time.monotonic() < deadline:
                time.sleep(0.005)
            assert not server.session_lock.locked()
    finally:
        release.set()

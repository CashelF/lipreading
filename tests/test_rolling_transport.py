"""Controlled backpressure tests through the real continuous WebSocket handler."""
import asyncio
from io import BytesIO
import struct
import threading

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from backend import app as server
from backend.stream import ClipBuffer

ORIGIN = {'origin': 'http://localhost:5173'}


class BlockingModel:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.calls = []
        self.active = 0
        self.maximum_active = 0
        self.guard = threading.Lock()
        self.answers = ['alpha beta', 'alpha beta gamma']

    def load(self):
        pass

    def status(self):
        return {'ready': True, 'model': 'explicit blocking test model', 'detail': 'Ready'}

    def transcribe(self, frames, fps, *, cache=None, speech_gate=True):
        with self.guard:
            index = len(self.calls)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.calls.append(tuple(frames))
        for frame in frames:
            cache.entry(frame)
        try:
            if index == 0:
                self.entered.set()
                assert self.release.wait(5), 'Test failed to release its model double'
            return {'text': self.answers[min(index, len(self.answers) - 1)],
                    'timings_ms': {'inference': 1., 'preprocessing': 1.}}
        finally:
            with self.guard:
                self.active -= 1
            self.finished.set()


@pytest.fixture
def session(monkeypatch):
    model = BlockingModel()
    buffers = []

    class ObservedBuffer(ClipBuffer):
        def __init__(self):
            super().__init__()
            self.snapshots = []
            self.changed = threading.Condition()
            buffers.append(self)

        def _snapshot(self, final):
            clip = super()._snapshot(final)
            with self.changed:
                self.snapshots.append(clip)
                self.changed.notify_all()
            return clip

        def wait_until(self, end, *, final=False):
            with self.changed:
                return self.changed.wait_for(
                    lambda: any(c.end_ms >= end and (not final or c.final) for c in self.snapshots),
                    timeout=2)

    monkeypatch.setattr(server, 'model', model)
    monkeypatch.setattr(server, 'loading', False)
    monkeypatch.setattr(server, 'session_lock', asyncio.Lock())
    monkeypatch.setattr(server, 'ClipBuffer', ObservedBuffer)
    try:
        with TestClient(server.app) as client:
            yield client, model, buffers
    finally:
        model.release.set()


def send_frames(ws, begin, end):
    data = BytesIO()
    Image.new('RGB', (128, 128), (80, 100, 120)).save(data, format='JPEG')
    for index in range(begin, end):
        ws.send_bytes(struct.pack('<d', index * 40.) + data.getvalue())


def start(ws):
    ws.send_json({'type': 'start'})
    assert ws.receive_json()['state'] == 'capturing'


def receive_until(ws, *, stopped=False, error=False):
    messages = []
    for _ in range(30):
        message = ws.receive_json()
        messages.append(message)
        if error and message['type'] == 'error':
            return messages
        if stopped and message['type'] == 'stopped':
            return messages
        if not stopped and not error and message.get('state') == 'capturing':
            return messages
    pytest.fail('The controlled stream did not finish its processing cycle')


def test_busy_worker_decodes_only_newest_pending_window_and_stop_reuses_it(session):
    client, model, buffers = session
    try:
        with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
            start(ws)
            send_frames(ws, 0, 51)
            assert model.entered.wait(2)
            send_frames(ws, 51, 126)
            assert buffers[0].wait_until(5000)
            assert len(model.calls) == 1
            model.release.set()
            messages = receive_until(ws)
            captions = [m for m in messages if m['type'] == 'transcript']
            assert [(m['window_start_ms'], m['window_end_ms']) for m in captions] == [(0, 2000), (0, 5000)]
            assert captions[-1]['text'] == 'alpha beta gamma'
            ws.send_json({'type': 'stop'})
            final_messages = receive_until(ws, stopped=True)
            final = next(m for m in final_messages if m['type'] == 'transcript')
            assert final['final'] is True
            assert final['timings_ms']['inference'] == 0
            assert final['text'] == 'alpha beta gamma'
        assert [len(frames) for frames in model.calls] == [51, 126]
        assert model.maximum_active == 1
        assert all(a is b for a, b in zip(model.calls[0], model.calls[1]))
        assert not server.session_lock.locked()
    finally:
        model.release.set()


def test_coalescing_across_moving_boundary_keeps_overlap_and_earlier_words(session):
    client, model, buffers = session
    model.answers = ['alpha beta', 'beta gamma']
    try:
        with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
            start(ws)
            send_frames(ws, 0, 51)
            assert model.entered.wait(2)
            send_frames(ws, 51, 176)
            assert buffers[0].wait_until(7000)
            model.release.set()
            messages = receive_until(ws)
            captions = [m for m in messages if m['type'] == 'transcript']
            assert [(m['window_start_ms'], m['window_end_ms']) for m in captions] == [(0, 2000), (1000, 7000)]
            assert captions[-1]['text'] == 'alpha beta gamma'
            ws.send_json({'type': 'stop'})
            receive_until(ws, stopped=True)
        assert [len(frames) for frames in model.calls] == [51, 151]
        assert all(a is b for a, b in zip(model.calls[0][25:], model.calls[1]))
        assert model.maximum_active == 1
    finally:
        model.release.set()


def test_stop_while_busy_supersedes_pending_partials_with_complete_final_window(session):
    client, model, buffers = session
    try:
        with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
            start(ws)
            send_frames(ws, 0, 51)
            assert model.entered.wait(2)
            send_frames(ws, 51, 104)
            ws.send_json({'type': 'stop'})
            assert buffers[0].wait_until(4120, final=True)
            model.release.set()
            messages = receive_until(ws, stopped=True)
            captions = [m for m in messages if m['type'] == 'transcript']
            assert [(m['window_end_ms'], m['final']) for m in captions] == [(2000, False), (4120, True)]
            assert captions[-1]['draft_text'] == ''
            assert captions[-1]['stable_text'] == 'alpha beta gamma'
        assert [len(frames) for frames in model.calls] == [51, 104]
        assert model.maximum_active == 1
        assert not server.session_lock.locked()
    finally:
        model.release.set()


def test_initial_window_cannot_start_after_unprocessed_video(monkeypatch, session):
    client, model, _ = session

    class DelayedFirstWindow(ClipBuffer):
        def append(self, packet):
            clip = super().append(packet)
            # Simulate a scheduler that offered nothing until capture had already
            # moved beyond the initial second; admission must reject the gap.
            return clip if clip is not None and clip.start_ms > 0 else None

    monkeypatch.setattr(server, 'ClipBuffer', DelayedFirstWindow)
    with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        start(ws)
        send_frames(ws, 0, 176)
        messages = receive_until(ws, error=True)
        assert 'preserve overlapping video' in messages[-1]['detail']
    assert not model.calls
    assert not server.session_lock.locked()


def test_disconnect_discards_pending_windows_but_holds_ownership_until_inflight_finishes(session):
    client, model, buffers = session
    try:
        with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
            start(ws)
            send_frames(ws, 0, 51)
            assert model.entered.wait(2)
            send_frames(ws, 51, 126)
            assert buffers[0].wait_until(5000)
            ws.close()
            with client.websocket_connect('/api/stream', headers=ORIGIN) as other:
                assert other.receive_json() == {'type': 'error', 'detail': 'Another reading session is active.'}
            assert server.session_lock.locked()
            model.release.set()
            assert model.finished.wait(2)
        assert len(model.calls) == 1
        assert model.maximum_active == 1
        assert not server.session_lock.locked()
    finally:
        model.release.set()

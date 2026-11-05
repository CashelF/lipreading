"""Exercise optimization ownership through the actual WebSocket application.

The model double records frame/cache identities instead of invoking a GPU. Pixel
and decoder equivalence are covered separately; this verifies the live wiring.
"""
import asyncio
from io import BytesIO
import struct

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from backend import app as server
from backend.preprocessing import PreprocessCache
from backend.stream import DecodeCache

ORIGIN = {'origin': 'http://localhost:5173'}


class RecordingModel:
    def __init__(self):
        self.calls = []
        self.failure = None

    def load(self):
        pass

    def status(self):
        return {'ready': True, 'model': 'integration double', 'detail': 'Ready'}

    def transcribe(self, frames, fps, *, cache=None, speech_gate=True):
        assert isinstance(cache, PreprocessCache)
        self.calls.append({'frames': tuple(frames), 'fps': fps, 'cache': cache,
                           'entries_before': cache.stats()['entries']})
        for frame in frames:
            cache.entry(frame)
        if self.failure:
            raise self.failure
        return {'text': 'test output', 'timings_ms': {'preprocessing': 1.25, 'inference': 2.5}}


@pytest.fixture
def session(monkeypatch):
    model = RecordingModel()
    decoded, processed = [], []

    def new_decode_cache():
        cache = DecodeCache()
        decoded.append(cache)
        return cache

    def new_preprocessing_cache():
        cache = PreprocessCache()
        processed.append(cache)
        return cache

    monkeypatch.setattr(server, 'model', model)
    monkeypatch.setattr(server, 'loading', False)
    monkeypatch.setattr(server, 'session_lock', asyncio.Lock())
    monkeypatch.setattr(server, 'DecodeCache', new_decode_cache)
    monkeypatch.setattr(server, 'PreprocessCache', new_preprocessing_cache)
    with TestClient(server.app) as client:
        yield client, model, decoded, processed


def send_frames(ws, start, stop):
    data = BytesIO()
    Image.new('RGB', (128, 128), (80, 100, 120)).save(data, format='JPEG')
    jpeg = data.getvalue()
    for index in range(start, stop):
        ws.send_bytes(struct.pack('<d', index * 40.) + jpeg)


def receive_cycle(ws, *, stopped=False):
    messages = []
    for _ in range(12):
        message = ws.receive_json()
        messages.append(message)
        if stopped and message['type'] == 'stopped':
            return messages
        if not stopped and message.get('state') == 'capturing':
            return messages
    pytest.fail('Worker did not finish its decode cycle')


def start(ws):
    ws.send_json({'type': 'start'})
    assert ws.receive_json()['state'] == 'capturing'


def assert_cleared(decoded, processed):
    assert not decoded._frames
    assert processed.stats()['entries'] == 0
    assert processed._detector is None


def test_rolling_windows_reuse_frames_until_session_end_and_stop_reuses_last_result(session):
    client, model, decoded, processed = session
    transcripts = []
    with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        start(ws)
        intervals = [(0, 51), *[(end - 25, end) for end in range(76, 502, 25)]]
        for begin, end in intervals:
            send_frames(ws, begin, end)
            messages = receive_cycle(ws)
            transcripts.extend(m for m in messages if m['type'] == 'transcript')
            assert len(decoded[0]._frames) <= decoded[0].max_frames
            assert processed[0].stats()['entries'] <= processed[0].max_frames
        # No six-second reset: bounded caches stay populated throughout reading.
        assert len(decoded[0]._frames) == decoded[0].max_frames
        assert processed[0].stats()['entries'] == processed[0].max_frames
        calls_before_stop = len(model.calls)
        ws.send_json({'type': 'stop'})
        transcripts.extend(m for m in receive_cycle(ws, stopped=True) if m['type'] == 'transcript')
    assert len(model.calls) == calls_before_stop == 19
    assert [len(call['frames']) for call in model.calls] == [51, 76, 101, 126, *([151] * 15)]
    assert [call['fps'] for call in model.calls] == [25] * 19
    assert [call['entries_before'] for call in model.calls] == [0, *[min(n, 400) for n in range(51, 477, 25)]]
    assert all(call['cache'] is processed[0] for call in model.calls)
    for index, (previous, current) in enumerate(zip(model.calls, model.calls[1:])):
        old_frames, new_frames = previous['frames'], current['frames']
        if index < 4:
            assert all(a is b for a, b in zip(old_frames, new_frames))
        else:
            assert all(a is b for a, b in zip(old_frames[25:], new_frames))
    assert all(not frame.flags.writeable for call in model.calls for frame in call['frames'])
    assert decoded[0].misses == 501
    assert decoded[0].hits == sum(len(call['frames']) for call in model.calls) - 501
    assert [(m['segment_id'], m['final']) for m in transcripts] == [
        *[(index, False) for index in range(19)], (19, True),
    ]
    for transcript in transcripts[:-1]:
        assert transcript['timings_ms']['preprocessing'] == 1.25
        assert transcript['timings_ms']['inference'] == 2.5
        assert transcript['timings_ms']['frame_decode'] >= 0
        assert transcript['timings_ms']['queue'] >= 0
    assert transcripts[-1]['timings_ms']['inference'] == 0
    assert transcripts[-1]['timings_ms']['frame_decode'] == 0
    assert_cleared(decoded[0], processed[0])


def test_stop_after_six_second_boundary_decodes_entire_overlapping_tail(session):
    client, model, decoded, processed = session
    with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        start(ws)
        for begin, end in [(0, 51), (51, 76), (76, 101), (101, 126), (126, 151)]:
            send_frames(ws, begin, end)
            receive_cycle(ws)
        previous = model.calls[-1]['frames']
        send_frames(ws, 151, 154)
        ws.send_json({'type': 'stop'})
        messages = receive_cycle(ws, stopped=True)
    final = model.calls[-1]
    assert len(model.calls) == 6
    assert len(final['frames']) == 151
    assert final['entries_before'] == 151
    assert all(a is b for a, b in zip(previous[3:], final['frames']))
    assert decoded[0].misses == 154
    final_message = next(m for m in messages if m['type'] == 'transcript')
    assert (final_message['window_start_ms'], final_message['window_end_ms']) == (120, 6120)
    assert final_message['final'] is True
    assert_cleared(decoded[0], processed[0])


def test_same_timestamps_in_new_connection_receive_independent_caches(session):
    client, model, decoded, processed = session
    for _ in range(2):
        with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
            start(ws)
            send_frames(ws, 0, 20)
            ws.send_json({'type': 'stop'})
            receive_cycle(ws, stopped=True)
    assert len(decoded) == len(processed) == 2
    assert decoded[0] is not decoded[1]
    assert processed[0] is not processed[1]
    assert model.calls[0]['frames'][0] is not model.calls[1]['frames'][0]
    for index in range(2):
        assert model.calls[index]['cache'] is processed[index]
        assert model.calls[index]['entries_before'] == 0
        assert_cleared(decoded[index], processed[index])


@pytest.mark.parametrize('failure', [ValueError('Capture quality'), RuntimeError('Inference failed')])
def test_failed_final_releases_cached_video_and_invalidates_caption(session, failure):
    client, model, decoded, processed = session
    model.failure = failure
    with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        start(ws)
        send_frames(ws, 0, 20)
        ws.send_json({'type': 'stop'})
        messages = receive_cycle(ws, stopped=True)
    assert len(model.calls) == 1
    assert any(m['type'] == 'transcript' and m['text'] == '' and m['final'] for m in messages)
    assert_cleared(decoded[0], processed[0])
    assert not server.session_lock.locked()


def test_disconnect_releases_a_partially_populated_cache(session):
    client, model, decoded, processed = session
    with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        start(ws)
        send_frames(ws, 0, 51)
        receive_cycle(ws)
        assert processed[0].stats()['entries'] == 51
        assert len(decoded[0]._frames) == 51
        ws.close()
    assert_cleared(decoded[0], processed[0])
    assert not server.session_lock.locked()

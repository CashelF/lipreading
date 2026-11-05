"""Still-mouth suppression must stop before CUDA and preserve the decode bypass.

All GPU-facing calls are explicit test doubles. No weights are loaded, and the
transport tests exercise the production WebSocket handler.
"""
import asyncio
from contextlib import nullcontext
from io import BytesIO
import struct
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from backend import app as server
from backend.model import FlareModel
from backend.preprocessing import PreprocessCache, Preprocessed

ORIGIN = {'origin': 'http://localhost:5173'}


def activity(state):
    return {'state': state, 'suppress': state == 'still',
            'detail': f'Test {state} activity', 'metrics': {'observed_fraction': 1.0}}


class NoTensorOperations:
    def __getattr__(self, name):
        raise AssertionError(f'Suppressed clip reached torch.{name}')


@pytest.mark.parametrize('state', ['still', 'uncertain'])
def test_suppressed_clip_skips_tensor_upload_and_decoder(monkeypatch, state):
    monkeypatch.setitem(sys.modules, 'torch', NoTensorOperations())
    model = FlareModel()
    model.model = object()
    frames = [object()]
    cache = PreprocessCache()
    seen = []
    still = {**activity(state), 'suppress': True}
    quality = {'frames': 51, 'fps': 25}

    def prepare(received, fps, *, cache=None, speech_gate=False):
        seen.append((received, fps, cache, speech_gate))
        return Preprocessed(None, None, quality, still)

    def forbidden_search(*args, **kwargs):
        pytest.fail('Suppressed clip invoked the FLARE decoder')

    model.preprocessor = prepare
    model._search = forbidden_search
    result = model.transcribe(frames, 25, cache=cache)
    assert seen == [(frames, 25, cache, True)]
    assert result['text'] == ''
    assert result['activity'] == still
    assert result['quality'] == quality
    assert result['detail'] == still['detail']
    assert result['timings_ms']['upload'] == result['timings_ms']['inference'] == 0.0


class TensorDouble:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.transfer = None

    def to(self, *, device, dtype):
        self.transfer = (device, dtype)
        return self


class TorchDouble:
    bfloat16 = 'test-bfloat16'
    bool = 'test-bool'

    def __init__(self):
        self.tensors = []

    def from_numpy(self, array):
        result = TensorDouble(array)
        self.tensors.append(result)
        return result

    def ones(self, shape, **kwargs):
        return ('ones', shape, kwargs)

    def zeros(self, shape, **kwargs):
        return ('zeros', shape, kwargs)

    def inference_mode(self):
        return nullcontext()


@pytest.mark.parametrize('gate,state', [(False, 'bypassed'), (True, 'motion'), (True, 'uncertain')])
def test_bypass_and_unsuppressed_clips_preserve_original_decoder_settings(monkeypatch, gate, state):
    monkeypatch.delenv('FLARE_BEAM_SIZE', raising=False)
    torch = TorchDouble()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    model = FlareModel()
    model.model = object()
    cache = PreprocessCache()
    frames = [object()]
    face = np.zeros((1, 1, 51, 88, 88), dtype=np.float32)
    lips = np.ones_like(face)
    seen = []
    searches = []
    detected = activity(state)

    def prepare(received, fps, *, cache=None, speech_gate=False):
        seen.append((received, fps, cache, speech_gate))
        return Preprocessed(face, lips, {'frames': 51}, detected)

    def search(models, sample, **settings):
        searches.append((models, sample, settings))
        return ['  <sil> same   decoded words  '], [('unchanged score', 1.0)]

    model.preprocessor = prepare
    model._search = search
    result = model.transcribe(frames, 25, cache=cache, speech_gate=gate)
    assert seen == [(frames, 25, cache, gate)]
    assert result['text'] == 'same decoded words'
    assert result['activity'] == detected
    assert len(searches) == 1
    models, sample, settings = searches[0]
    assert models == [model.model]
    assert settings == {'beam_size': 30, 'audio_only': False, 'video_only': True, 'temp': 1.2}
    assert torch.tensors[0].array is face
    assert torch.tensors[1].array is lips
    assert all(t.transfer == ('cuda', torch.bfloat16) for t in torch.tensors)
    source = sample['net_input']['source']
    assert source['audio'] is None
    assert source['video'] == tuple(torch.tensors)
    assert source['face_lip_mask'] == (('ones', 1, {'device': 'cuda'}),) * 2
    assert sample['net_input']['padding_mask'] == ('zeros', (1, 51), {'dtype': torch.bool, 'device': 'cuda'})


class GateModel:
    def __init__(self):
        self.calls = []
        self.results = []

    def load(self):
        pass

    def status(self):
        return {'ready': True, 'detail': 'Ready', 'model': 'explicit test double'}

    def transcribe(self, frames, fps, *, cache=None, speech_gate=True):
        self.calls.append((len(frames), fps, cache, speech_gate))
        if self.results:
            return self.results.pop(0)
        state = 'still' if speech_gate else 'bypassed'
        return {'text': '' if speech_gate else 'test caption', 'activity': activity(state)}


@pytest.fixture
def client(monkeypatch):
    model = GateModel()
    monkeypatch.setattr(server, 'model', model)
    monkeypatch.setattr(server, 'session_lock', asyncio.Lock())
    monkeypatch.setattr(server, 'loading', False)
    with TestClient(server.app) as test_client:
        yield test_client, model


def send_frames(ws, start=0, end=20):
    out = BytesIO()
    Image.new('RGB', (128, 128), (80, 90, 100)).save(out, format='JPEG')
    for index in range(start, end):
        ws.send_bytes(struct.pack('<d', index * 40.) + out.getvalue())


def receive_cycle(ws, *, stopped=False):
    messages = []
    for _ in range(15):
        message = ws.receive_json()
        messages.append(message)
        if stopped and message['type'] == 'stopped':
            return messages
        if not stopped and message.get('state') == 'capturing':
            return messages
    pytest.fail('Stream did not finish its processing cycle')


@pytest.mark.parametrize('command,expected', [({'type': 'start'}, True),
    ({'type': 'start', 'speech_gate': True}, True), ({'type': 'start', 'speech_gate': False}, False)])
def test_start_passes_default_and_explicit_filter_choice_to_model(client, command, expected):
    test_client, model = client
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json(command)
        assert ws.receive_json()['state'] == 'capturing'
        send_frames(ws)
        ws.send_json({'type': 'stop'})
        messages = receive_cycle(ws, stopped=True)
    assert len(model.calls) == 1
    count, fps, cache, enabled = model.calls[0]
    assert (count, fps, enabled) == (20, 25, expected)
    assert isinstance(cache, PreprocessCache)
    activity_index = next(i for i, m in enumerate(messages) if m['type'] == 'activity')
    transcript_index = next(i for i, m in enumerate(messages) if m['type'] == 'transcript')
    assert activity_index < transcript_index
    assert messages[activity_index] == {**activity('still' if expected else 'bypassed'), 'type': 'activity', 'segment_id': 0}
    assert messages[transcript_index]['text'] == ('' if expected else 'test caption')
    assert messages[transcript_index]['final'] is True
    assert not server.session_lock.locked()


@pytest.mark.parametrize('bad_value', [0, 1, 'false', None, [], {}])
@pytest.mark.parametrize('field', ['speech_gate', 'verify_visual'])
def test_invalid_filter_type_rejected_without_leaking_session_lock(client, bad_value, field):
    test_client, model = client
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start', field: bad_value})
        assert ws.receive_json() == {'type': 'error', 'detail': f'{field} must be true or false.'}
    assert not model.calls
    assert not server.session_lock.locked()


def test_filter_choice_does_not_leak_into_next_session(client):
    test_client, model = client
    for command in ({'type': 'start', 'speech_gate': False}, {'type': 'start'}):
        with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
            ws.send_json(command)
            ws.receive_json()
            send_frames(ws)
            ws.send_json({'type': 'stop'})
            receive_cycle(ws, stopped=True)
    assert [call[3] for call in model.calls] == [False, True]


def test_suppression_retracts_same_origin_draft_without_redundant_stop_decode(client):
    test_client, model = client
    model.results = [
        {'text': 'provisional phrase', 'activity': activity('motion')},
        {'text': '', 'activity': activity('still')},
    ]
    all_messages = []
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start'})
        ws.receive_json()
        for begin, end in ((0, 51), (51, 76)):
            send_frames(ws, begin, end)
            all_messages.extend(receive_cycle(ws))
        ws.send_json({'type': 'stop'})
        final_messages = receive_cycle(ws, stopped=True)
        all_messages.extend(final_messages)
    finals = [m for m in all_messages if m['type'] == 'transcript' and m['final']]
    assert [m['text'] for m in finals] == ['']
    assert any(m['type'] == 'transcript' and m['text'] == 'provisional phrase' for m in all_messages)
    assert len(model.calls) == 2
    assert all(m['suppress'] for m in final_messages if m['type'] == 'activity')


def test_recent_speech_gate_keeps_existing_words_when_speech_stops(client):
    test_client, model = client
    model.results = [
        {'text': 'words before the pause', 'activity': activity('motion')},
        {'text': '', 'activity': {**activity('still'), 'preserve_transcript': True}},
    ]
    snapshots = []
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start'})
        ws.receive_json()
        for begin, end in ((0, 51), (51, 76)):
            send_frames(ws, begin, end)
            snapshots.extend(m for m in receive_cycle(ws) if m['type'] == 'transcript')
        ws.send_json({'type': 'stop'})
        snapshots.extend(m for m in receive_cycle(ws, stopped=True) if m['type'] == 'transcript')
    assert all(m['text'] == 'words before the pause' for m in snapshots)
    assert snapshots[-1]['stable_text'] == 'words before the pause'
    assert snapshots[-1]['final']
    assert len(model.calls) == 2
    assert [m['revision'] for m in snapshots] == sorted({m['revision'] for m in snapshots})


def test_sliding_silent_windows_preserve_speech_before_their_start(client):
    test_client, model = client
    model.results = [{'text': 'earlier phrase', 'activity': activity('motion')} for _ in range(5)]
    snapshots = []
    with test_client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start'})
        ws.receive_json()
        begin = 0
        # Five growing windows contain speech; subsequent moving windows are
        # classified still. Their silence must not wipe prior spoken words.
        for end in range(51, 352, 25):
            send_frames(ws, begin, end)
            begin = end
            snapshots.extend(m for m in receive_cycle(ws) if m['type'] == 'transcript')
        ws.send_json({'type': 'stop'})
        snapshots.extend(m for m in receive_cycle(ws, stopped=True) if m['type'] == 'transcript')
    assert all(m['text'] == 'earlier phrase' for m in snapshots)
    assert snapshots[-1]['stable_text'] == 'earlier phrase'
    assert snapshots[-1]['draft_text'] == ''
    assert snapshots[-1]['final'] is True
    assert [m['revision'] for m in snapshots] == sorted({m['revision'] for m in snapshots})

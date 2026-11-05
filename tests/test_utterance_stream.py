"""Full utterance previews/finalization with real JPEG transport and fake models."""
import asyncio
from collections import deque
from io import BytesIO
import struct

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from backend import app as server
from backend.speech_gate import SpeechGateState
from backend.stream import Frame
from backend.utterance_stream import UtteranceReader


def jpeg():
    output = BytesIO()
    Image.new('RGB', (96, 96), (70, 80, 90)).save(output, format='JPEG')
    return output.getvalue()


def activity(state, end, *, first=None, last=None):
    metrics = {'speech_evidence_end_ms': end - 480.}
    if state == 'motion':
        metrics.update(speech_first_active_ms=first if first is not None else 2600.,
                       speech_last_active_ms=last if last is not None else end - 480.)
    return {'state': state, 'suppress': state != 'motion',
            'preserve_transcript': state != 'motion', 'metrics': metrics,
            'detail': f'Controlled {state} activity'}


class UtteranceModel:
    def __init__(self, inspections=(), previews=(), finals=()):
        self.inspections = deque(inspections)
        self.previews = deque(previews)
        self.finals = deque(finals)
        self.calls = []
        self.inspection_times = []

    def load(self):
        pass

    def status(self):
        return {'ready': True, 'model': 'explicit utterance test double', 'detail': 'Ready'}

    def inspect_activity(self, frames, fps=25, *, cache=None, speech_gate=True):
        self.inspection_times.append(cache.sample_times_ms.copy())
        assert len(frames) == len(cache.sample_times_ms)
        assert fps == 25
        return {'activity': self.inspections.popleft(), 'quality': {'frames': len(frames)},
                'timings_ms': {'preprocessing': 1., 'upload': 0., 'inference': 0.}}

    def _decode(self, kind, frames, fps, cache, outputs):
        self.calls.append({'kind': kind, 'frames': len(frames), 'fps': fps,
                           'first': frames[0], 'last': frames[-1], 'cache': cache})
        return {'text': outputs.popleft() if outputs else f'test {kind}',
                'timings_ms': {'preprocessing': 2., 'upload': 1., 'inference': 10.}}

    def decode_utterance(self, frames, fps=25, *, cache=None):
        return self._decode('preview', frames, fps, cache, self.previews)

    def refine_utterance(self, frames, fps=25, *, cache=None):
        return self._decode('final', frames, fps, cache, self.finals)


class Scenario:
    def __init__(self, model, *, origin=1000.):
        self.model, self.reader = model, UtteranceReader(model)
        self.origin = origin
        self.next_timestamp = origin
        self.messages = []
        self.segment = 0
        self.image = jpeg()

    async def send(self, message):
        self.messages.append(message)

    async def advance(self, end, state='motion', *, final=False, first=None, last=None,
                      speech_gate=True, inspection=None):
        while self.next_timestamp <= end:
            self.reader.history.append(Frame(self.next_timestamp, self.image))
            self.next_timestamp += 40.
        start = max(self.origin, end - 6000.)
        clip = self.reader.history.slice(start, end, self.segment, final=final)
        self.segment += 1
        self.model.inspections.append(inspection if inspection is not None else
                                      activity(state, end, first=first, last=last))
        return await self.reader.process(clip, speech_gate, self.send)


def test_quiet_session_never_previews_or_refines_including_stop():
    async def run():
        model = UtteranceModel()
        scenario = Scenario(model)
        await scenario.advance(3000., 'still')
        await scenario.advance(4000., 'still')
        result = await scenario.advance(5000., 'still', final=True)
        assert result['snapshot']['final']
        assert result['snapshot']['text'] == ''
        assert model.calls == []
        assert not any(message['state'] == 'refining' for message in scenario.messages)
    asyncio.run(run())


def test_live_preview_replaces_every_word_and_preserves_utterance_start_beyond_six_seconds():
    async def run():
        model = UtteranceModel(previews=['we should send it', 'we could send it tomorrow'])
        scenario = Scenario(model)
        first = await scenario.advance(3000.)
        newer = await scenario.advance(9000.)
        assert first['snapshot']['draft_text'] == 'we should send it'
        assert newer['snapshot']['draft_text'] == 'we could send it tomorrow'
        assert newer['snapshot']['stable_text'] == ''
        assert [call['frames'] for call in model.calls] == [51, 201]
        assert model.calls[0]['first'] is model.calls[1]['first']
        assert model.calls[0]['cache'] is scenario.reader.utterance_preprocess
        assert model.calls[0]['cache'] is not scenario.reader.gate_preprocess
        assert np.array_equal(model.inspection_times[0], np.arange(1000., 3001., 40.))
        assert np.all(np.diff(model.inspection_times[1]) == 40.)
    asyncio.run(run())


def test_pause_refines_once_replaces_leading_correction_and_then_holds():
    async def run():
        model = UtteranceModel(previews=['we should send it tomorrow'],
                               finals=['we could send it tomorrow'])
        scenario = Scenario(model)
        await scenario.advance(3000.)
        refined = await scenario.advance(4000., 'still')
        held = await scenario.advance(5000., 'still')
        stopped = await scenario.advance(5000., 'still', final=True)
        assert refined['activity']['suppress']
        assert refined['timings_ms']['refinement_passes'] == 1
        assert refined['snapshot']['stable_text'] == 'we could send it tomorrow'
        assert refined['snapshot']['draft_text'] == ''
        assert held['snapshot']['text'] == stopped['snapshot']['text'] == refined['snapshot']['text']
        assert [call['kind'] for call in model.calls] == ['preview', 'final']
        assert stopped['snapshot']['final']
        assert len([m for m in scenario.messages if m['state'] == 'refining']) == 1
    asyncio.run(run())


def test_stop_refines_active_utterance_once_even_when_last_source_is_unchanged():
    async def run():
        model = UtteranceModel(previews=['I need a coffee'], finals=['We need some coffee'])
        scenario = Scenario(model)
        await scenario.advance(3000.)
        result = await scenario.advance(3000., 'still', final=True)
        assert result['snapshot']['stable_text'] == 'We need some coffee'
        assert result['snapshot']['draft_text'] == ''
        assert result['snapshot']['final']
        assert [call['kind'] for call in model.calls] == ['preview', 'final']
        assert model.calls[0]['frames'] == model.calls[1]['frames'] == 51
    asyncio.run(run())


def test_coalesced_motion_then_quiet_refines_immediately_and_exactly_once():
    async def run():
        model = UtteranceModel(previews=['we should send it'], finals=['we could send it'])
        scenario = Scenario(model)
        gate = SpeechGateState()
        first_times = np.arange(1000., 3001., 40.)
        first = gate.update(np.full(len(first_times), .8), first_times)
        await scenario.advance(3000., inspection=first)

        # Exercise the actual gate's interval maximum and last-active semantics:
        # an old active mean keeps state=motion despite 880 ms of proven silence.
        mixed_times = np.arange(2000., 10041., 40.)
        mixed = gate.update(np.where(mixed_times <= 8200., .8, 0.), mixed_times)
        assert mixed['state'] == 'motion'
        assert mixed['metrics']['speech_last_active_ms'] == 8680.
        assert mixed['metrics']['speech_evidence_end_ms'] == 9560.
        assert mixed['metrics']['speech_latest_mean'] == 0.
        refined = await scenario.advance(10040., inspection=mixed)
        held = await scenario.advance(11040., 'still')
        stopped = await scenario.advance(11040., 'still', final=True)

        assert refined['timings_ms']['refinement_passes'] == 1
        assert refined['snapshot']['stable_text'] == 'we could send it'
        assert refined['snapshot']['draft_text'] == ''
        assert held['snapshot']['text'] == stopped['snapshot']['text'] == 'we could send it'
        assert [call['kind'] for call in model.calls] == ['preview', 'final']
        assert model.calls[1]['frames'] == 205
        assert model.calls[0]['first'] is model.calls[1]['first']
        assert len([m for m in scenario.messages if m['state'] == 'refining']) == 1
    asyncio.run(run())


@pytest.mark.parametrize(('silence_ms', 'expected_kind'), [(760., 'preview'), (800., 'final')])
def test_motion_interval_requires_the_full_existing_pause_duration(silence_ms, expected_kind):
    async def run():
        model = UtteranceModel()
        scenario = Scenario(model)
        await scenario.advance(3000.)
        result = await scenario.advance(10040., first=6520., last=9560. - silence_ms)
        assert [call['kind'] for call in model.calls] == ['preview', expected_kind]
        assert result['timings_ms']['refinement_passes'] == int(expected_kind == 'final')
    asyncio.run(run())


@pytest.mark.parametrize(('state', 'flags'), [
    ('uncertain', {}),
    ('motion', {'speech_invalid': 1}),
    ('motion', {'speech_gap': 1}),
    ('still', {'speech_invalid': 1}),
    ('still', {'speech_gap': 1}),
])
def test_unreliable_activity_cannot_prove_a_pause(state, flags):
    async def run():
        model = UtteranceModel()
        scenario = Scenario(model)
        await scenario.advance(3000.)
        inspection = activity(state, 10040., first=6520., last=8680.)
        inspection['metrics'].update(flags, speech_last_active_ms=8680.)
        result = await scenario.advance(10040., inspection=inspection)
        assert result['timings_ms']['refinement_passes'] == 0
        assert result['snapshot']['stable_text'] == ''
        assert scenario.reader.start_ms == 1000.
        assert not any(call['kind'] == 'final' for call in model.calls)
    asyncio.run(run())


def test_motion_without_explicit_last_active_evidence_keeps_the_utterance_open():
    async def run():
        model = UtteranceModel()
        scenario = Scenario(model)
        await scenario.advance(3000.)
        inspection = activity('motion', 10040., first=6520.)
        inspection['metrics'].pop('speech_last_active_ms')
        result = await scenario.advance(10040., inspection=inspection)
        assert result['timings_ms']['refinement_passes'] == 0
        assert [call['kind'] for call in model.calls] == ['preview', 'preview']
        assert scenario.reader.last_speech_ms == 9560.
    asyncio.run(run())


def test_first_coalesced_activity_request_recovers_early_source_video():
    async def run():
        model = UtteranceModel(previews=['speech near the beginning'])
        scenario = Scenario(model)
        # A pending request has moved beyond the initial six-second window. The
        # gate still receives all unexamined video from the retained history.
        result = await scenario.advance(8000.)
        assert model.inspection_times[0][0] == 1000.
        assert model.inspection_times[0][-1] == 8000.
        assert len(model.inspection_times[0]) == model.calls[0]['frames'] == 176
        assert result['snapshot']['draft_text'] == 'speech near the beginning'
    asyncio.run(run())


def test_second_utterance_preserves_committed_prefix_without_fuzzy_merge():
    async def run():
        model = UtteranceModel(previews=['first draft', 'second draft'],
                               finals=['first corrected', 'second corrected'])
        scenario = Scenario(model)
        await scenario.advance(3000.)
        await scenario.advance(4000., 'still')
        second = await scenario.advance(6000., first=5200., last=5520.)
        assert second['snapshot']['stable_text'] == 'first corrected'
        assert second['snapshot']['draft_text'] == 'second draft'
        final = await scenario.advance(7000., 'still', final=True)
        assert final['snapshot']['text'] == 'first corrected second corrected'
        assert final['snapshot']['draft_text'] == ''
        assert [call['kind'] for call in model.calls] == ['preview', 'final', 'preview', 'final']
    asyncio.run(run())


def test_twelve_second_utterance_finalizes_as_one_complete_input():
    async def run():
        model = UtteranceModel(previews=['draft'], finals=['complete twelve-second utterance'])
        scenario = Scenario(model)
        await scenario.advance(3000.)
        await scenario.advance(7000.)
        await scenario.advance(11000.)
        result = await scenario.advance(13000., final=True)
        final_calls = [call for call in model.calls if call['kind'] == 'final']
        assert [call['frames'] for call in final_calls] == [301]
        assert result['snapshot']['text'] == 'complete twelve-second utterance'
        assert result['snapshot']['final']
        assert result['timings_ms']['refinement_passes'] == 1
    asyncio.run(run())


@pytest.mark.parametrize('regions', [False, True])
def test_full_minute_preserves_first_frame_and_finishes_once(regions):
    async def run():
        model = UtteranceModel(finals=['complete long utterance'])
        scenario = Scenario(model)
        scenario.reader.set_utterance_limit(60)
        outputs = []
        for end in (3000., 7000., 13000., 21000., 31000., 41000., 51000., 61000.):
            inspection = None
            if regions:
                inspection = activity('motion', end, first=2600., last=end - 480.)
                inspection['metrics']['speech_regions_version'] = 1
                inspection['speech_regions'] = [{'start_ms': 2600., 'confirmed_ms': 2680.,
                                                 'last_active_ms': end - 480., 'closed_at_ms': None,
                                                 'fresh_active': True}]
            outputs.append(await scenario.advance(end, final=end == 61000., inspection=inspection))
        assert all(call['first'] is model.calls[0]['first'] for call in model.calls)
        assert model.calls[-1]['frames'] == 1501
        assert [call['kind'] for call in model.calls].count('final') == 1
        assert outputs[-1]['snapshot']['text'] == 'complete long utterance'
        assert all(not output['snapshot']['stable_text'] for output in outputs[:-1])
        assert outputs[-1]['snapshot']['final']
        assert scenario.reader.utterance_decode.max_frames >= 1501
        assert scenario.reader.utterance_preprocess.max_frames >= 1501
    asyncio.run(run())


def test_default_thirty_second_limit_preserves_draft_at_the_boundary():
    async def run():
        model = UtteranceModel(previews=['latest words'] * 4)
        scenario = Scenario(model)
        assert scenario.reader.max_utterance_seconds == 30
        for end in (3000., 11000., 21000., 31000.):
            await scenario.advance(end, speech_gate=False)
        assert model.calls[-1]['frames'] == 751
        with pytest.raises(RuntimeError, match='30 seconds'):
            await scenario.advance(32000., speech_gate=False)
        assert scenario.reader.transcript.snapshot()['text'] == 'latest words'
    asyncio.run(run())


@pytest.mark.parametrize('value', [True, 0, 12, 31, 61, 30.0, '60', None])
def test_websocket_rejects_invalid_utterance_limits(monkeypatch, value):
    monkeypatch.setattr(server, 'model', UtteranceModel())
    with TestClient(server.app) as client, client.websocket_connect('/api/stream', headers={'origin': 'http://localhost:8000'}) as ws:
        ws.send_json({'type': 'start', 'max_utterance_seconds': value})
        message = ws.receive_json()
        assert message['type'] == 'error' and '30 or 60 seconds' in message['detail']


@pytest.mark.parametrize('seconds', [30, 60])
def test_websocket_admits_and_reports_selected_utterance_limit(monkeypatch, seconds):
    monkeypatch.setattr(server, 'model', UtteranceModel())
    with TestClient(server.app) as client, client.websocket_connect('/api/stream', headers={'origin': 'http://localhost:8000'}) as ws:
        ws.send_json({'type': 'start', 'max_utterance_seconds': seconds})
        message = ws.receive_json()
        assert message['state'] == 'capturing' and message['max_utterance_seconds'] == seconds


def test_pauses_allow_a_session_longer_than_one_minute():
    async def run():
        model = UtteranceModel(finals=['first long phrase', 'second long phrase'])
        scenario = Scenario(model)
        scenario.reader.set_utterance_limit(60)
        for end in range(3000, 40001, 1000):
            await scenario.advance(float(end))
        first = await scenario.advance(41000., 'still')
        assert first['snapshot']['stable_text'] == 'first long phrase'
        for end in range(44000, 83001, 1000):
            await scenario.advance(float(end), first=43600.)
        final = await scenario.advance(84000., 'still', final=True)
        assert final['snapshot']['text'] == 'first long phrase second long phrase'
        assert len([call for call in model.calls if call['kind'] == 'final']) == 2
    asyncio.run(run())


@pytest.mark.parametrize('end', [61400., 63000.])
def test_overlong_utterance_errors_without_silently_dropping_or_splitting_tail(end):
    async def run():
        model = UtteranceModel(previews=['first draft', 'more context', 'last complete draft'])
        scenario = Scenario(model)
        scenario.reader.set_utterance_limit(60)
        await scenario.advance(3000.)
        await scenario.advance(7000.)
        await scenario.advance(11000.)
        before = scenario.reader.transcript.snapshot()
        with pytest.raises(RuntimeError, match='60.second'):
            await scenario.advance(end, final=True, speech_gate=False)
        assert scenario.reader.transcript.snapshot()['text'] == before['text']
        assert not scenario.reader.transcript.snapshot()['final']
        assert not any(call['kind'] == 'final' for call in model.calls)
        # The application can explicitly end the failed session and retain the
        # previous draft; there is no purportedly successful truncated decode.
        assert scenario.reader.transcript.finish()['text'] == 'last complete draft'
    asyncio.run(run())


def test_real_websocket_path_runs_one_pause_refinement_and_keeps_quiet_results(monkeypatch):
    model = UtteranceModel(
        inspections=[activity('motion', 2000., first=1600.), activity('still', 3000.),
                     activity('still', 3000.)],
        previews=['we should send it'], finals=['we could send it'])
    monkeypatch.setattr(server, 'model', model)
    monkeypatch.setattr(server, 'session_lock', asyncio.Lock())
    monkeypatch.setattr(server, 'loading', False)
    encoded = jpeg()

    def receive_cycle(ws, stopped=False):
        result = []
        for _ in range(30):
            message = ws.receive_json()
            assert message['type'] != 'error'
            result.append(message)
            if message['type'] == 'stopped' or (not stopped and message.get('state') == 'capturing'):
                return result
        raise AssertionError('The utterance worker did not complete its cycle.')

    with TestClient(server.app) as client:
        with client.websocket_connect('/api/stream', headers={'origin': 'http://localhost:5173'}) as ws:
            ws.send_json({'type': 'start'})
            ws.receive_json()
            snapshots = []
            for first, end in ((0, 51), (51, 76)):
                for index in range(first, end):
                    ws.send_bytes(struct.pack('<d', index * 40.) + encoded)
                snapshots.extend(m for m in receive_cycle(ws) if m['type'] == 'transcript')
            ws.send_json({'type': 'stop'})
            snapshots.extend(m for m in receive_cycle(ws, stopped=True) if m['type'] == 'transcript')
    assert snapshots[0]['draft_text'] == 'we should send it'
    assert snapshots[-1]['text'] == 'we could send it'
    assert snapshots[-1]['stable_text'] == 'we could send it'
    assert snapshots[-1]['final']
    assert [call['kind'] for call in model.calls] == ['preview', 'final']
    assert not server.session_lock.locked()

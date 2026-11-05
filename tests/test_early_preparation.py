"""Early work must neither consume speech evidence nor compete with decoding."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend import app as server
from backend.preprocessing import FrameCache, PreprocessCache
from backend.stream import Frame, PreparationSampler
from test_preprocessing_cache import moving_capture, assert_identical
from test_visual_gate_preprocessing import make_processor
from test_rolling_transport import ORIGIN, start, send_frames, receive_until


@pytest.mark.parametrize('fps', [15, 23, 25, 30, 40])
def test_early_sampling_matches_offline_nearest_neighbor_grid(fps):
    rng = np.random.default_rng(183)
    times = 8131.7 + np.cumsum(rng.uniform(.8, 1.2, 250) * 1000 / fps)
    frames = [Frame(float(t), b'jpeg') for t in times]
    sampler = PreparationSampler()
    selected = [source for frame in frames for source in sampler.append(frame)]
    target = times[0] + np.arange(int(np.floor((times[-1]-times[0])/40)) + 1) * 40
    expected = np.unique(np.abs(times[:, None] - target).argmin(axis=0))
    assert [id(frame) for frame in selected] == [id(frames[index]) for index in expected]


def test_early_sampler_ties_and_large_gaps_keep_work_bounded():
    sampler = PreparationSampler()
    frames = [Frame(t, b'jpeg') for t in (0., 20., 60., 10**10)]
    selected = [source for frame in frames for source in sampler.append(frame)]
    assert [frame.timestamp for frame in selected] == [0., 20., 60., 10**10]


@pytest.mark.parametrize('dropouts', [(), (0, 22, 49)])
def test_parallel_warps_and_early_observations_preserve_growing_inputs(dropouts):
    frames, observations = moving_capture(101, dropouts=dropouts)
    old = make_processor(frames, observations)
    new = make_processor(frames, observations)
    shared = FrameCache()
    early = PreprocessCache(frames=shared)
    gate = PreprocessCache(frames=shared)
    recognition = PreprocessCache(frames=shared)
    old_gate, old_recognition = PreprocessCache(), PreprocessCache()
    with ThreadPoolExecutor(max_workers=4) as new.roi_pool:
        for count in (51, 76, 101):
            for frame in frames[:count]:
                new.prepare_frame(frame, early, speech_gate=True)
            assert early.speech_state is early.sample_times_ms is None
            times = np.arange(count) * 40.
            gate.set_sample_times(times)
            old_gate.set_sample_times(times)
            actual = new(frames[:count], cache=gate, speech_gate=True, activity_only=True)
            expected = old(frames[:count], cache=old_gate, speech_gate=True, activity_only=True)
            assert actual.activity == expected.activity
            assert actual.quality == expected.quality
            watermark = gate.speech_state.last_timestamp_ms
            assert_identical(new(frames[:count], cache=recognition),
                             old(frames[:count], cache=old_recognition))
            assert gate.speech_state.last_timestamp_ms == watermark
    assert gate.detector_calls == recognition.detector_calls == 0
    assert new.detector.calls == 101


def test_parallel_repeated_frames_and_changing_alignment_are_exact():
    frames, observations = moving_capture(51)
    repeated = [f for frame in frames for f in (frame, frame)]
    old, new = make_processor(frames, observations), make_processor(frames, observations)
    cache = PreprocessCache()
    with ThreadPoolExecutor(max_workers=4) as new.roi_pool:
        for count in (52, 76, 102, 102):
            assert_identical(new(repeated[:count], cache=cache), old(repeated[:count]))
        assert all(len(entry.rois) <= 4 for entry in cache._entries.values())


class EarlyModel:
    """Block one speculative frame while transport receives a full window."""
    def __init__(self, failure=None):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.failure = failure
        self.events = []

    def load(self):
        pass

    def status(self):
        return {'ready': True}

    def prepare_frame(self, frame, *, cache, speech_gate):
        self.events.append('prepare')
        self.entered.set()
        assert self.release.wait(5)
        if self.failure:
            raise self.failure

    def inspect_activity(self, frames, fps, *, cache, speech_gate):
        self.events.append('inspect')
        return {'activity': {'state': 'still', 'suppress': True, 'metrics': {
            'speech_regions_version': 1, 'speech_evidence_end_ms': cache.sample_times_ms[-1]-480}},
                'timings_ms': {'preprocessing': 1.}}


@pytest.mark.parametrize('failure', [None, ValueError('bad speculative frame'), RuntimeError('preparation unavailable')])
def test_pending_window_takes_priority_and_stop_drops_preparation_backlog(monkeypatch, failure):
    model = EarlyModel(failure)
    monkeypatch.setattr(server, 'model', model)
    monkeypatch.setattr(server, 'loading', False)
    monkeypatch.setattr(server, 'session_lock', asyncio.Lock())
    stop_received = threading.Event()
    class Buffer(server.ClipBuffer):
        def finish(self):
            result = super().finish()
            stop_received.set()
            return result
    monkeypatch.setattr(server, 'ClipBuffer', Buffer)
    readers = []
    original = server.UtteranceReader
    class Reader(original):
        def __init__(self, model):
            super().__init__(model)
            readers.append(self)
    monkeypatch.setattr(server, 'UtteranceReader', Reader)
    try:
        with TestClient(server.app) as client:
            with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
                start(ws)
                send_frames(ws, 0, 1)
                assert model.entered.wait(2)
                send_frames(ws, 1, 76)
                ws.send_json({'type': 'stop'})
                assert stop_received.wait(2)
                model.release.set()
                receive_until(ws, stopped=True)
        assert model.events[0] == 'prepare'
        assert model.events[1] == 'inspect'
        assert model.events.count('prepare') == 1
        assert not server.session_lock.locked()
        assert not readers[0].gate_decode._frames
        assert not readers[0].gate_preprocess.frames.entries
        assert readers[0].prefetch_preprocess.speech_state is None
    finally:
        model.release.set()

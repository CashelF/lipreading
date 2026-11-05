"""Activity inspection must be independent from full-utterance recognition."""
from contextlib import nullcontext
import sys

import numpy as np
import pytest

from backend.model import FlareModel, ModelUnavailable
from backend.preprocessing import PreprocessCache, Preprocessed


class ForbiddenTorch:
    def __getattr__(self, name):
        raise AssertionError(f'Activity-only inspection reached torch.{name}')


@pytest.mark.parametrize('state', ['motion', 'still', 'uncertain', 'bypassed'])
def test_inspection_never_uploads_or_decodes_even_when_speech_is_detected(monkeypatch, state):
    runtime = FlareModel()
    runtime.model = object()
    cache = PreprocessCache()
    frames = [object()] * 51
    quality = {'frames': 51, 'fps': 25}
    activity = {'state': state, 'suppress': state in ('still', 'uncertain'),
                'detail': f'Test {state}', 'metrics': {}}
    gate = state != 'bypassed'
    calls = []

    def prepare(received, fps, *, cache=None, speech_gate=True, activity_only=False):
        assert runtime._lock.locked()
        assert activity_only is True
        calls.append((received, fps, cache, speech_gate))
        # An admitted preprocessor can return tensors-to-be; inspection must not
        # upload them or retain them in its result.
        return Preprocessed(object(), object(), quality, activity if gate else None)

    def forbidden(*args, **kwargs):
        pytest.fail('Activity inspection invoked the FLARE decoder')

    runtime.preprocessor = prepare
    runtime._search = forbidden
    monkeypatch.setitem(sys.modules, 'torch', ForbiddenTorch())
    result = runtime.inspect_activity(frames, cache=cache, speech_gate=gate)
    assert calls == [(frames, 25, cache, gate)]
    assert result['activity']['state'] == state
    assert result['quality'] == quality
    assert result['timings_ms']['upload'] == result['timings_ms']['inference'] == 0
    assert result['timings_ms']['preprocessing'] >= 0
    assert 'text' not in result and 'face' not in result and 'lips' not in result
    assert not runtime._lock.locked()


class TensorDouble:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.transfer = None

    def to(self, *, device, dtype):
        self.transfer = device, dtype
        return self


class TorchDouble:
    bfloat16 = 'bf16'
    bool = 'bool'

    def __init__(self):
        self.tensors = []

    def from_numpy(self, array):
        tensor = TensorDouble(array)
        self.tensors.append(tensor)
        return tensor

    def ones(self, *args, **kwargs):
        return ('ones', args, kwargs)

    def zeros(self, *args, **kwargs):
        return ('zeros', args, kwargs)

    def inference_mode(self):
        return nullcontext()


@pytest.mark.parametrize('method', ['decode_utterance', 'refine_utterance'])
@pytest.mark.parametrize('frame_count', [301, 1501])
def test_full_utterance_decode_keeps_all_frames_and_original_search_settings(monkeypatch, method, frame_count):
    monkeypatch.delenv('FLARE_BEAM_SIZE', raising=False)
    torch = TorchDouble()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    runtime = FlareModel()
    runtime.model = object()
    # Full utterances must not be truncated to the six-second gate clip.
    frames = [object()] * frame_count
    face = np.zeros((1, 1, frame_count, 88, 88), dtype=np.float32)
    lips = np.ones_like(face)
    activity_cache = PreprocessCache()
    watermark = object()
    activity_cache.speech_state = watermark
    decode_cache = PreprocessCache()
    prepared_calls, searches = [], []

    def prepare(received, fps, *, cache=None, speech_gate=True):
        assert runtime._lock.locked()
        prepared_calls.append((received, fps, cache, speech_gate))
        return Preprocessed(face, lips, {'frames': frame_count, 'fps': 25})

    def search(models, sample, **settings):
        searches.append((models, sample, settings))
        return ['  full utterance   correction '], []

    runtime.preprocessor, runtime._search = prepare, search
    result = getattr(runtime, method)(frames, cache=decode_cache, verify_visual=False)
    assert prepared_calls == [(frames, 25, decode_cache, False)]
    assert activity_cache.speech_state is watermark
    assert result['text'] == 'full utterance correction'
    assert len(searches) == 1
    models, sample, settings = searches[0]
    assert models == [runtime.model]
    assert settings == {'beam_size': 30, 'audio_only': False, 'video_only': True, 'temp': 1.2}
    assert [tensor.shape[2] for tensor in torch.tensors] == [frame_count, frame_count]
    assert torch.tensors[0].array is face and torch.tensors[1].array is lips
    assert all(tensor.transfer == ('cuda', 'bf16') for tensor in torch.tensors)
    assert sample['net_input']['padding_mask'] == ('zeros', ((1, frame_count),), {'dtype': 'bool', 'device': 'cuda'})
    assert not runtime._lock.locked()


@pytest.mark.parametrize('method', ['inspect_activity', 'decode_utterance', 'refine_utterance'])
def test_activity_and_utterance_paths_require_loaded_model(monkeypatch, method):
    runtime = FlareModel()
    monkeypatch.setattr(runtime, 'status', lambda: {'detail': 'Model is unavailable.'})
    with pytest.raises(ModelUnavailable, match='Model is unavailable'):
        getattr(runtime, method)([object()])

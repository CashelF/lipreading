"""Unsupported proposals must not create captions or erase supported drafts."""
import asyncio
from collections import deque
from contextlib import contextmanager
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from backend import grounding
from backend.model import FlareModel
from backend.preprocessing import Preprocessed
from test_utterance_model import TorchDouble
from test_utterance_stream import Scenario, UtteranceModel


@pytest.mark.parametrize('accepted,replacement', [(False, False), (True, False), (True, True)])
def test_model_checks_exact_search_candidate_under_model_lock(monkeypatch, accepted, replacement):
    torch = TorchDouble()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    runtime = FlareModel()
    runtime.model = SimpleNamespace(llm_tokenizer=SimpleNamespace(decode=lambda *args, **kwargs: 'verified alternative words'))
    array = np.zeros((1, 1, 51, 88, 88), dtype=np.float32)
    runtime.preprocessor = lambda *args, **kwargs: Preprocessed(array, array, {})
    calls, candidate = [], object()
    def search(models, sample, **kwargs):
        assert runtime._lock.locked()
        calls.append((sample, kwargs))
        return ['test proposed words'], []
    runtime._search = search
    @contextmanager
    def capture(actual):
        assert actual is search
        yield [candidate]
    def verify(model, sample, actual_candidate):
        assert runtime._lock.locked()
        assert model is runtime.model and actual_candidate is candidate
        assert sample is calls[0][0]
        return {'accepted': accepted, 'detail': 'test evidence result',
                **({'_replacement_tokens': SimpleNamespace(tolist=lambda: [7])} if replacement else {})}
    monkeypatch.setattr(grounding, 'capture_candidate', capture)
    monkeypatch.setattr(grounding, 'verify_candidate', verify)
    result = runtime.decode_utterance([object()] * 51)
    assert len(calls) == 1 and calls[0][1]['beam_size'] == 30
    assert result['text'] == ('verified alternative words' if replacement else 'test proposed words' if accepted else '')
    assert result['recognition']['accepted'] is accepted
    assert '_replacement_tokens' not in result['recognition']
    assert not runtime._lock.locked()


class VerifiedModel(UtteranceModel):
    supports_visual_grounding = True
    def __init__(self, accepted):
        super().__init__()
        self.accepted = iter(accepted)
        self.verify_options = []
    def _recognized(self, kind, frames, fps, cache, verify_visual):
        self.verify_options.append(verify_visual)
        result = self._decode(kind, frames, fps, cache, deque(['supported words']))
        accepted = next(self.accepted)
        if verify_visual:
            result['recognition'] = {'accepted': accepted, 'detail': 'Visual evidence was insufficient.'}
            if not accepted:
                result['text'] = ''
        return result
    def decode_utterance(self, frames, fps=25, *, cache=None, verify_visual=True):
        return self._recognized('preview', frames, fps, cache, verify_visual)
    def refine_utterance(self, frames, fps=25, *, cache=None, verify_visual=True):
        return self._recognized('final', frames, fps, cache, verify_visual)


def test_rejected_preview_and_final_never_display_an_unsupported_phrase():
    async def run():
        model = VerifiedModel([False, False])
        scenario = Scenario(model)
        first = await scenario.advance(3000.)
        final = await scenario.advance(4000., 'still')
        assert first['snapshot']['text'] == final['snapshot']['text'] == ''
        assert final['timings_ms']['grounding_withheld'] == 1
        assert 'insufficient' in final['detail']
    asyncio.run(run())


def test_rejected_revision_keeps_previous_supported_words():
    async def run():
        model = VerifiedModel([True, False, False])
        scenario = Scenario(model)
        first = await scenario.advance(3000.)
        revised = await scenario.advance(4000.)
        final = await scenario.advance(5000., 'still')
        assert first['snapshot']['text'] == revised['snapshot']['text'] == final['snapshot']['text'] == 'supported words'
        assert final['snapshot']['stable_text'] == 'supported words'
        assert final['snapshot']['draft_text'] == ''
    asyncio.run(run())


def test_verification_can_be_disabled_without_disabling_speech_filter():
    async def run():
        model = VerifiedModel([False])
        scenario = Scenario(model)
        scenario.reader.verify_visual = False
        result = await scenario.advance(3000.)
        assert model.verify_options == [False]
        assert result['snapshot']['text'] == 'supported words'
    asyncio.run(run())

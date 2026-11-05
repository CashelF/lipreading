"""Identical-source reuse must never skip new evidence or a changed policy."""
import asyncio

from backend.stream import Frame
from backend.utterance_stream import UtteranceReader
from test_utterance_stream import jpeg


class Model:
    supports_exact_result_reuse = True
    supports_visual_grounding = True
    beam = 30

    def __init__(self):
        self.calls = []

    def decode_utterance(self, frames, fps, *, cache, verify_visual):
        self.calls.append((len(frames), verify_visual))
        return {'text': 'same source', 'recognition': {'accepted': True, 'nested': {'score': .3}},
                'timings_ms': {'inference': 200.}, 'latency_ms': 200}

    refine_utterance = decode_utterance


def test_final_reuses_only_identical_source_and_policy_with_owned_results():
    async def run():
        model = Model()
        reader = UtteranceReader(model)
        image = jpeg()
        for timestamp in range(0, 2041, 40):
            reader.history.append(Frame(timestamp, image))
        clip = reader.history.slice(0, 2000, 0)
        async def send(message):
            pass
        first = await reader._decode(0, 2000, clip, False, send)
        first['recognition']['nested']['score'] = 999
        final = await reader._decode(0, 2000, clip, True, send)
        assert final['text'] == 'same source'
        assert final['recognition']['nested']['score'] == .3
        assert final['timings_ms']['recognition_reused'] == 1
        assert model.calls == [(51, True)]
        # A single new camera sample must invalidate, even with identical JPEGs.
        await reader._decode(0, 2040, clip, False, send)
        assert model.calls[-1] == (52, True)
        reader.verify_visual = False
        await reader._decode(0, 2040, clip, True, send)
        assert model.calls[-1] == (52, False)
        assert len(model.calls) == 3
        reader.clear()
        assert reader._last_decode_key is reader._last_decode_result is None
    asyncio.run(run())


def test_models_with_different_preview_and_final_semantics_are_not_reused():
    async def run():
        model = Model()
        model.supports_exact_result_reuse = False
        reader = UtteranceReader(model)
        for timestamp in range(0, 2001, 40):
            reader.history.append(Frame(timestamp, jpeg()))
        clip = reader.history.slice(0, 2000, 0)
        async def send(message):
            pass
        await reader._decode(0, 2000, clip, False, send)
        await reader._decode(0, 2000, clip, True, send)
        assert len(model.calls) == 2
    asyncio.run(run())

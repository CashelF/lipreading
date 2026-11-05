"""New utterances require confirmed speech without hardening continuation."""
import asyncio

import numpy as np
import pytest

from backend.speech_gate import SpeechGateState
from test_utterance_stream import Scenario, UtteranceModel, activity


def onset(end, *, confirmed=False, candidate=None, first=None, last=None, flags=None):
    observation = activity('motion', end, first=first, last=last)
    observation['metrics'].update(
        speech_start_threshold=.8,
        speech_start_confirm_frames=3,
        speech_start_confirmed=confirmed,
        speech_max_mean=.91 if confirmed else .604,
        speech_latest_mean=.91 if confirmed else .604,
    )
    if candidate is not None:
        observation['metrics']['speech_start_candidate_ms'] = candidate
    if confirmed:
        observation['metrics']['speech_start_confirmed_ms'] = end - 480.
    observation['metrics'].update(flags or {})
    return observation


def assert_waiting(result, original_metrics):
    assert result['activity']['state'] == 'uncertain'
    assert result['activity']['suppress']
    assert result['activity']['preserve_transcript']
    assert 'speech' in result['activity']['detail'].lower()
    for key, value in original_metrics.items():
        assert result['activity']['metrics'][key] == value


def test_weak_unconfirmed_onset_never_creates_a_draft_or_refinement_even_on_stop():
    async def run():
        model = UtteranceModel()
        scenario = Scenario(model)
        for end, final in ((3000., False), (4000., False), (4000., True)):
            inspection = onset(end, candidate=2440.)
            result = await scenario.advance(end, final=final, inspection=inspection)
            assert_waiting(result, inspection['metrics'])
            assert result['snapshot']['text'] == ''
            assert result['snapshot']['draft_text'] == ''
            assert result['timings_ms']['refinement_passes'] == 0
            assert scenario.reader.start_ms is None
            assert not scenario.reader.transcript.active
        assert result['snapshot']['final']
        assert model.calls == []
        assert scenario.messages == []
    asyncio.run(run())


def test_weak_onset_after_completed_speech_preserves_its_caption():
    async def run():
        model = UtteranceModel(previews=['provisional words'], finals=['completed words'])
        scenario = Scenario(model)
        await scenario.advance(3000., inspection=onset(3000., confirmed=True, candidate=2440.))
        complete = await scenario.advance(4000., 'still')
        assert complete['snapshot']['stable_text'] == 'completed words'
        for end, final in ((5000., False), (6000., False), (6000., True)):
            inspection = onset(end, candidate=4520., first=end - 800.)
            held = await scenario.advance(end, final=final, inspection=inspection)
            assert_waiting(held, inspection['metrics'])
            assert held['snapshot']['text'] == 'completed words'
            assert held['snapshot']['draft_text'] == ''
            assert held['timings_ms']['refinement_passes'] == 0
        assert [call['kind'] for call in model.calls] == ['preview', 'final']
    asyncio.run(run())


def test_confirmed_start_keeps_context_from_the_original_weak_crossing():
    async def run():
        model = UtteranceModel()
        scenario = Scenario(model)
        await scenario.advance(4000., inspection=onset(4000., candidate=2800., first=2800.))
        assert model.calls == []
        confirmed = onset(6000., confirmed=True, candidate=2800., first=4800.)
        result = await scenario.advance(6000., inspection=confirmed)
        assert scenario.reader.start_ms == 1200.  # candidate - 1.6 s, not confirmation - 1.6 s
        assert len(model.calls) == 1
        assert model.calls[0]['frames'] == 121
        assert result['snapshot']['draft_text']
        assert not result['activity']['suppress']
    asyncio.run(run())


def test_confirmed_next_start_cannot_reuse_the_previous_completed_utterance():
    async def run():
        model = UtteranceModel(previews=['first draft', 'second draft'], finals=['first final'])
        scenario = Scenario(model)
        await scenario.advance(3000., inspection=onset(3000., confirmed=True, candidate=2440.))
        await scenario.advance(4000., 'still')
        previous_end = scenario.reader.committed_end_ms
        second = await scenario.advance(6000., inspection=onset(
            6000., confirmed=True, candidate=4200., first=5200.))
        assert previous_end > 4200. - 1600.
        assert scenario.reader.start_ms == previous_end
        assert second['snapshot']['stable_text'] == 'first final'
        assert second['snapshot']['draft_text'] == 'second draft'
    asyncio.run(run())


@pytest.mark.parametrize('flag', ['speech_invalid', 'speech_gap'])
def test_invalid_or_interrupted_confirmed_observation_cannot_open_speech(flag):
    async def run():
        model = UtteranceModel()
        scenario = Scenario(model)
        # Guard the controller even if an upstream observation retains a
        # confirmation flag alongside invalid/discontinuous evidence.
        inspection = onset(3000., confirmed=True, candidate=2440., flags={flag: 1})
        result = await scenario.advance(3000., inspection=inspection)
        assert scenario.reader.start_ms is None
        assert result['snapshot']['text'] == ''
        assert model.calls == []
    asyncio.run(run())


def test_disabling_filter_bypasses_start_confirmation():
    async def run():
        model = UtteranceModel()
        scenario = Scenario(model)
        result = await scenario.advance(3000., speech_gate=False, inspection=onset(3000.))
        assert [call['kind'] for call in model.calls] == ['preview']
        assert scenario.reader.start_ms == scenario.origin
        assert result['snapshot']['draft_text']
    asyncio.run(run())


def test_legacy_activity_without_confirmation_metrics_remains_supported():
    async def run():
        model = UtteranceModel()
        scenario = Scenario(model)
        result = await scenario.advance(3000., inspection=activity('motion', 3000.))
        assert [call['kind'] for call in model.calls] == ['preview']
        assert result['snapshot']['draft_text']
    asyncio.run(run())


def test_real_gate_confirms_then_keeps_subtle_continuation_and_rejects_new_weak_start():
    async def run():
        model = UtteranceModel(previews=['first words', 'more words', 'subtle continuation'],
                               finals=['completed subtle speech'])
        scenario = Scenario(model)
        gate = SpeechGateState()

        async def scores(end, value, *, final=False):
            start = max(scenario.origin, end - 6000.)
            timestamps = np.arange(start, end + 1., 40.)
            inspection = gate.update(np.full(len(timestamps), value), timestamps)
            result = await scenario.advance(end, final=final, inspection=inspection)
            return inspection, result

        weak, held = await scores(3000., .5)
        assert weak['state'] == 'motion'  # The unchanged continuation classifier admits it.
        assert not weak['metrics']['speech_start_confirmed']
        assert_waiting(held, weak['metrics'])
        assert model.calls == []

        strong, opened = await scores(4000., .95)
        assert strong['metrics']['speech_start_confirmed']
        assert strong['metrics']['speech_start_candidate_ms'] <= weak['metrics']['speech_first_active_ms']
        assert opened['snapshot']['draft_text'] == 'first words'

        await scores(6000., .604)  # Replace all strong score history.
        subtle, continued = await scores(7000., .604)
        assert subtle['state'] == 'motion'
        assert subtle['metrics']['speech_latest_mean'] == pytest.approx(.604)
        assert not subtle['metrics']['speech_start_confirmed']
        assert not continued['activity']['suppress']
        assert continued['snapshot']['draft_text'] == 'subtle continuation'

        quiet, completed = await scores(9000., .05)
        assert completed['timings_ms']['refinement_passes'] == 1
        assert completed['snapshot']['stable_text'] == 'completed subtle speech'
        assert completed['snapshot']['draft_text'] == ''

        new_weak, rejected = await scores(10000., .604)
        assert new_weak['state'] == 'motion'
        assert not new_weak['metrics']['speech_start_confirmed']
        assert_waiting(rejected, new_weak['metrics'])
        assert rejected['snapshot']['text'] == 'completed subtle speech'
        _, stopped = await scores(10000., .604, final=True)
        assert stopped['snapshot']['final']
        assert stopped['snapshot']['text'] == 'completed subtle speech'
        assert [call['kind'] for call in model.calls] == ['preview', 'preview', 'preview', 'final']
    asyncio.run(run())

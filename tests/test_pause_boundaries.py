"""Confirmed pauses must survive coalescing and cannot license weak new speech."""
import asyncio

import numpy as np
import pytest

from backend.speech_gate import SpeechGateState
from backend.stream import Frame
from test_utterance_stream import Scenario, UtteranceModel


def observed(gate, end, second=.6, *, start=0.):
    times = np.arange(start, end + 1., 40.)
    # Two speaking-shaped regions separated by enough observed quiet to close A.
    scores = np.where(times < 2000., .95,
                      np.where((times >= 4000.) & (times < 6000.), second, .02))
    return gate.update(scores, times)


async def replay(ends, second=.6, *, stop=False):
    model = UtteranceModel(finals=['first completed phrase', 'second completed phrase'])
    scenario = Scenario(model, origin=0.)
    gate = SpeechGateState()
    results = []
    for index, end in enumerate(ends):
        result = await scenario.advance(end, final=stop and index == len(ends) - 1,
                                        inspection=observed(gate, end, second))
        results.append(result)
    return scenario, gate, results


@pytest.mark.parametrize('stop', [False, True])
def test_coalesced_pause_closes_a_before_unconfirmed_weak_b_including_stop(stop):
    async def run():
        scenario, gate, results = await replay([2000., 8000.], stop=stop)
        result = results[-1]
        assert [call['kind'] for call in scenario.model.calls] == ['preview', 'final']
        assert result['timings_ms']['refinement_passes'] == 1
        assert result['snapshot']['stable_text'] == 'first completed phrase'
        assert not result['snapshot']['draft_text']
        interval = result['decoded_intervals'][0]
        assert interval['end_ms'] == interval['last_active_ms'] + 500.
        assert interval['end_ms'] < 4000.  # weak B begins at 4 s
        assert scenario.reader.start_ms is None
        # Neither duplicate score evidence nor repeated Stop can reuse A.
        again = await scenario.advance(8000., final=True, inspection=observed(gate, 8000.))
        assert again['timings_ms']['refinement_passes'] == 0
        assert again['decoded_intervals'] == []
        assert len(scenario.model.calls) == 2
    asyncio.run(run())


@pytest.mark.parametrize('second, expected', [(.6, 1), (.95, 2)])
def test_sequential_and_single_coalesced_windows_have_identical_final_intervals(second, expected):
    async def run():
        seq, _, seq_results = await replay([2000., 3000., 4000., 5000., 6000., 7000., 8000.],
                                           second, stop=True)
        merged, _, merged_results = await replay([8000.], second, stop=True)
        def finals(results):
            return [interval for result in results for interval in result['decoded_intervals']
                    if interval['final']]
        assert len(finals(seq_results)) == len(finals(merged_results)) == expected
        assert finals(seq_results) == finals(merged_results)
        assert seq_results[-1]['snapshot']['text'] == merged_results[-1]['snapshot']['text']
        assert merged_results[-1]['timings_ms']['refinement_passes'] == expected
        if expected == 2:
            first, second_interval = finals(merged_results)
            assert second_interval['start_ms'] >= first['end_ms']
    asyncio.run(run())


def test_coalesced_completed_a_and_ongoing_confirmed_b_keeps_b_as_separate_draft():
    async def run():
        scenario, _, results = await replay([2000., 6000.], .95)
        result = results[-1]
        assert [call['kind'] for call in scenario.model.calls] == ['preview', 'final', 'preview']
        assert result['snapshot']['stable_text'] == 'first completed phrase'
        assert result['snapshot']['draft_text']
        first, second = result['decoded_intervals']
        assert first['final'] and not second['final']
        assert second['start_ms'] >= first['end_ms']
    asyncio.run(run())


def test_invalid_stop_preserves_active_draft_without_decoding_uncertain_tail():
    async def run():
        scenario, gate, results = await replay([2000.])
        before = results[-1]['snapshot']['text']
        invalid = gate.abstain()
        held = await scenario.advance(4000., final=True, inspection=invalid)
        assert held['snapshot']['text'] == before
        assert held['timings_ms']['refinement_passes'] == 0
        assert len(scenario.model.calls) == 1
        assert 'without final refinement' in held['detail']
    asyncio.run(run())


def test_quiet_recovery_does_not_trim_unknown_speech_to_a_pre_interruption_endpoint():
    async def run():
        scenario, gate, results = await replay([2000.])
        before = results[-1]['snapshot']['text']
        await scenario.advance(4000., inspection=gate.abstain())
        gap = gate.update(np.full(151, .02), np.arange(2000., 8001., 40.))
        assert gap['metrics']['speech_gap']
        await scenario.advance(8000., inspection=gap)
        quiet = gate.update(np.full(151, .02), np.arange(3000., 9001., 40.))
        assert quiet['speech_regions'][0]['closed_at_ms'] is not None
        held = await scenario.advance(9000., inspection=quiet)
        assert held['snapshot']['stable_text'] == before
        assert held['snapshot']['draft_text'] == ''
        assert held['decoded_intervals'] == []
        assert held['timings_ms']['refinement_passes'] == 0
        assert held['timings_ms']['preserved_drafts'] == 1
        assert 'without final refinement' in held['detail']
        assert scenario.reader.committed_end_ms == quiet['speech_regions'][0]['last_active_ms'] + 500.
        assert len(scenario.model.calls) == 1
    asyncio.run(run())


def test_new_speech_after_interruption_retains_its_beginning_before_quiet_confirmation():
    async def run():
        scenario, gate, results = await replay([2000.])
        before = results[-1]['snapshot']['text']
        await scenario.advance(4000., inspection=gate.abstain())
        gap = gate.update(np.full(151, .02), np.arange(2000., 8001., 40.))
        await scenario.advance(8000., inspection=gap)
        times = np.arange(3000., 10001., 40.)
        # B physically starts at8s; smoothed A closure is8320ms and B's
        # low-threshold candidate is8400ms. Clamping B at8320ms loses its start.
        recovered = gate.update(np.where(times >= 8000., .95, .02), times)
        closed_a, open_b = recovered['speech_regions']
        assert closed_a['closed_at_ms'] == 8320.
        assert open_b['start_ms'] == 8400.
        result = await scenario.advance(10000., inspection=recovered)
        assert result['snapshot']['stable_text'] == before
        assert result['timings_ms']['refinement_passes'] == 0
        assert len(result['decoded_intervals']) == 1
        interval = result['decoded_intervals'][0]
        assert interval['start_ms'] == open_b['start_ms'] - 1600.
        assert interval['start_ms'] < 8000.
        assert interval['end_ms'] == 10000.
        assert not interval['final']
        assert 'without final refinement' in result['detail']
    asyncio.run(run())


def test_gap_requires_new_observed_quiet_and_does_not_erase_confirmed_region():
    gate = SpeechGateState()
    first = gate.update(np.full(51, .95), np.arange(0., 2001., 40.))
    candidate = first['speech_regions'][0]['start_ms']
    gap = gate.update(np.full(51, .02), np.arange(6000., 8001., 40.))
    assert gap['metrics']['speech_gap']
    assert gap['speech_regions'] == []
    # A half second of newly observed quiet cannot include the missing interval.
    short = gate.update(np.full(64, .02), np.arange(6000., 8521., 40.))
    assert short['speech_regions'][0]['start_ms'] == candidate
    assert short['speech_regions'][0]['closed_at_ms'] is None
    enough = gate.update(np.full(76, .02), np.arange(6000., 9001., 40.))
    assert enough['speech_regions'][0]['start_ms'] == candidate
    assert enough['speech_regions'][0]['closed_at_ms'] is not None


def test_confirmed_low_score_continuation_recovers_after_invalid_observation():
    gate = SpeechGateState()
    initial = gate.update(np.full(51, .95), np.arange(0., 2001., 40.))
    candidate = initial['speech_regions'][0]['start_ms']
    assert gate.abstain()['speech_regions'] == []
    recovered = gate.update(np.full(101, .6), np.arange(0., 4001., 40.))
    assert not recovered['metrics']['speech_start_confirmed']
    assert recovered['speech_regions'][0]['start_ms'] == candidate
    assert recovered['speech_regions'][0]['fresh_active']
    assert recovered['speech_regions'][0]['closed_at_ms'] is None


def test_active_speech_preserves_irregular_capture_endpoint_and_last_sample():
    async def run():
        model = UtteranceModel()
        scenario = Scenario(model, origin=0.)
        for timestamp in np.arange(0., 3196., 45.):
            scenario.reader.history.append(Frame(timestamp, scenario.image))
        clip = scenario.reader.history.slice(0., 3195., 0)
        times = np.arange(0., 3161., 40.)
        gate = SpeechGateState()
        model.inspections.append(gate.update(np.full(len(times), .95), times))
        result = await scenario.reader.process(clip, True, scenario.send)
        assert result['decoded_intervals'][0]['end_ms'] == 3195.
        assert model.calls[0]['frames'] == len(times)
    asyncio.run(run())


def test_preview_with_observed_short_trailing_quiet_caps_extra_capture_tail():
    async def run():
        model = UtteranceModel()
        scenario = Scenario(model, origin=0.)
        gate = SpeechGateState()
        times = np.arange(0., 3401., 40.)
        scores = np.where(times < 2000., .95, .02)
        inspection = gate.update(scores, times)
        region = inspection['speech_regions'][0]
        assert region['closed_at_ms'] is None
        assert region['last_active_ms'] < inspection['metrics']['speech_evidence_end_ms']
        result = await scenario.advance(3400., inspection=inspection)
        assert not result['decoded_intervals'][0]['final']
        assert result['decoded_intervals'][0]['end_ms'] == region['last_active_ms'] + 500.
    asyncio.run(run())


def test_failed_closed_region_refinement_is_terminal_before_next_phrase_can_attach():
    async def run():
        scenario, gate, results = await replay([2000.], .95)
        before = results[-1]['snapshot']['text']
        def failed_final(*args, **kwargs):
            raise ValueError('Controlled preprocessing quality failure')
        scenario.model.refine_utterance = failed_final
        # Both A and confirmed B end in this coalesced update. Failure to
        # finalize A must stop before B could overwrite its unfinished draft.
        with pytest.raises(RuntimeError, match='Final refinement failed.*draft was kept'):
            await scenario.advance(8000., inspection=observed(gate, 8000., .95))
        assert scenario.reader.transcript.snapshot()['text'] == before
        assert len(scenario.model.calls) == 1
        assert scenario.reader.transcript.finish()['text'] == before
    asyncio.run(run())

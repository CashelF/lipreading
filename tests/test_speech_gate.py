"""Temporal contracts of the calibrated gate, without camera data or weights."""
import numpy as np
import pytest

from backend.speech_gate import (CONTEXT_FRAMES, MEAN_FRAMES, START_CONFIRM_FRAMES,
                                START_SCORE_THRESHOLD, SpeechGateState)


def window(start=0., end=2000., score=.1, phase=.0):
    times = np.arange(start, end + 1, 40.) + phase
    return np.full(len(times), score), times


def assert_suppressed(result, state):
    assert result['state'] == state
    assert result['suppress'] is True
    assert result['preserve_transcript'] is True


def test_quiet_scores_suppress_without_retracting_previous_words():
    gate = SpeechGateState()
    result = gate.update(*window(score=.15))
    assert_suppressed(result, 'still')
    assert result['metrics']['speech_new_frames'] == 27
    assert result['metrics']['speech_mean_count'] == 3
    assert len(gate.history) == 24


@pytest.mark.parametrize('score,expected', [(.39999, 'still'), (.4, 'motion'), (.8, 'motion')])
def test_frozen_threshold_and_mean(score, expected):
    result = SpeechGateState().update(*window(score=score))
    assert result['state'] == expected
    assert result['suppress'] is (expected != 'motion')
    assert result['preserve_transcript'] is (expected != 'motion')


def test_one_high_score_does_not_make_quiet_history_speech():
    scores, times = window(score=.1)
    scores[25] = 1.
    assert_suppressed(SpeechGateState().update(scores, times), 'still')


def test_insufficient_context_can_accumulate_across_updates():
    gate = SpeechGateState()
    scores, times = window(end=1880., score=.8)  # 48 frames; 24 mature.
    assert_suppressed(gate.update(scores, times), 'uncertain')
    result = gate.update(*window(end=1920., score=.8))
    assert result['state'] == 'motion'
    assert result['metrics']['speech_new_frames'] == 1
    assert result['metrics']['speech_mean_count'] == 1


def test_short_final_window_abstains_without_invented_padding():
    gate = SpeechGateState()
    result = gate.update(*window(end=920., score=.99))
    assert_suppressed(result, 'uncertain')
    assert gate.last_timestamp_ms is None


def test_identical_or_older_windows_never_replay_a_speech_event():
    gate = SpeechGateState()
    assert gate.update(*window(score=.9))['state'] == 'motion'
    watermark = gate.last_timestamp_ms
    for args in (window(score=.9), window(end=1920., score=.99)):
        result = gate.update(*args)
        assert_suppressed(result, 'still')
        assert result['metrics']['speech_new_frames'] == 0
        assert gate.last_timestamp_ms == watermark


def test_overlapping_history_does_not_readmit_old_event_after_quiet_means():
    gate = SpeechGateState()
    assert gate.update(*window(end=3000., score=.9))['state'] == 'motion'
    # Old speech legitimately affects the first few new means, but it must not
    # keep admitting the entire moving window after fresh quiet evidence.
    scores, times = window(end=4000., score=.05)
    scores[times <= 2520] = .9
    gate.update(scores, times)
    scores, times = window(start=1000., end=5000., score=.05)
    scores[times <= 2520] = .9
    result = gate.update(scores, times)
    assert_suppressed(result, 'still')
    assert result['metrics']['speech_new_frames'] == 25


def test_coalesced_updates_check_speech_in_middle_not_only_last_second():
    gate = SpeechGateState()
    assert_suppressed(gate.update(*window()), 'still')
    scores, times = window(end=5000., score=.05)
    scores[(times >= 2000) & (times <= 3000)] = .9
    result = gate.update(scores, times)
    assert result['state'] == 'motion'
    assert result['metrics']['speech_new_frames'] == 75
    assert result['metrics']['speech_latest_mean'] == .05
    assert result['metrics']['speech_max_mean'] == .9
    assert_suppressed(gate.update(*window(start=1000., end=6000., score=.05)), 'still')


def test_watermark_uses_absolute_sample_phase_and_only_new_mature_scores():
    gate = SpeechGateState()
    _, times = window(phase=103.25)
    gate.update(*window(phase=103.25))
    assert gate.last_timestamp_ms == times[-CONTEXT_FRAMES - 1]
    result = gate.update(*window(end=3000., phase=103.25))
    assert result['metrics']['speech_new_frames'] == 25
    assert gate.last_timestamp_ms == 2623.25


def test_reset_releases_history_and_does_not_leak_session_speech():
    gate = SpeechGateState()
    gate.update(*window(score=.9))
    gate.reset()
    assert gate.last_timestamp_ms is None
    assert not gate.history
    assert_suppressed(gate.update(*window(score=.05)), 'still')


def test_missing_face_abstention_clears_mean_history_but_not_watermark():
    gate = SpeechGateState()
    gate.update(*window(score=.9))
    watermark = gate.last_timestamp_ms
    result = gate.abstain('Keep one face visible.')
    assert_suppressed(result, 'uncertain')
    assert result['detail'] == 'Keep one face visible.'
    assert not gate.history
    assert gate.last_timestamp_ms == watermark
    assert_suppressed(gate.update(*window(score=.9)), 'still')
    assert gate.update(*window(end=3000., score=.9))['state'] == 'motion'


@pytest.mark.parametrize('bad', [np.nan, np.inf, -.01, 1.01])
def test_invalid_scores_abstain_and_cannot_replay_old_speech(bad):
    gate = SpeechGateState()
    gate.update(*window(score=.9))
    watermark = gate.last_timestamp_ms
    scores, times = window(end=3000., score=.9)
    scores[30] = bad
    result = gate.update(scores, times)
    assert_suppressed(result, 'uncertain')
    assert result['metrics']['speech_invalid'] == 1
    assert not gate.history
    assert gate.last_timestamp_ms == watermark
    assert_suppressed(gate.update(*window(score=.9)), 'still')


@pytest.mark.parametrize('scores,times', [([], []), ([.2], []), ([[.2]], [0]), (None, [0]), (['bad'], [0])])
def test_missing_or_malformed_inputs_never_admit(scores, times):
    assert_suppressed(SpeechGateState().update(scores, times), 'uncertain')


@pytest.mark.parametrize('change', ['nan', 'negative', 'backwards', 'gap', 'wrong_fps'])
def test_invalid_time_grid_abstains(change):
    scores, times = window()
    if change == 'nan':
        times[20] = np.nan
    elif change == 'negative':
        times[0] = -40
    elif change == 'backwards':
        times[20] = times[19]
    elif change == 'gap':
        times[20:] += 40
    else:
        times *= .5
    assert_suppressed(SpeechGateState().update(scores, times), 'uncertain')


def test_evidence_gap_is_explicit_and_next_continuous_update_can_recover():
    gate = SpeechGateState()
    gate.update(*window())
    # Last checked time1520; next maturefirst2480 leaves an unseeninterval.
    result = gate.update(*window(start=2000., end=4000., score=.9))
    assert_suppressed(result, 'uncertain')
    assert result['metrics']['speech_gap'] == 1
    assert gate.last_timestamp_ms == 3520
    result = gate.update(*window(start=2000., end=5000., score=.9))
    assert result['state'] == 'motion'
    assert result['metrics']['speech_gap'] == 0


def test_misaligned_sample_phase_is_reported_as_gap():
    gate = SpeechGateState()
    gate.update(*window())
    result = gate.update(*window(end=3000., phase=20.))
    assert_suppressed(result, 'uncertain')
    assert result['metrics']['speech_gap'] == 1


def test_histories_remain_bounded_during_long_sessions():
    gate = SpeechGateState()
    for end in range(2000, 102001, 1000):
        gate.update(*window(start=max(0, end - 6000), end=end))
    assert len(gate.history) == MEAN_FRAMES - 1
    assert gate.last_timestamp_ms == 101520.


@pytest.mark.parametrize('score,confirmed', [(.79999, 0), (.8, 1), (.80001, 1)])
def test_start_confirmation_has_separate_inclusive_threshold(score, confirmed):
    result = SpeechGateState().update(*window(score=score))
    assert result['state'] == 'motion'  # Original .4 gate is unchanged.
    metrics = result['metrics']
    assert metrics['speech_start_threshold'] == START_SCORE_THRESHOLD == .8
    assert metrics['speech_start_confirm_frames'] == START_CONFIRM_FRAMES == 3
    assert metrics['speech_start_confirmed'] == confirmed
    assert metrics['speech_start_candidate_ms'] == 1440.
    if confirmed:
        assert metrics['speech_start_confirmed_ms'] == 1520.
    else:
        assert 'speech_start_confirmed_ms' not in metrics


def test_continuous_weak_speech_keeps_original_activity_but_cannot_confirm_start():
    gate = SpeechGateState()
    for end in (2000., 3000., 6000.):
        result = gate.update(*window(end=end, score=.6))
        assert result['state'] == 'motion'
        assert result['suppress'] is False
        assert result['metrics']['speech_start_confirmed'] == 0
        assert result['metrics']['speech_start_candidate_ms'] == 1440.
        assert 'speech_start_confirmed_ms' not in result['metrics']


def test_confirmation_accumulates_across_calls_without_replaying_duplicates():
    gate = SpeechGateState()
    first = gate.update(*window(end=1960., score=.9))
    assert first['metrics']['speech_mean_count'] == 2
    assert first['metrics']['speech_start_confirmed'] == 0
    duplicate = gate.update(*window(end=1960., score=.9))
    assert duplicate['metrics']['speech_start_confirmed'] == 0
    assert 'speech_start_confirmed_ms' not in duplicate['metrics']
    confirmed = gate.update(*window(score=.9))
    assert confirmed['metrics']['speech_start_confirmed'] == 1
    assert confirmed['metrics']['speech_start_candidate_ms'] == 1440.
    assert confirmed['metrics']['speech_start_confirmed_ms'] == 1520.
    duplicate = gate.update(*window(score=.9))
    assert duplicate['metrics']['speech_start_confirmed'] == 0
    assert 'speech_start_confirmed_ms' not in duplicate['metrics']
    # A continuing strong run can confirm again, but only on fresh evidence.
    newer = gate.update(*window(end=2040., score=.9))
    assert newer['metrics']['speech_start_confirmed'] == 1
    assert newer['metrics']['speech_start_confirmed_ms'] == 1560.


def test_start_confirmation_requires_consecutive_strong_means():
    gate = SpeechGateState()
    gate.update(*window(end=1960., score=.8))  # Two strong means.
    for end in (2000., 3040.):
        scores, times = window(end=end, score=.8)
        scores[times == 1520.] = .79  # Breaks the run until this leaves the mean.
        result = gate.update(scores, times)
        assert result['state'] == 'motion'
        assert result['metrics']['speech_start_confirmed'] == 0
    scores, times = window(end=3080., score=.8)
    scores[times == 1520.] = .79
    result = gate.update(scores, times)
    assert result['metrics']['speech_start_confirmed'] == 1
    assert result['metrics']['speech_start_candidate_ms'] == 1440.
    assert result['metrics']['speech_start_confirmed_ms'] == 2600.


def test_candidate_survives_brief_quiet_but_resets_at_800ms():
    gate = SpeechGateState()
    gate.update(*window(score=.5))
    for end, pending in ((2960., True), (3000., False)):
        scores, times = window(end=end, score=0.)
        scores[times <= 1520.] = .5
        result = gate.update(scores, times)
        assert result['metrics']['speech_start_confirmed'] == 0
        assert ('speech_start_candidate_ms' in result['metrics']) is pending
        if pending:
            assert result['metrics']['speech_start_candidate_ms'] == 1440.


@pytest.mark.parametrize('last_cluster_score,expected_next_candidate', [(1., 7960.), (.5, 8360.)])
def test_coalesced_confirmation_uses_correct_onset_and_preserves_first_confirmed_cluster(
        last_cluster_score, expected_next_candidate):
    gate = SpeechGateState()

    def clusters(end):
        scores, times = window(end=end, score=0.)
        scores[times <= 2000.] = .5  # Early weak false start, then enough quiet.
        scores[(times >= 4200.) & (times <= 5600.)] = 1.
        scores[times >= 7600.] = last_cluster_score
        return scores, times

    result = gate.update(*clusters(10000.))
    metrics = result['metrics']
    assert metrics['speech_first_active_ms'] == 1440.  # Unchanged activity metric.
    assert metrics['speech_start_confirmed'] == 1
    assert metrics['speech_start_candidate_ms'] == 4560.  # First confirmed cluster.
    assert metrics['speech_start_confirmed_ms'] == 5040.
    newer = gate.update(*clusters(10040.))
    assert newer['metrics']['speech_start_candidate_ms'] == expected_next_candidate
    assert newer['metrics']['speech_start_confirmed'] == int(last_cluster_score == 1.)


@pytest.mark.parametrize('interruption', ['abstain', 'invalid', 'reset'])
def test_interruption_clears_pending_confirmation(interruption):
    gate = SpeechGateState()
    gate.update(*window(end=1960., score=.9))
    if interruption == 'abstain':
        result = gate.abstain()
    elif interruption == 'invalid':
        scores, times = window(score=.9)
        scores[20] = np.nan
        result = gate.update(scores, times)
    else:
        gate.reset()
        result = gate.update(*window(end=1960., score=.9))
        assert result['metrics']['speech_start_confirmed'] == 0
        return
    assert result['metrics']['speech_start_confirmed'] == 0
    assert 'speech_start_candidate_ms' not in result['metrics']
    assert 'speech_start_confirmed_ms' not in result['metrics']
    result = gate.update(*window(end=3000., score=.9))
    assert result['metrics']['speech_mean_count'] == 2
    assert result['metrics']['speech_start_confirmed'] == 0
    assert result['metrics']['speech_start_candidate_ms'] == 2480.
    result = gate.update(*window(end=3040., score=.9))
    assert result['metrics']['speech_start_confirmed'] == 1


@pytest.mark.parametrize('gap_end', [5960., 6000.])
def test_gap_cannot_confirm_and_recovery_uses_only_new_cluster(gap_end):
    gate = SpeechGateState()
    gate.update(*window(end=1960., score=.9))
    result = gate.update(*window(start=4000., end=gap_end, score=.9))
    assert_suppressed(result, 'uncertain')
    assert result['metrics']['speech_gap'] == 1
    assert result['metrics']['speech_start_confirmed'] == 0
    assert 'speech_start_confirmed_ms' not in result['metrics']
    assert result['metrics']['speech_start_candidate_ms'] == 5440.
    result = gate.update(*window(start=4000., end=gap_end + 40., score=.9))
    assert result['metrics']['speech_start_confirmed'] == 1
    assert result['metrics']['speech_start_candidate_ms'] == 5440.
    assert result['metrics']['speech_start_confirmed_ms'] == gap_end - 440.

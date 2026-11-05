"""Heuristic gate tests: no trained accuracy or probability is implied."""
import json

import numpy as np
import pytest

from backend.activity import DEFAULT_THRESHOLDS, analyze_activity


def face():
    theta = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    result = np.stack([128 + 80 * np.cos(theta), 128 + 90 * np.sin(theta)], axis=1)
    result[[33, 36, 39, 42, 45]] = [[128, 155], [68, 100], [98, 100], [158, 100], [188, 100]]
    outer = np.array([[-.4, .95], [-.3, .88], [-.15, .84], [0, .85], [.15, .84], [.3, .88],
                      [.4, .95], [.3, 1.02], [.15, 1.07], [0, 1.08], [-.15, 1.07], [-.3, 1.02]])
    inner = np.array([[-.26, .95], [-.14, .925], [0, .92], [.14, .925],
                      [.26, .95], [.14, .98], [0, .985], [-.14, .98]])
    result[48:68] = np.vstack((outer, inner)) * 100 + [128, 100]
    return result


def capture(count=151):
    return np.repeat(face()[None], count, axis=0)


def classify(track, observed=None):
    return analyze_activity(track, np.ones(len(track), dtype=bool) if observed is None else observed)


def articulate(track, amplitude):
    phase = np.sin(np.arange(len(track)) * 2 * np.pi / 13)
    track[:, [55, 56, 57, 58, 59, 65, 66, 67], 1] += phase[:, None] * amplitude
    track[:, [49, 50, 51, 52, 53, 61, 62, 63], 1] -= phase[:, None] * amplitude * .3
    return track


def rigid_motion(track):
    result = []
    for index, points in enumerate(track):
        angle = .6 * np.sin(index / 11)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        scale = 1 + .3 * np.sin(index / 17)
        result.append((points - [128, 100]) @ rotation * scale + [150 + index * .8, 80 - index * .3])
    return np.array(result)


def test_stationary_closed_or_open_mouth_is_still():
    for opening in (0, 20):
        track = capture()
        track[:, [65, 66, 67], 1] += opening
        result = classify(track)
        assert result['state'] == 'still'
        assert result['suppress'] is True
        assert result['metrics']['shape_motion'] == 0
        assert result['detail'].startswith('Waiting for speech')


def test_rigid_translation_rotation_and_scale_do_not_become_speech():
    result = classify(rigid_motion(capture()))
    assert result['state'] == 'still'
    assert result['metrics']['anchor_residual'] == 0
    assert result['metrics']['shape_motion'] < 1e-6


@pytest.mark.parametrize('amplitude', [.35, .7, 3.])
def test_subtle_and_normal_articulation_pass(amplitude):
    result = classify(articulate(capture(), amplitude))
    assert result['suppress'] is False
    assert result['state'] in ('uncertain', 'motion')


def test_motion_class_does_not_claim_to_be_probability_of_speaking():
    result = classify(articulate(capture(), 3))
    assert result['state'] == 'motion'
    assert 'confidence' not in result
    assert 'probability' not in result['metrics']


def test_genuine_articulation_survives_head_motion_compensation():
    track = articulate(capture(), .35)
    assert classify(rigid_motion(track))['suppress'] is False


def test_early_brief_motion_is_not_diluted_by_appending_silence():
    track = capture()
    track[7:11, [65, 66, 67], 1] += .7
    values = [classify(track[:count]) for count in (51, 101, 151)]
    assert all(not result['suppress'] for result in values)
    assert len({result['metrics']['opening_motion'] for result in values}) == 1


def test_single_frame_movement_is_not_erased_by_median_filter():
    track = capture(51)
    track[20, [65, 66, 67], 1] += 1.
    result = classify(track)
    assert result['metrics']['opening_motion'] == 0
    assert result['metrics']['raw_opening_step'] > DEFAULT_THRESHOLDS.still_raw_opening_step
    assert result['suppress'] is False


def test_slow_accumulated_motion_passes_even_when_each_window_is_quiet():
    track = capture()
    track[:, [65, 66, 67], 1] += np.linspace(0, 1, len(track))[:, None]
    result = classify(track)
    assert result['metrics']['opening_motion'] < DEFAULT_THRESHOLDS.still_opening
    assert result['metrics']['global_opening_motion'] > DEFAULT_THRESHOLDS.still_global_opening
    assert result['suppress'] is False


def test_mouth_translation_relative_to_head_is_not_removed():
    track = capture()
    track[:, 48:68, 1] += np.sin(np.arange(len(track)) / 4)[:, None]
    assert classify(track)['suppress'] is False


def test_small_independent_tracker_noise_can_still_be_rejected():
    rng = np.random.default_rng(2026)
    track = capture() + rng.normal(0, .025, (151, 68, 2))
    assert classify(track)['state'] == 'still'


def test_large_nonrigid_anchor_jitter_is_uncertain_not_silence():
    track = capture()
    track[40:50, 33, 1] += 12
    result = classify(track)
    assert result['state'] == 'uncertain'
    assert result['suppress'] is False
    assert 'tracking' in result['detail'] or 'pose' in result['detail']


def test_missing_detection_cannot_be_replaced_by_interpolated_stillness():
    observed = np.ones(151, dtype=bool)
    observed[20] = False
    result = classify(capture(), observed)
    assert result['state'] == 'uncertain'
    assert result['suppress'] is False
    assert result['metrics']['tracked_fraction'] < 1


@pytest.mark.parametrize('change', ['nose_offset', 'eye_asymmetry', 'collapsed'])
def test_unreliable_pose_or_geometry_fails_open(change):
    track = capture()
    if change == 'nose_offset':
        track[:, 33, 0] += 80
    elif change == 'eye_asymmetry':
        track[:, 39] = track[:, 36] + [2, 0]
    else:
        track[:, [33, 36, 39, 42, 45]] = [128, 100]
    result = classify(track)
    assert result['state'] == 'uncertain'
    assert result['suppress'] is False


@pytest.mark.parametrize('fps', [0, -1, np.nan, np.inf])
def test_bad_timing_is_uncertain(fps):
    result = analyze_activity(capture(), np.ones(151, dtype=bool), fps)
    assert result['suppress'] is False
    assert result['state'] == 'uncertain'


@pytest.mark.parametrize('count', [0, 1, 19])
def test_short_or_empty_history_is_uncertain(count):
    assert classify(capture(count))['suppress'] is False


def test_bad_shape_or_nonfinite_geometry_is_uncertain():
    assert analyze_activity(np.zeros((25, 67, 2)), np.ones(25))['suppress'] is False
    assert analyze_activity(capture(25), np.ones(24))['suppress'] is False
    track = capture(25)
    track[0, 48, 0] = np.nan
    result = classify(track)
    assert result['suppress'] is False
    json.dumps(result, allow_nan=False)


def test_analyzer_does_not_mutate_input_and_metrics_are_json_safe():
    track = rigid_motion(articulate(capture(), 3))
    original = track.copy()
    track.setflags(write=False)
    result = classify(track)
    np.testing.assert_array_equal(track, original)
    json.dumps(result, allow_nan=False)
    assert result['metrics']['window_frames'] == 13
    assert result['metrics']['still_shape_threshold'] == DEFAULT_THRESHOLDS.still_shape

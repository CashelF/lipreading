"""Positive fallback evidence; absence of evidence is not a silence decision."""
import json

import numpy as np
import pytest

from backend.articulation import detect_articulation
from test_activity import capture, articulate, rigid_motion


def detect(points, observed=None):
    return detect_articulation(points, np.ones(len(points), dtype=bool) if observed is None else observed)


def test_clear_opening_and_closing_passes_without_modifying_landmarks():
    points = articulate(capture(51), 3.)
    before = points.copy()
    result = detect(points)
    assert result['active']
    assert result['metrics']['articulation_opening'] >= .04
    np.testing.assert_array_equal(points, before)
    json.dumps(result)


def test_clear_articulation_survives_rigid_head_movement():
    assert detect(rigid_motion(articulate(capture(51), 3.)))['active']


@pytest.mark.parametrize('opening', [0., 20.])
def test_stationary_open_or_closed_mouth_is_not_articulation(opening):
    points = capture()
    points[:, [65, 66, 67], 1] += opening
    assert not detect(points)['active']
    assert not detect(rigid_motion(points))['active']


def test_three_frame_burst_is_retained_when_silence_is_appended():
    points = capture()
    points[7:10, [65, 66, 67], 1] += 6
    assert all(detect(points[:count])['active'] for count in (20, 51, 101, 151))


@pytest.mark.parametrize('length', [1, 2])
def test_isolated_coherent_detector_spikes_cannot_admit_speech(length):
    points = capture()
    points[10:10 + length, [65, 66, 67], 1] += 20
    assert not detect(points)['active']


def test_one_bad_lip_point_is_not_coherent_articulation():
    points = capture()
    points[7:15, 66, 1] += 20
    assert not detect(points)['active']


def test_oppositely_moving_gaps_are_not_combined_into_evidence():
    points = capture()
    points[7:15, 65, 1] += 10
    points[7:15, 67, 1] -= 10
    assert not detect(points)['active']


def test_small_random_landmark_jitter_does_not_admit_speech():
    rng = np.random.default_rng(241)
    assert not detect(capture() + rng.normal(0, .35, (151, 68, 2)))['active']


def test_small_motion_does_not_claim_clear_articulation():
    # Small movements remain eligible through reliable image registration; this
    # fallback should not overrule uncertainty on weak evidence alone.
    assert not detect(articulate(capture(), .35))['active']


def test_yaw_foreshortening_cannot_invent_articulation():
    points = capture()
    for index, scale in enumerate(1 - .45 * (np.sin(np.arange(len(points)) / 5) + 1) / 2):
        points[index, :, 0] = (points[index, :, 0] - 128) * scale + 128
    assert not detect(points)['active']


def test_pose_change_during_large_mouth_change_remains_uncertain():
    points = capture()
    points[7:15, [65, 66, 67], 1] += 6
    points[7:15, 33, 0] += 12
    assert not detect(points)['active']


def test_tracking_dropout_during_apparent_movement_cannot_admit_it():
    points = capture()
    points[7:10, [65, 66, 67], 1] += 6
    observed = np.ones(len(points), dtype=bool)
    observed[7:10] = False
    assert not detect(points, observed)['active']


def test_tracking_dropout_elsewhere_does_not_erase_clear_later_articulation():
    points = capture()
    points[30:34, [65, 66, 67], 1] += 6
    observed = np.ones(len(points), dtype=bool)
    observed[5] = False
    assert detect(points, observed)['active']


def test_coherent_outer_and_inner_width_changes_are_detected():
    points = capture()
    points[7:15, [48, 60], 0] -= 4
    points[7:15, [54, 64], 0] += 4
    assert detect(points)['active']


@pytest.mark.parametrize('kind', ['nan', 'short', 'shape', 'fps', 'eyes'])
def test_invalid_geometry_does_not_provide_positive_evidence(kind):
    points = articulate(capture(), 6.)
    observed = np.ones(len(points), dtype=bool)
    fps = 25.
    if kind == 'nan':
        points[5, 62, 0] = np.nan
    elif kind == 'short':
        points, observed = points[:5], observed[:5]
    elif kind == 'shape':
        points = points[:, :67]
    elif kind == 'fps':
        fps = 0
    else:
        points[:, [36, 39, 42, 45]] = 100
    assert not detect_articulation(points, observed, fps)['active']

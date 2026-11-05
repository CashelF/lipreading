"""Behavioral checks for rigid registration; synthetic images are not speech data."""
import json

import cv2
import numpy as np
import pytest

from backend.registered_activity import confirm_registered_stillness
from test_activity import capture


def image():
    rng = np.random.default_rng(47)
    texture = rng.integers(70, 190, (256, 256), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    cv2.ellipse(texture, (128, 196), (36, 9), 0, 0, 360, 30, -1)
    cv2.line(texture, (105, 194), (151, 194), 220, 2)
    return np.repeat(texture[:, :, None], 3, axis=2)


def clip(count=26):
    return [image() for _ in range(count)], capture(count), np.ones(count, dtype=bool)


def mouth_shift(frame, amount):
    x, y = np.meshgrid(np.arange(256, dtype=np.float32), np.arange(256, dtype=np.float32))
    envelope = np.exp(-(((x - 128) / 42) ** 2 + ((y - 196) / 14) ** 2))
    return cv2.remap(frame, x, y - amount * envelope, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def test_exact_still_has_reliable_zero_mouth_change_without_input_mutation():
    frames, track, observed = clip()
    saved_frame, saved_track = frames[0].copy(), track.copy()
    result = confirm_registered_stillness(frames, track, observed)
    assert result['reliable'] and result['confirmed'], result
    assert result['metrics']['registered_mouth_tail'] == 0
    assert np.array_equal(frames[0], saved_frame)
    assert np.array_equal(track, saved_track)
    json.dumps(result)


def test_head_translation_is_not_local_mouth_movement():
    frames, track, observed = clip()
    for index in range(len(frames)):
        dx, dy = index % 5 - 2, index % 3 - 1
        frames[index] = np.roll(frames[index], (dy, dx), axis=(0, 1))
        track[index] += [dx, dy]
    result = confirm_registered_stillness(frames, track, observed)
    assert result['reliable'] and result['confirmed'], result


@pytest.mark.parametrize('amount', [.25, .5, 1.])
def test_brief_local_mouth_motion_survives_with_frozen_landmarks(amount):
    frames, track, observed = clip()
    frames[7] = mouth_shift(frames[7], amount)
    result = confirm_registered_stillness(frames, track, observed)
    assert result['reliable'] and not result['confirmed'], result


def test_subtle_mouth_motion_remains_when_head_also_moves():
    frames, track, observed = clip()
    frames[7] = mouth_shift(frames[7], .25)
    for index in range(len(frames)):
        dx, dy = index % 5 - 2, index % 3 - 1
        frames[index] = np.roll(frames[index], (dy, dx), axis=(0, 1))
        track[index] += [dx, dy]
    result = confirm_registered_stillness(frames, track, observed)
    assert result['reliable'] and not result['confirmed'], result


def test_mild_rotation_interpolation_is_uncertain_instead_of_speech():
    frames, track, observed = clip()
    for index in range(len(frames)):
        transform = cv2.getRotationMatrix2D((128., 150.), .5 * np.sin(index / 7), 1.)
        frames[index] = cv2.warpAffine(frames[index], transform, (256, 256),
                                      borderMode=cv2.BORDER_REPLICATE)
        track[index] = cv2.transform(track[index][None].astype(np.float32), transform)[0]
    result = confirm_registered_stillness(frames, track, observed)
    assert not result['reliable'] and not result['confirmed'], result
    assert result['metrics']['registration_control_residual'] > 2


def test_uniform_exposure_change_is_measured_outside_mouth():
    frames, track, observed = clip()
    frames[7] = frames[7] + 5
    result = confirm_registered_stillness(frames, track, observed)
    assert result['reliable'] and result['confirmed'], result
    assert result['metrics']['registration_brightness_offset'] == pytest.approx(5, abs=.2)


def test_exposure_correction_does_not_erase_local_articulation():
    frames, track, observed = clip()
    frames[7] = mouth_shift(frames[7], .25) + 5
    result = confirm_registered_stillness(frames, track, observed)
    assert result['reliable'] and not result['confirmed'], result


def test_appending_still_video_does_not_dilute_brief_motion():
    frames, track, observed = clip(76)
    frames[7] = mouth_shift(frames[7], .25)
    short = confirm_registered_stillness(frames[:26], track[:26], observed[:26])
    long = confirm_registered_stillness(frames, track, observed)
    assert short['reliable'] and long['reliable']
    assert not short['confirmed'] and not long['confirmed']
    assert short['metrics']['registered_mouth_tail'] == long['metrics']['registered_mouth_tail']


@pytest.mark.parametrize('failure', ['dropout', 'profile', 'nan', 'small', 'short', 'format', 'dimensions', 'texture', 'large_motion'])
def test_uncertain_registration_never_claims_stillness(failure):
    frames, track, observed = clip()
    if failure == 'dropout':
        observed[3] = False
    elif failure == 'profile':
        track[:, 33, 0] += 80
    elif failure == 'nan':
        track[3, 10, 0] = np.nan
    elif failure == 'small':
        track[:, [36, 39, 42, 45]] = 100
    elif failure == 'short':
        frames, track, observed = frames[:4], track[:4], observed[:4]
    elif failure == 'format':
        frames[3] = frames[3].astype(np.float32)
    elif failure == 'dimensions':
        frames[3] = frames[3][:-1]
    elif failure == 'texture':
        frames = [np.full_like(frame, 128) for frame in frames]
    elif failure == 'large_motion':
        frames[3] = np.roll(frames[3], 50, axis=1)
    result = confirm_registered_stillness(frames, track, observed)
    assert not result['reliable'] and not result['confirmed'], result

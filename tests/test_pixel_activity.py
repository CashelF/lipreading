"""Synthetic image checks for the conservative still-mouth corroboration."""
import json

import cv2
import numpy as np
import pytest

from backend.activity import confirm_image_stillness
from test_activity import capture


def image():
    frame = np.full((256, 256, 3), 160, dtype=np.uint8)
    cv2.ellipse(frame, (128, 196), (32, 8), 0, 0, 360, (30, 30, 30), -1)
    cv2.ellipse(frame, (128, 196), (37, 11), 0, 0, 360, (110, 110, 110), 2)
    return frame


def clip(count=51):
    return [image() for _ in range(count)], capture(count), np.ones(count, dtype=bool)


def shifted(frame, offset):
    return cv2.warpAffine(frame, np.float32([[1, 0, 0], [0, 1, offset]]),
                          (frame.shape[1], frame.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def test_still_mouth_confirmed_without_mutating_inputs():
    frames, track, observed = clip()
    original = frames[0].copy()
    original_track = track.copy()
    result = confirm_image_stillness(frames, track, observed)
    assert result['confirmed'] is True
    assert result['metrics']['pixel_tail_change'] == 0
    assert result['metrics']['pixel_peak_change'] == 0
    assert np.array_equal(frames[0], original)
    assert np.array_equal(track, original_track)
    json.dumps(result)


def test_one_gray_level_sensor_noise_is_still():
    frames, track, observed = clip()
    rng = np.random.default_rng(420)
    frames = [np.clip(frame.astype(float) + rng.normal(0, 1, frame.shape), 0, 255)
              .round().astype(np.uint8) for frame in frames]
    assert confirm_image_stillness(frames, track, observed)['confirmed'] is True


def test_sensor_noise_with_jpeg_is_still():
    frames, track, observed = clip()
    rng = np.random.default_rng(420)
    encoded_frames = []
    for frame in frames:
        noisy = np.clip(frame.astype(float) + rng.normal(0, 1, frame.shape), 0, 255).round().astype(np.uint8)
        encoded = cv2.imencode('.jpg', noisy, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]
        encoded_frames.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
    assert confirm_image_stillness(encoded_frames, track, observed)['confirmed'] is True


def test_lower_contrast_quarter_pixel_motion_is_not_still():
    frames, track, observed = clip()
    base = np.full(frames[0].shape, 160, dtype=np.uint8)
    cv2.ellipse(base, (128, 196), (32, 8), 0, 0, 360, (140, 140, 140), -1)
    frames = [base.copy() for _ in frames]
    frames[7] = shifted(frames[7], .25)
    assert confirm_image_stillness(frames, track, observed)['confirmed'] is False


@pytest.mark.parametrize('offset', [.25, .5, 1])
def test_one_frame_subpixel_motion_survives_in_unmoving_landmarks(offset):
    frames, track, observed = clip()
    frames[7] = shifted(frames[7], offset)
    result = confirm_image_stillness(frames, track, observed)
    assert result['confirmed'] is False
    assert result['metrics']['pixel_tail_change'] > 2


def test_early_short_burst_is_not_diluted_by_appending_silence():
    frames, track, observed = clip(151)
    frames[7] = shifted(frames[7], .25)
    values = [confirm_image_stillness(frames[:n], track[:n], observed[:n]) for n in (51, 101, 151)]
    assert all(not value['confirmed'] for value in values)
    assert len({value['metrics']['pixel_tail_change'] for value in values}) == 1


def test_slow_accumulated_motion_passes():
    frames, track, observed = clip()
    frames = [shifted(frame, n / len(frames)) for n, frame in enumerate(frames)]
    assert confirm_image_stillness(frames, track, observed)['confirmed'] is False


def test_head_translation_does_not_get_image_confirmation():
    frames, track, observed = clip()
    for n in range(len(frames)):
        offset = .5 * np.sin(n / 4)
        frames[n] = shifted(frames[n], offset)
        track[n, :, 1] += offset
    assert confirm_image_stillness(frames, track, observed)['confirmed'] is False


def test_global_brightness_change_is_not_normalized_away():
    frames, track, observed = clip()
    frames[10] += 4
    assert confirm_image_stillness(frames, track, observed)['confirmed'] is False


def test_mouth_must_remain_inside_fixed_box_for_every_frame():
    frames, track, observed = clip()
    track[7, 48:68, 0] += 70
    result = confirm_image_stillness(frames, track, observed)
    assert result['confirmed'] is False
    assert 'outside the fixed' in result['detail']


@pytest.mark.parametrize('bad', ['dropout', 'short', 'nan', 'shape', 'dtype', 'dimensions', 'gray', 'eye', 'mouth', 'bounds', 'small'])
def test_bad_tracking_or_images_fail_open(bad):
    frames, track, observed = clip()
    if bad == 'dropout':
        observed[3] = False
    elif bad == 'short':
        frames, track, observed = frames[:3], track[:3], observed[:3]
    elif bad == 'nan':
        track[3, 49, 0] = np.nan
    elif bad == 'shape':
        track = track[:, :67]
    elif bad == 'dtype':
        frames[3] = frames[3].astype(np.float32)
    elif bad == 'dimensions':
        frames[3] = frames[3][:250]
    elif bad == 'gray':
        frames[3] = frames[3][:, :, 0]
    elif bad == 'eye':
        track[:, [36, 39, 42, 45]] = 100
    elif bad == 'mouth':
        track[:, 48:68] = 100
    elif bad == 'bounds':
        track[:, 48:68, 0] -= 100
    elif bad == 'small':
        track[:, 48:68] = (track[:, 48:68] - [128, 196]) * .05 + [128, 196]
    assert confirm_image_stillness(frames, track, observed)['confirmed'] is False

"""The image confirmation guard must control the real preprocessing boundary."""
import numpy as np
import pytest

from backend import activity as activity_module
from backend.preprocessing import PreprocessCache
from test_preprocessing import preprocessor, reference_face


def capture(count=20):
    y, x = np.mgrid[:256, :256]
    image = np.stack([(x + 50) % 256, (y + 75) % 256, (x + y + 100) % 256], axis=2).astype(np.uint8)
    return [image.copy() for _ in range(count)]


def processor(count=20):
    return preprocessor([[reference_face()]] * count)


def landmark_result(state, *, full_geometry=True):
    metrics = {'tracked_fraction': 1.0, 'frames': 20}
    if full_geometry:
        metrics['shape_motion'] = 0.0 if state == 'still' else .02
    return {'state': state, 'suppress': state == 'still',
            'detail': 'Controlled landmark observation.', 'metrics': metrics}


def install_helpers(monkeypatch, landmarks, confirmed, *, registered_reliable=True, registered_confirmed=False):
    from backend import registered_activity
    calls = []

    def analyze(track, observed, fps=25.):
        calls.append(('landmarks', track.copy(), observed.copy(), fps))
        return {**landmarks, 'metrics': landmarks['metrics'].copy()}

    def confirm(frames, track, observed, fps=25.):
        calls.append(('pixels', frames, track.copy(), observed.copy(), fps))
        return {'confirmed': confirmed, 'detail': 'Controlled image observation.',
                'metrics': {'pixel_tail_change': 0.0 if confirmed else 30.0}}

    monkeypatch.setattr(activity_module, 'analyze_activity', analyze)
    monkeypatch.setattr(activity_module, 'confirm_image_stillness', confirm)

    def register(frames, track, observed, fps=25.):
        calls.append(('registered', frames, track.copy(), observed.copy(), fps))
        return {'confirmed': registered_confirmed, 'reliable': registered_reliable,
                'detail': 'Controlled registration.', 'metrics': {'registration_reliable': registered_reliable}}

    monkeypatch.setattr(registered_activity, 'confirm_registered_stillness', register)
    return calls


def no_warp(*args, **kwargs):
    pytest.fail('Suppressed clip reached image warping')


def assert_same_tensors(actual, expected):
    assert actual.face.dtype == expected.face.dtype
    assert actual.lips.dtype == expected.lips.dtype
    assert actual.face.tobytes() == expected.face.tobytes()
    assert actual.lips.tobytes() == expected.lips.tobytes()
    assert actual.quality == expected.quality


def test_landmark_still_requires_pixel_confirmation_and_returns_before_roi_warp(monkeypatch):
    import skimage.transform
    frames = capture()
    cache = PreprocessCache()
    calls = install_helpers(monkeypatch, landmark_result('still'), confirmed=True)
    monkeypatch.setattr(skimage.transform, 'warp', no_warp)
    result = processor()(frames, speech_gate=True, cache=cache)
    assert result.face is result.lips is None
    assert result.activity['state'] == 'still'
    assert result.activity['suppress'] is True
    assert result.activity['metrics']['shape_motion'] == 0.0
    assert result.activity['metrics']['pixel_tail_change'] == 0.0
    assert cache.stats()['warp_calls'] == 0
    assert [call[0] for call in calls] == ['landmarks', 'pixels']
    assert calls[0][1].shape == calls[1][2].shape == (20, 68, 2)
    assert np.array_equal(calls[0][1], calls[1][2])
    assert calls[0][2].all() and calls[1][3].all()
    assert all(sent is original for sent, original in zip(calls[1][1], frames))
    assert calls[0][-1] == calls[1][-1] == 25


def test_changed_pixels_override_landmark_stillness_and_preserve_roi_values(monkeypatch):
    frames = capture()
    expected = processor()(frames, speech_gate=False)
    calls = install_helpers(monkeypatch, landmark_result('still'), confirmed=False)
    result = processor()(frames, speech_gate=True)
    assert result.activity['state'] == 'motion'
    assert result.activity['suppress'] is False
    assert result.activity['metrics']['pixel_tail_change'] == 30.0
    assert [call[0] for call in calls] == ['landmarks', 'pixels', 'registered']
    assert_same_tensors(result, expected)


@pytest.mark.parametrize('state', ['motion', 'uncertain'])
def test_confirmed_static_pixels_can_override_landmark_jitter_after_geometry_checks(monkeypatch, state):
    import skimage.transform
    calls = install_helpers(monkeypatch, landmark_result(state), confirmed=True)
    monkeypatch.setattr(skimage.transform, 'warp', no_warp)
    result = processor()(capture(), speech_gate=True)
    assert result.face is result.lips is None
    assert result.activity['state'] == 'still'
    assert result.activity['suppress'] is True
    assert result.activity['metrics']['shape_motion'] == .02
    assert result.activity['metrics']['pixel_tail_change'] == 0.0
    assert [call[0] for call in calls] == ['landmarks', 'pixels']


def test_unreliable_tracking_abstains_instead_of_decoding_guesses(monkeypatch):
    frames = capture()
    landmarks = landmark_result('uncertain', full_geometry=False)
    landmarks['detail'] = 'Head pose or tracking is uncertain.'
    calls = install_helpers(monkeypatch, landmarks, confirmed=True, registered_reliable=False)
    result = processor()(frames, speech_gate=True)
    assert result.activity['state'] == 'uncertain'
    assert result.activity['suppress'] is True
    assert result.face is result.lips is None
    assert [call[0] for call in calls] == ['landmarks', 'registered']


def test_independent_registration_resolves_landmark_alignment_jitter(monkeypatch):
    calls = install_helpers(monkeypatch, landmark_result('uncertain', full_geometry=False),
                            confirmed=False, registered_confirmed=True)
    result = processor()(capture(), speech_gate=True)
    assert result.activity['state'] == 'still'
    assert result.activity['suppress'] is True
    assert result.face is result.lips is None
    assert [call[0] for call in calls] == ['landmarks', 'registered']


def test_unreliable_registration_does_not_treat_apparent_motion_as_speech(monkeypatch):
    install_helpers(monkeypatch, landmark_result('motion'), confirmed=False, registered_reliable=False)
    result = processor()(capture(), speech_gate=True)
    assert result.activity['state'] == 'uncertain'
    assert result.activity['suppress'] is True
    assert result.face is result.lips is None


def test_clear_articulation_survives_failed_image_alignment(monkeypatch):
    from test_activity import capture as landmark_capture, articulate
    from backend import registered_activity
    track = articulate(landmark_capture(51), 8.).astype(np.float32)
    frames = capture(51)
    # Raw frames change, and alignment reports the same kind of small skin-pixel
    # residual seen in the user's logs. Neither may veto coherent lip opening.
    for index, frame in enumerate(frames):
        frames[index] = np.roll(frame, index % 5, axis=0)
    monkeypatch.setattr(registered_activity, 'confirm_registered_stillness',
                        lambda *args, **kwargs: {'confirmed': False, 'reliable': False,
                            'detail': 'Lighting residual', 'metrics': {'registration_control_residual': 2.54}})
    observations = [[points] for points in track]
    expected = preprocessor(observations)(frames, speech_gate=False)
    actual = preprocessor(observations)(frames, speech_gate=True)
    assert actual.activity['state'] == 'motion'
    assert actual.activity['suppress'] is False
    assert actual.activity['metrics']['articulation_active'] is True
    assert_same_tensors(actual, expected)


def test_confirmed_image_stillness_cannot_be_overridden_by_landmark_jitter(monkeypatch):
    from backend import articulation
    install_helpers(monkeypatch, landmark_result('motion'), confirmed=False, registered_confirmed=True)

    def forbidden(*args, **kwargs):
        pytest.fail('Positive landmark evidence must not override confirmed image stillness')

    monkeypatch.setattr(articulation, 'detect_articulation', forbidden)
    actual = processor()(capture(), speech_gate=True)
    assert actual.activity['suppress'] is True
    assert actual.face is actual.lips is None


def test_motion_without_image_confirmation_keeps_original_decode_input(monkeypatch):
    frames = capture()
    expected = processor()(frames, speech_gate=False)
    install_helpers(monkeypatch, landmark_result('motion'), confirmed=False)
    result = processor()(frames, speech_gate=True)
    assert result.activity['state'] == 'motion'
    assert result.activity['suppress'] is False
    assert_same_tensors(result, expected)


def test_gate_off_bypasses_both_activity_helpers_and_preserves_exact_roi_tensors(monkeypatch):
    from backend import registered_activity
    frames = capture()
    expected = processor()(frames)

    def forbidden(*args, **kwargs):
        pytest.fail('Disabled filter called an activity helper')

    monkeypatch.setattr(activity_module, 'analyze_activity', forbidden)
    monkeypatch.setattr(activity_module, 'confirm_image_stillness', forbidden)
    monkeypatch.setattr(registered_activity, 'confirm_registered_stillness', forbidden)
    actual = processor()(frames, speech_gate=False, cache=PreprocessCache())
    assert_same_tensors(actual, expected)
    assert actual.activity is None


def test_real_helpers_reject_detector_jitter_over_unchanged_mouth_images(monkeypatch):
    import skimage.transform
    frames = capture()
    observations = []
    for index in range(len(frames)):
        points = reference_face()
        # Articulatory landmark jitter alone must not create captions when the
        # same source pixels prove that the mouth did not actually change.
        points[48:68, 0] += 1.5 if index % 4 < 2 else -1.5
        observations.append([points])
    monkeypatch.setattr(skimage.transform, 'warp', no_warp)
    result = preprocessor(observations)(frames, speech_gate=True)
    assert result.face is result.lips is None
    assert result.activity['suppress'] is True
    assert result.activity['metrics']['shape_motion'] > .012
    assert result.activity['metrics']['pixel_tail_change'] == 0.0


def test_real_tracking_gap_abstains_despite_static_source_pixels(monkeypatch):
    frames = capture()
    observations = [[reference_face()] for _ in frames]
    observations[10] = None

    def forbidden_confirmation(*args, **kwargs):
        pytest.fail('Tracking uncertainty must not be overridden by static pixels')

    monkeypatch.setattr(activity_module, 'confirm_image_stillness', forbidden_confirmation)
    result = preprocessor(observations)(frames, speech_gate=True)
    assert result.face is result.lips is None
    assert result.activity['state'] == 'uncertain'
    assert result.activity['suppress'] is True
    assert result.activity['metrics']['tracked_fraction'] == .95
    assert 'shape_motion' not in result.activity['metrics']

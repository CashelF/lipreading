"""Integration of learned speech gating with unchanged FLARE image tensors."""
import numpy as np
import pytest

from backend.preprocessing import PreprocessCache
from test_preprocessing_cache import moving_capture, processor, assert_identical


class BoxDetector:
    """Explicit SFD/FAN return-value double, including box collection requests."""

    def __init__(self, observations, missing_boxes=()):
        self.observations = observations
        self.missing_boxes = set(missing_boxes)
        self.calls = 0
        self.box_calls = 0

    def get_landmarks_from_image(self, frame, *, return_bboxes=False):
        self.calls += 1
        points = self.observations[id(frame)]
        landmarks = None if points is None else [points.copy()]
        if not return_bboxes:
            return landmarks
        self.box_calls += 1
        boxes = (None if points is None or id(frame) in self.missing_boxes
                 else [np.array([40., 35., 220., 230., .99])])
        return landmarks, None, boxes


class VisualDouble:
    def __init__(self, score=.9):
        self.score = score
        self.calls = []
        self.error = None

    def predict(self, frames, boxes):
        self.calls.append((frames, boxes.copy()))
        if self.error is not None:
            raise self.error
        return np.full(len(frames), self.score, dtype=np.float32)


def make_processor(frames, observations, score=.9, missing_boxes=()):
    detector = BoxDetector(observations, missing_boxes)
    result = processor(detector)
    result.visual_speech = VisualDouble(score)
    return result


def test_admitted_visual_speech_preserves_exact_flare_tensors(monkeypatch):
    import backend.activity
    frames, observations = moving_capture(51)
    baseline = make_processor(frames, observations)(frames, speech_gate=False)
    proc = make_processor(frames, observations)

    def forbidden(*args, **kwargs):
        pytest.fail('The old landmark heuristics must never override the learned detector.')

    monkeypatch.setattr(backend.activity, 'analyze_activity', forbidden)
    actual = proc(frames, speech_gate=True)
    assert actual.activity['state'] == 'motion'
    assert actual.activity['suppress'] is False
    assert_identical(actual, baseline)
    assert len(proc.visual_speech.calls) == 1
    assert proc.detector.calls == proc.detector.box_calls == 51


def test_quiet_visual_scores_skip_warps_and_preserve_transcript(monkeypatch):
    import skimage.transform
    frames, observations = moving_capture(51)
    proc = make_processor(frames, observations, score=.1)

    def forbidden(*args, **kwargs):
        pytest.fail('A quiet clip must not allocate FLARE face/lip image tensors.')

    monkeypatch.setattr(skimage.transform, 'warp', forbidden)
    result = proc(frames, speech_gate=True)
    assert result.face is result.lips is None
    assert result.activity['state'] == 'still'
    assert result.activity['suppress'] is True
    assert result.activity['preserve_transcript'] is True
    assert result.activity['metrics']['speech_max_mean'] == pytest.approx(.1)


def test_filter_off_does_not_run_visual_detector_or_request_extra_boxes():
    frames, observations = moving_capture(51)
    proc = make_processor(frames, observations)
    proc.visual_speech.error = AssertionError('Visual detector should be bypassed.')
    result = proc(frames, speech_gate=False)
    assert result.activity is None
    assert result.face is not None
    assert not proc.visual_speech.calls
    assert proc.detector.calls == 51
    assert proc.detector.box_calls == 0


def test_overlapping_windows_reuse_sfd_boxes_and_preserve_sample_phase():
    frames, observations = moving_capture(76)
    proc = make_processor(frames, observations)
    cache = PreprocessCache()
    supplied = 10000. + np.arange(51) * 40
    cache.set_sample_times(supplied)
    supplied[:] = 0  # The setter owns its timestamp copy.
    first = proc(frames[:51], speech_gate=True, cache=cache)
    assert first.activity['state'] == 'motion'
    assert cache.speech_state.last_timestamp_ms == 11520.
    cache.set_sample_times(10000. + np.arange(76) * 40)
    second = proc(frames, speech_gate=True, cache=cache)
    assert second.activity['state'] == 'motion'
    assert second.activity['metrics']['speech_new_frames'] == 25
    assert cache.speech_state.last_timestamp_ms == 12520.
    assert proc.detector.calls == proc.detector.box_calls == 76
    assert cache.detector_hits == 51
    assert len(proc.visual_speech.calls) == 2
    cache.clear()
    assert cache.sample_times_ms is None and cache.speech_state is None
    assert cache.stats()['entries'] == 0


def test_missing_face_abstains_without_legacy_motion_override():
    frames, observations = moving_capture(51, dropouts=(24,))
    proc = make_processor(frames, observations)
    result = proc(frames, speech_gate=True)
    assert result.activity['state'] == 'uncertain'
    assert result.activity['suppress'] is True
    assert result.activity['preserve_transcript'] is True
    assert not proc.visual_speech.calls
    assert result.face is result.lips is None


@pytest.mark.parametrize('failure', [ValueError('bad crop'), RuntimeError('classifier unavailable')])
def test_visual_failure_abstains_and_clears_mean_history_without_rewinding(failure):
    frames, observations = moving_capture(76)
    proc = make_processor(frames, observations)
    cache = PreprocessCache()
    cache.set_sample_times(np.arange(51) * 40.)
    proc(frames[:51], speech_gate=True, cache=cache)
    assert cache.speech_state.history
    watermark = cache.speech_state.last_timestamp_ms
    proc.visual_speech.error = failure
    cache.set_sample_times(np.arange(76) * 40.)
    result = proc(frames, speech_gate=True, cache=cache)
    assert result.activity['state'] == 'uncertain'
    assert result.activity['suppress'] is True
    assert result.activity['preserve_transcript'] is True
    assert cache.speech_state.last_timestamp_ms == watermark
    assert not cache.speech_state.history


def test_exact_frozen_video_cannot_activate_even_if_classifier_would():
    source, observations = moving_capture(1)
    frames = source * 51
    proc = make_processor(frames, observations, score=.99)
    cache = PreprocessCache()
    cache.set_sample_times(np.arange(51) * 40.)
    result = proc(frames, speech_gate=True, cache=cache)
    assert not proc.visual_speech.calls
    assert result.activity['state'] == 'still'
    assert result.activity['metrics']['frozen_video'] is True
    assert cache.speech_state.last_timestamp_ms == 1520.
    assert proc.detector.calls == 1


def test_mismatched_timestamp_count_abstains_instead_of_guessing_time():
    frames, observations = moving_capture(51)
    proc = make_processor(frames, observations)
    cache = PreprocessCache()
    cache.set_sample_times(np.arange(50) * 40.)
    result = proc(frames, speech_gate=True, cache=cache)
    assert result.activity['state'] == 'uncertain'
    assert result.activity['preserve_transcript'] is True
    assert not proc.visual_speech.calls


def test_short_final_without_context_abstains_and_keeps_prior_words():
    frames, observations = moving_capture(20)
    proc = make_processor(frames, observations)
    result = proc(frames, speech_gate=True)
    assert result.activity['state'] == 'uncertain'
    assert result.activity['preserve_transcript'] is True
    assert result.face is result.lips is None


def test_small_face_guidance_abstains_and_clears_when_tracking_recovers():
    frames, observations = moving_capture(51)
    small = {key: points * .4 for key, points in observations.items()}
    result = make_processor(frames, small)(frames, speech_gate=True)
    assert result.activity['camera_hint'] == 'move_closer'
    assert result.activity['state'] == 'uncertain'
    assert result.activity['suppress'] is True
    assert result.activity['preserve_transcript'] is True
    assert result.face is result.lips is None
    recovered = make_processor(frames, observations)(frames, speech_gate=True)
    assert 'camera_hint' not in recovered.activity
    assert recovered.activity['suppress'] is False


def test_missing_face_has_actionable_camera_guidance():
    frames, observations = moving_capture(51)
    missing = {key: None for key in observations}
    result = make_processor(frames, missing)(frames, speech_gate=True)
    assert result.activity['camera_hint'] == 'face_not_visible'
    assert result.activity['preserve_transcript'] is True
    assert result.face is result.lips is None

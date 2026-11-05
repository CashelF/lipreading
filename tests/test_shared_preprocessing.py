"""Shared immutable work must preserve pixels, decisions and state ownership."""
import numpy as np
import pytest

from backend.preprocessing import FrameCache, PreprocessCache
from test_preprocessing_cache import moving_capture, assert_identical
from test_visual_gate_preprocessing import make_processor


def test_activity_only_skips_even_admitted_flare_warps(monkeypatch):
    import skimage.transform
    frames, observations = moving_capture(76)
    expected = make_processor(frames, observations)(frames, speech_gate=True)
    proc = make_processor(frames, observations)
    def forbidden(*args, **kwargs):
        pytest.fail('Activity inspection must not construct unused FLARE ROIs')
    monkeypatch.setattr(skimage.transform, 'warp', forbidden)
    actual = proc(frames, speech_gate=True, activity_only=True)
    assert actual.activity == expected.activity
    assert actual.quality == expected.quality
    assert actual.face is actual.lips is None


def test_shared_observations_do_not_share_speech_state_or_change_pixels():
    frames, observations = moving_capture(151)
    proc = make_processor(frames, observations)
    pool = FrameCache()
    gate, recognition = PreprocessCache(frames=pool), PreprocessCache(frames=pool)
    for count in [51, 76, 101, 151]:
        gate.set_sample_times(np.arange(count) * 40.)
        activity = proc(frames[:count], cache=gate, speech_gate=True, activity_only=True)
        watermark = gate.speech_state.last_timestamp_ms
        actual = proc(frames[:count], cache=recognition)
        expected = make_processor(frames, observations)(frames[:count])
        assert_identical(actual, expected)
        assert gate.speech_state.last_timestamp_ms == watermark
        assert recognition.speech_state is recognition.sample_times_ms is None
        assert activity.activity['state'] == 'motion'
    assert proc.detector.calls == 151
    assert recognition.detector_calls == 0
    assert gate.warp_calls == 0


def test_repeated_sampled_frames_reuse_distinct_exact_alignments():
    frames, observations = moving_capture(51)
    repeated = [frame for frame in frames for _ in range(2)]
    proc = make_processor(frames, observations)
    cache = PreprocessCache()
    expected = proc(repeated)
    first = proc(repeated, cache=cache)
    warps = cache.warp_calls
    second = proc(repeated, cache=cache)
    assert_identical(first, expected)
    assert_identical(second, expected)
    assert cache.warp_calls == warps
    assert cache.warp_hits >= len(repeated)
    assert all(len(entry.rois) <= 4 for entry in cache._entries.values())


def test_binding_second_consumer_keeps_observations_and_detector_change_invalidates():
    frames, observations = moving_capture(51)
    pool = FrameCache()
    gate, recognition = PreprocessCache(frames=pool), PreprocessCache(frames=pool)
    proc = make_processor(frames, observations)
    gate.set_sample_times(np.arange(51) * 40.)
    proc(frames, cache=gate, speech_gate=True, activity_only=True)
    old_state = gate.speech_state
    proc(frames, cache=recognition)
    assert proc.detector.calls == 51
    assert gate.speech_state is old_state
    replacement = make_processor(frames, observations)
    replacement(frames, cache=recognition)
    assert replacement.detector.calls == 51
    gate.bind(replacement.detector)
    assert gate.speech_state is None
    assert not old_state.history


def test_reader_shares_pixels_but_owns_two_temporal_caches():
    from backend.utterance_stream import UtteranceReader
    reader = UtteranceReader(object())
    assert reader.gate_decode is reader.utterance_decode
    assert reader.gate_preprocess is not reader.utterance_preprocess
    assert reader.gate_preprocess.frames is reader.utterance_preprocess.frames
    frames, _ = moving_capture(1)
    reader.gate_preprocess.entry(frames[0])
    reader.clear()
    assert reader.gate_preprocess.stats()['entries'] == 0
    assert reader.utterance_preprocess.stats()['entries'] == 0


def test_speaking_crop_reuse_matches_original_pixels_and_tracks_geometry(monkeypatch):
    import cv2
    from backend.visual_speech import crop_faces
    frames, _ = moving_capture(76)
    cache = PreprocessCache()
    boxes = np.tile([40., 35., 220., 230., .99], (len(frames), 1))
    for count in (51, 76):
        expected = crop_faces(frames[:count], boxes[:count])
        assert np.array_equal(crop_faces(frames[:count], boxes[:count], cache=cache), expected)
    original = cv2.resize
    calls = []
    def resize(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, 'resize', resize)
    assert np.array_equal(crop_faces(frames, boxes, cache=cache), expected)
    assert not calls
    changed = boxes.copy()
    changed[:, :4] += 3
    actual = crop_faces(frames, changed, cache=cache)
    assert calls
    assert np.array_equal(actual, crop_faces(frames, changed))

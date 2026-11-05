"""Accuracy checks for reuse across growing clips, with production image geometry."""
import numpy as np
import pytest

from backend.preprocessing import FacePreprocessor, FaceQualityError, PreprocessCache
from backend.limits import MAX_PREPROCESS_FRAMES


def reference_face():
    theta = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    points = np.stack([128 + 80 * np.cos(theta), 128 + 85 * np.sin(theta)], axis=1).astype(np.float32)
    points[[33, 36, 39, 42, 45]] = [[128, 145], [90, 100], [110, 100], [146, 100], [166, 100]]
    mouth = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    points[48:68] = np.stack([128 + 25 * np.cos(mouth), 180 + 10 * np.sin(mouth)], axis=1)
    return points


class IdentityDetector:
    def __init__(self, observations):
        self.observations = observations
        self.calls = 0

    def get_landmarks_from_image(self, frame):
        self.calls += 1
        result = self.observations[id(frame)]
        return None if result is None else [result.copy()]


def processor(detector):
    proc = FacePreprocessor.__new__(FacePreprocessor)
    proc.detector = detector
    proc.mean_face = reference_face()
    proc.mean, proc.std, proc.crop_size = .421, .165, 88
    return proc


def moving_capture(count=151, dropouts=()):
    y, x = np.mgrid[:256, :256]
    frames, observations = [], {}
    for i in range(count):
        frame = np.stack([(x + i) % 256, (y + i * 3) % 256, (x + y + i) % 256], axis=2).astype(np.uint8)
        frame.setflags(write=False)
        points = reference_face()
        points += np.array([np.sin(i / 7) * 9, np.cos(i / 9) * 6], dtype=np.float32)
        points[48:68, 1] += np.sin(i / 3) * 4
        observations[id(frame)] = None if i in dropouts else points
        frames.append(frame)
    return frames, observations


def assert_identical(actual, expected):
    assert actual.face.dtype == expected.face.dtype
    assert actual.lips.dtype == expected.lips.dtype
    assert actual.face.tobytes() == expected.face.tobytes()
    assert actual.lips.tobytes() == expected.lips.tobytes()
    assert actual.quality == expected.quality


@pytest.mark.parametrize('dropouts', [(), (0, 22, 23, 49, 50, 99, 100)])
def test_growing_clips_reuse_only_exact_rois_and_keep_new_context(dropouts):
    frames, observations = moving_capture(dropouts=dropouts)
    uncached = processor(IdentityDetector(observations))
    cached = processor(IdentityDetector(observations))
    cache = PreprocessCache()
    previous = None
    for length in (51, 101, 151):
        actual = cached(frames[:length], cache=cache)
        expected = uncached(frames[:length])
        assert_identical(actual, expected)
        if previous is not None:
            # New future context must actually revise the old tail, not freeze it.
            assert actual.face[:, :, previous.face.shape[2] - 1].tobytes() != previous.face[:, :, -1].tobytes()
        previous = actual
    assert cached.detector.calls == 151
    assert uncached.detector.calls == 303
    assert cache.stats()['detector_hits'] == 152
    assert 151 < cache.stats()['warp_calls'] < 303
    assert cache.stats()['warp_hits'] > 0
    if not dropouts:
        assert cache.stats()['warp_calls'] == 173
        assert cache.stats()['warp_hits'] == 130


def test_cached_missing_observations_remain_quality_errors():
    frames, observations = moving_capture(25, dropouts=tuple(range(25)))
    proc = processor(IdentityDetector(observations))
    cache = PreprocessCache()
    for _ in range(2):
        with pytest.raises(FaceQualityError, match='fully visible'):
            proc(frames, cache=cache)
    assert proc.detector.calls == 25
    assert cache.stats()['detector_hits'] == 25


def test_normalization_changes_invalidate_rois_but_reuse_observations():
    frames, observations = moving_capture(25)
    proc = processor(IdentityDetector(observations))
    cache = PreprocessCache()
    proc(frames, cache=cache)
    proc.mean, proc.std, proc.crop_size = .3, .2, 80
    actual = proc(frames, cache=cache)
    expected = proc(frames)
    assert_identical(actual, expected)
    assert cache.stats()['detector_calls'] == 25
    assert cache.stats()['warp_calls'] == 50


def test_cache_is_bound_to_detector_and_clear_releases_video():
    frames, observations = moving_capture(25)
    proc = processor(IdentityDetector(observations))
    cache = PreprocessCache()
    proc(frames, cache=cache)
    replacement = IdentityDetector(observations)
    proc.detector = replacement
    proc(frames, cache=cache)
    assert replacement.calls == 25
    assert cache.stats()['entries'] == 25
    cache.clear()
    assert cache.stats()['entries'] == 0
    assert cache._detector is None


def test_cache_bound_and_identity_collision_defense():
    cache = PreprocessCache(max_frames=2)
    a, b, c = (np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3))
    entry_a = cache.entry(a)
    cache.entry(b)
    cache.entry(c)
    assert cache.stats()['entries'] == 2
    assert id(a) not in cache._entries
    # Even if an entry were put under the wrong id, pixels are never confused.
    cache._entries[id(c)] = entry_a
    assert cache.entry(c).frame is c


@pytest.mark.parametrize('bound', [0, MAX_PREPROCESS_FRAMES + 1, True, 1.5])
def test_cache_cannot_grow_beyond_server_clip_bound(bound):
    with pytest.raises(ValueError, match=f'between 1 and {MAX_PREPROCESS_FRAMES}'):
        PreprocessCache(max_frames=bound)

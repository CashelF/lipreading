"""JPEG reuse must preserve capture timing, values, and session isolation."""
from io import BytesIO
import math
import struct

import numpy as np
from PIL import Image
import pytest

from backend.stream import Clip, ClipBuffer, DecodeCache, Frame, resample


def source_frames(count=151):
    result = []
    for index in range(count):
        data = BytesIO()
        Image.new('RGB', (128, 128), (index % 256, 30, 60)).save(data, format='JPEG')
        result.append(Frame(index * 40., data.getvalue()))
    return result


def test_growing_clips_decode_each_source_once_with_exact_values_and_timing():
    frames = source_frames()
    cache = DecodeCache()
    previous = None
    for count in (51, 101, 151):
        clip = Clip(0, tuple(frames[:count]), count == 151)
        expected, expected_fps = resample(clip)
        actual, actual_fps = resample(clip, cache=cache)
        assert actual_fps == expected_fps == 25
        assert len(actual) == count
        assert all(a.tobytes() == b.tobytes() for a, b in zip(actual, expected))
        assert all(not array.flags.writeable for array in actual)
        if previous is not None:
            assert all(a is b for a, b in zip(actual, previous))
        previous = actual
    assert cache.misses == 151
    assert cache.hits == 152


def test_same_timestamp_with_new_pixels_does_not_reuse_another_session():
    frames = source_frames(25)
    cache = DecodeCache()
    old, _ = resample(Clip(0, tuple(frames), True), cache=cache)
    changed = [Frame(frame.timestamp, frames[-1].jpeg) for frame in frames]
    new, _ = resample(Clip(0, tuple(changed), True), cache=cache)
    assert old[0] is not new[0]
    assert not np.array_equal(old[0], new[0])


def test_eviction_and_clear_release_old_frame_entries():
    frames = source_frames(3)
    cache = DecodeCache(max_frames=2)
    for frame in frames:
        cache.put(frame, np.zeros((96, 96, 3), dtype=np.uint8))
    assert cache.get(frames[0]) is None
    assert cache.get(frames[1]) is not None
    assert cache.get(frames[2]) is not None
    cache.clear()
    assert cache.get(frames[2]) is None
    assert not cache._frames


def test_bad_timing_is_still_rejected_on_cache_hits():
    frames = source_frames(25)
    cache = DecodeCache()
    resample(Clip(0, tuple(frames), False), cache=cache)
    with pytest.raises(ValueError, match='slowly'):
        resample(Clip(0, (frames[0], frames[-1]), True), cache=cache)


def test_bad_new_frame_cannot_be_hidden_by_cache():
    frames = source_frames(25)
    cache = DecodeCache()
    resample(Clip(0, tuple(frames), False), cache=cache)
    frames.append(Frame(1000., b'not a JPEG'))
    with pytest.raises(ValueError, match='decode'):
        resample(Clip(0, tuple(frames), True), cache=cache)


def test_resolution_change_is_rejected_even_when_both_frames_cached():
    frames = source_frames(2)
    cache = DecodeCache()
    cache.put(frames[0], np.zeros((96, 96, 3), dtype=np.uint8))
    cache.put(frames[1], np.zeros((128, 128, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match='resolution changed'):
        resample(Clip(0, tuple(frames), True), cache=cache)


def test_cache_requires_positive_capacity():
    with pytest.raises(ValueError):
        DecodeCache(0)


def test_rolling_windows_share_one_grid_and_reuse_exact_decoded_frames():
    buffer = ClipBuffer()
    cache = DecodeCache()
    origin = 123.45
    image = source_frames(1)[0].jpeg
    clips = []
    for index in range(250):
        clip = buffer.append(struct.pack('<d', origin + index * 33) + image)
        if clip is not None:
            clips.append(clip)

    def target_map(clip, decoded):
        first = math.ceil((clip.start_ms - origin) / 40 - 1e-9)
        last = math.floor((clip.end_ms - origin) / 40 + 1e-9)
        assert len(decoded) == last - first + 1
        return dict(zip(range(first, last + 1), decoded))

    previous = {}
    for clip in clips:
        expected, expected_fps = resample(clip)
        actual, actual_fps = resample(clip, cache=cache)
        assert actual_fps == expected_fps
        assert all(a.tobytes() == b.tobytes() for a, b in zip(actual, expected))
        current = target_map(clip, actual)
        shared = current.keys() & previous.keys()
        if previous:
            # Exclude the old rightmost grid point: a newly arriving closer frame
            # can legitimately improve its nearest-neighbor choice.
            for tick in shared - {max(previous)}:
                assert current[tick] is previous[tick]
        previous = current
    assert cache.hits > cache.misses
    assert len(cache._frames) <= cache.max_frames

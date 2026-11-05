"""The speech watermark must use the same exact grid as frame resampling."""
import numpy as np

from backend.stream import Clip, Frame, sample_times


def test_fractional_window_boundaries_do_not_shift_score_timestamps():
    frames = tuple(Frame(1000. + index * 1000 / 23, b'') for index in range(185))
    clip = Clip(0, frames, False, sample_origin=1000., window_start=3021.739)
    times = sample_times(clip)
    assert times[0] == 3040.
    assert times[-1] == 9000.
    assert np.all(np.diff(times) == 40.)
    assert times[0] >= clip.start_ms
    assert times[-1] <= clip.end_ms

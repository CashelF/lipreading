"""Ensure offline examples are evaluated with the live temporal constraints."""
import pytest

from scripts.evaluate_speaking_examples import replay_windows


def test_replay_has_no_future_frames_and_keeps_rolling_context(tmp_path):
    timestamps = [10000. + index * 40 for index in range(201)]
    names = [f'{index:06d}.jpg' for index in range(len(timestamps))]
    for name in names:
        # The buffer validates framing; actual JPEG decoding is tested elsewhere.
        (tmp_path / name).write_bytes(b'\xff\xd8bounded-frame-fixture')
    windows = list(replay_windows(tmp_path, {
        'frames': len(names), 'frame_files': names, 'timestamps_ms': timestamps,
    }))
    assert [window.end_ms for window in windows] == list(range(12000, 18001, 1000))
    assert [window.start_ms for window in windows] == [10000] * 5 + [11000, 12000]
    for window in windows:
        assert all(frame.timestamp <= window.end_ms for frame in window.frames)
        assert window.end_ms - window.start_ms <= 6000
        assert window.sample_origin == 10000
    # Adjacent updates share source video rather than reset at six seconds.
    assert windows[-2].frames[-1] is windows[-1].frames[-26]


def test_manifest_cannot_read_outside_recording(tmp_path):
    with pytest.raises(ValueError, match='filename'):
        list(replay_windows(tmp_path, {
            'frames': 1, 'frame_files': ['../../private.jpg'], 'timestamps_ms': [0.],
        }))


def test_manifest_mismatched_timestamps_rejected(tmp_path):
    with pytest.raises(ValueError, match='frame counts'):
        list(replay_windows(tmp_path, {
            'frames': 2, 'frame_files': ['000000.jpg'], 'timestamps_ms': [0., 40.],
        }))

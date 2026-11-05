from io import BytesIO
import struct

import numpy as np
from PIL import Image
import pytest

from backend.stream import (Clip, ClipBuffer, Frame, MAX_FRAMES, MAX_PACKET,
                            ensure_overlap, parse_packet, resample)


def jpeg(value=64, size=(128, 128)):
    out = BytesIO()
    Image.new('RGB', size, (value, value, value)).save(out, format='JPEG')
    return out.getvalue()


def packet(timestamp, data=None):
    return struct.pack('<d', timestamp) + (jpeg() if data is None else data)


def test_binary_protocol_preserves_capture_timestamp_and_pixels():
    data = jpeg()
    frame = parse_packet(packet(123.5, data), previous=100)
    assert frame.timestamp == 123.5
    assert frame.jpeg == data


@pytest.mark.parametrize('timestamp', [float('nan'), float('inf'), -1])
def test_rejects_invalid_timestamps(timestamp):
    with pytest.raises(ValueError, match='timestamp'):
        parse_packet(packet(timestamp), previous=None)


@pytest.mark.parametrize('timestamp', [100, 99])
def test_rejects_repeated_or_backward_timestamps(timestamp):
    with pytest.raises(ValueError, match='increase'):
        parse_packet(packet(timestamp), previous=100)


@pytest.mark.parametrize('data', [b'', b'123456789', b'x' * (MAX_PACKET + 1)])
def test_rejects_invalid_packet_sizes(data):
    with pytest.raises(ValueError, match='size'):
        parse_packet(data, None)


def test_rejects_non_jpeg_payload():
    with pytest.raises(ValueError, match='JPEG'):
        parse_packet(packet(0, b'not a jpeg image'), None)


def test_resamples_at_25fps_using_elapsed_time_not_frame_count():
    # 20 fps input still describes one second. The 25 fps model must receive 26
    # samples including both endpoints, rather than play 21 samples too quickly.
    clip = Clip(0, tuple(Frame(i * 50, jpeg(i * 10)) for i in range(21)), True)
    frames, observed = resample(clip)
    assert observed == pytest.approx(20)
    assert len(frames) == 26
    assert frames[0][0, 0, 0] == pytest.approx(0, abs=2)
    assert frames[-1][0, 0, 0] == pytest.approx(200, abs=2)
    assert np.array_equal(frames[2], frames[3])


@pytest.mark.parametrize('timestamps', [[0, 40, 80, 400], [0, 100, 200, 300]])
def test_rejects_stalls_or_insufficient_frame_rate(timestamps):
    clip = Clip(0, tuple(Frame(t, jpeg()) for t in timestamps), True)
    with pytest.raises(ValueError, match='too slowly'):
        resample(clip)


@pytest.mark.parametrize('size', [(95, 128), (1281, 128), (128, 961)])
def test_rejects_out_of_range_decoded_dimensions(size):
    clip = Clip(0, (Frame(0, jpeg(size=size)), Frame(40, jpeg(size=size))), True)
    with pytest.raises(ValueError, match='dimensions'):
        resample(clip)


def test_rejects_corrupted_jpeg_after_magic_bytes():
    clip = Clip(0, (Frame(0, b'\xff\xd8corrupt'), Frame(40, b'\xff\xd8corrupt')), True)
    with pytest.raises(ValueError, match='decode'):
        resample(clip)


def test_rejects_resolution_change_within_clip():
    clip = Clip(0, (Frame(0, jpeg()), Frame(40, jpeg(size=(160, 128)))), True)
    with pytest.raises(ValueError, match='resolution changed'):
        resample(clip)


def test_clip_buffer_emits_overlapping_revisions_without_six_second_reset():
    buf = ClipBuffer()
    image = jpeg()
    clips = [clip for i in range(201) if (clip := buf.append(packet(i * 40, image))) is not None]
    assert [(clip.segment_id, clip.final, len(clip.frames)) for clip in clips] == [
        (0, False, 51), (1, False, 76), (2, False, 101), (3, False, 126),
        (4, False, 151), (5, False, 151), (6, False, 151),
    ]
    assert [(clip.start_ms, clip.end_ms) for clip in clips] == [
        (0, 2000), (0, 3000), (0, 4000), (0, 5000), (0, 6000),
        (1000, 7000), (2000, 8000),
    ]
    assert len(buf.frames) == 151
    assert buf.segment_id == 7
    assert clips[4].frames[25:] == clips[5].frames[:126]
    assert all(a is b for a, b in zip(clips[4].frames[25:], clips[5].frames))
    for i in range(201, 204):
        buf.append(packet(i * 40, image))
    final = buf.finish()
    assert final.segment_id == 7
    assert final.final is True
    assert len(final.frames) == 151
    assert final.start_ms == 2120
    assert final.end_ms == 8120
    assert clips[-1].frames[-1] is final.frames[-4]
    assert buf.frames == []
    assert buf.finish() is None


def test_frame_limit_bounds_memory_at_maximum_accepted_rate():
    buf = ClipBuffer()
    image = jpeg()
    emitted = []
    for i in range(4001):
        clip = buf.append(packet(i * 25, image))
        assert len(buf.frames) < MAX_FRAMES
        if clip:
            emitted.append(clip)
            assert len(clip.frames) <= MAX_FRAMES
            assert clip.end_ms - clip.start_ms <= 6000
            assert clip.frames[0].timestamp <= clip.start_ms
    assert len(emitted) == 99
    assert not any(clip.final for clip in emitted)
    assert emitted[-1].start_ms == 94000
    assert emitted[-1].end_ms == 100000
    assert len(emitted[-1].frames) == 241
    assert len(resample(emitted[-1])[0]) == 151


def test_fast_input_is_dropped_without_changing_timestamps():
    buf = ClipBuffer()
    image = jpeg()
    for timestamp in range(100):
        buf.append(packet(timestamp, image))
    assert [frame.timestamp for frame in buf.frames] == [0, 25, 50, 75]


def test_tiny_final_fragment_is_not_inferred():
    buf = ClipBuffer()
    for i in range(9):
        buf.append(packet(i * 40))
    assert buf.finish() is None


def test_stop_right_after_old_boundary_keeps_context_instead_of_tiny_fragment():
    buf = ClipBuffer()
    image = jpeg()
    for i in range(152):
        buf.append(packet(i * 40, image))
    final = buf.finish()
    assert final.final is True
    assert (final.start_ms, final.end_ms, len(final.frames)) == (40, 6040, 151)


def test_warmup_and_update_cadence_use_capture_time_not_arrival_count():
    buf = ClipBuffer()
    image = jpeg()
    origin = 123.45
    clips = [clip for i in range(222)
             if (clip := buf.append(packet(origin + i * 33, image))) is not None]
    assert clips[0].end_ms == pytest.approx(origin + 2013)
    assert all(clip.sample_origin == origin for clip in clips)
    assert len(clips) == 6
    for expected, clip in zip(range(2000, 8000, 1000), clips):
        assert origin + expected <= clip.end_ms < origin + expected + 33
    assert clips[-1].frames[0].timestamp <= clips[-1].start_ms
    assert clips[-1].frames[1].timestamp > clips[-1].start_ms


def test_a_late_first_window_cannot_silently_drop_the_beginning():
    frames = (Frame(1000, b''), Frame(7000, b''))
    with pytest.raises(ValueError, match='fell too far behind'):
        ensure_overlap(None, Clip(5, frames, False, sample_origin=0, window_start=1000))
    ensure_overlap(None, Clip(0, frames, False, sample_origin=1000, window_start=1000))


def test_latest_candidate_requires_enough_context_with_last_admitted_window():
    old = Clip(0, (Frame(0, b''), Frame(2000, b'')), False)
    ensure_overlap(old, Clip(1, (Frame(0, b''), Frame(3000, b'')), False))
    ensure_overlap(old, Clip(5, (Frame(1000, b''), Frame(7000, b'')), False))
    with pytest.raises(ValueError, match='fell too far behind'):
        ensure_overlap(old, Clip(6, (Frame(1040, b''), Frame(7040, b'')), False))


def test_final_copy_or_short_initial_clip_does_not_fail_overlap_guard():
    old = Clip(0, (Frame(0, b''), Frame(760, b'')), False)
    ensure_overlap(old, Clip(1, old.frames, True))
    ensure_overlap(None, Clip(0, old.frames, True, sample_origin=0))

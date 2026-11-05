"""Bounded overlapping video windows on one continuous capture-time sampling grid."""
from dataclasses import dataclass, field
from collections import OrderedDict
from io import BytesIO
import math
import struct
import time

import numpy as np
from PIL import Image, UnidentifiedImageError

FPS = 25
SEGMENT_MS = 6000
PARTIAL_MS = 1000
INITIAL_MS = 2000
MIN_OVERLAP_MS = 1000
MAX_PACKET = 512_000
# Six seconds at the maximum admitted 40 fps, with a sample on either side of the
# interval boundary. The duration bound normally retains at most 242 frames.
MAX_FRAMES = 244


@dataclass(frozen=True)
class Frame:
    timestamp: float
    jpeg: bytes


@dataclass(frozen=True)
class Clip:
    segment_id: int
    frames: tuple[Frame, ...]
    final: bool
    queued_at: float = field(default_factory=time.perf_counter, compare=False, repr=False)
    sample_origin: float | None = None
    window_start: float | None = None

    @property
    def start_ms(self) -> float:
        return self.window_start if self.window_start is not None else self.frames[0].timestamp

    @property
    def end_ms(self) -> float:
        return self.frames[-1].timestamp


def ensure_overlap(previous: Clip | None, candidate: Clip,
                   min_overlap_ms: float = MIN_OVERLAP_MS) -> None:
    """Reject a latest-only update that has already lost unprocessed context.

    Compare with the last admitted/in-flight window. Skipping intermediate
    requests is safe only while their video remains in the next candidate.
    The first admitted window must still cover the beginning of the session.
    This guards coverage; it does not promise word accuracy or alignment.
    """
    if previous is None:
        if candidate.sample_origin is None:
            return  # Legacy/offline clips do not carry a session origin.
        latest_start = candidate.sample_origin
    else:
        overlap = min(min_overlap_ms, previous.end_ms - previous.start_ms)
        latest_start = previous.end_ms - overlap
    if candidate.start_ms > latest_start + 1e-6:
        raise ValueError('Inference fell too far behind to preserve overlapping video. Stop and restart reading.')


class DecodeCache:
    """Session-owned, bounded reuse of immutable decoded camera frames."""

    def __init__(self, max_frames: int = MAX_FRAMES):
        if max_frames < 1:
            raise ValueError('max_frames must be positive')
        self.max_frames = max_frames
        self._frames = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, frame: Frame):
        entry = self._frames.get(id(frame))
        if entry is not None and entry[0] is frame:
            self._frames.move_to_end(id(frame))
            self.hits += 1
            return entry[1]
        self.misses += 1
        return None

    def put(self, frame: Frame, rgb: np.ndarray):
        rgb.setflags(write=False)
        self._frames[id(frame)] = (frame, rgb)
        self._frames.move_to_end(id(frame))
        while len(self._frames) > self.max_frames:
            self._frames.popitem(last=False)

    def clear(self):
        self._frames.clear()


class PreparationSampler:
    """Offer only source frames selected by the unchanged 25 fps session grid.

    A newly arrived frame closes the nearest-neighbor interval after its
    predecessor. The first and last new grid points suffice: selection can
    switch from predecessor to current frame only once within that interval.
    """
    def __init__(self):
        self.previous = None
        self.origin = None
        self.next_index = 1
        self.last_offered = None

    def append(self, frame):
        if self.previous is None:
            self.previous = frame
            self.origin = self.last_offered = frame.timestamp
            return (frame,)
        last_index = math.floor((frame.timestamp - self.origin) / (1000 / FPS) + 1e-9)
        selected = []
        if last_index >= self.next_index:
            for index in (self.next_index, last_index):
                target = self.origin + index * (1000 / FPS)
                source = (self.previous if abs(self.previous.timestamp-target) <= abs(frame.timestamp-target)
                          else frame)
                if source.timestamp > self.last_offered:
                    selected.append(source)
                    self.last_offered = source.timestamp
            self.next_index = last_index + 1
        self.previous = frame
        return tuple(selected)


def parse_packet(packet: bytes, previous: float | None) -> Frame:
    if not 10 <= len(packet) <= MAX_PACKET:
        raise ValueError('Invalid camera frame size.')
    timestamp = struct.unpack('<d', packet[:8])[0]
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError('Invalid camera timestamp.')
    if previous is not None and timestamp <= previous:
        raise ValueError('Camera timestamps must increase.')
    if not packet[8:].startswith(b'\xff\xd8'):
        raise ValueError('Camera frames must be JPEG images.')
    return Frame(timestamp, packet[8:])


def sample_times(clip: Clip, fps: int = FPS) -> np.ndarray:
    """Exact capture timestamps of the shared, session-aligned sampling grid."""
    if not clip.frames or not math.isfinite(fps) or fps <= 0:
        raise ValueError('A nonempty clip and positive sample rate are required.')
    interval = 1000 / fps
    origin = clip.frames[0].timestamp if clip.sample_origin is None else clip.sample_origin
    first = math.ceil((clip.start_ms - origin) / interval - 1e-9)
    last = math.floor((clip.end_ms - origin) / interval + 1e-9)
    return origin + np.arange(first, last + 1) * interval


def decode_frame(source: Frame, cache: DecodeCache | None = None) -> np.ndarray:
    """One exact JPEG decode, shared by early preparation and window resampling."""
    frame = cache.get(source) if cache is not None else None
    if frame is None:
        try:
            with Image.open(BytesIO(source.jpeg)) as im:
                if im.format != 'JPEG' or im.width > 1280 or im.height > 960 or im.width < 96 or im.height < 96:
                    raise ValueError('Camera image dimensions must be between 96×96 and 1280×960.')
                frame = np.asarray(im.convert('RGB'))
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError('Could not decode a camera frame.') from exc
        if cache is not None:
            cache.put(source, frame)
    return frame


def resample(clip: Clip, fps: int = FPS, cache: DecodeCache | None = None) -> tuple[list[np.ndarray], float]:
    """Nearest-neighbor sampling on capture timestamps, not arrival timestamps."""
    if len(clip.frames) < 2:
        raise ValueError('Capture a little more video before decoding.')
    timestamps = np.array([f.timestamp for f in clip.frames])
    gaps = np.diff(timestamps)
    duration = (timestamps[-1] - timestamps[0]) / 1000
    observed_fps = (len(timestamps) - 1) / duration
    if max(gaps) > 250 or observed_fps < 15:
        raise ValueError('Camera frames are arriving too slowly. Improve lighting or close other camera apps.')
    # Moving the window must not change the 25 fps phase for shared camera frames.
    # Keep the legacy first-frame origin for independently supplied offline clips.
    target = sample_times(clip, fps)
    if not len(target):
        raise ValueError('Capture a little more video before decoding.')
    right = np.searchsorted(timestamps, target).clip(0, len(timestamps) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(timestamps[left] - target) <= abs(timestamps[right] - target), left, right)
    decoded = {}
    shape = None
    for index in np.unique(indices):
        source = clip.frames[index]
        frame = decode_frame(source, cache)
        if shape is not None and frame.shape != shape:
            raise ValueError('Camera resolution changed. Restart the session.')
        shape = frame.shape
        decoded[index] = frame
    return [decoded[i] for i in indices], observed_fps


class ClipBuffer:
    def __init__(self):
        self.frames: list[Frame] = []
        self.segment_id = 0
        self.previous: float | None = None
        self.origin: float | None = None
        self.next_update: float | None = None

    def _snapshot(self, final: bool) -> Clip:
        clip = Clip(self.segment_id, tuple(self.frames), final,
                    sample_origin=self.origin,
                    window_start=max(self.origin, self.frames[-1].timestamp - SEGMENT_MS))
        self.segment_id += 1
        return clip

    def append(self, packet: bytes) -> Clip | None:
        frame = parse_packet(packet, self.previous)
        if self.previous is not None and frame.timestamp - self.previous < 25:
            return None  # Limit input rate to at most 40 fps.
        self.previous = frame.timestamp
        if self.origin is None:
            self.origin = frame.timestamp
            self.next_update = self.origin + INITIAL_MS
        self.frames.append(frame)
        cutoff = frame.timestamp - SEGMENT_MS
        # Retain the nearest frame preceding the window for timestamp resampling.
        # Snapshots share immutable Frame objects rather than copying their JPEGs.
        trim = 0
        while trim + 1 < len(self.frames) and self.frames[trim + 1].timestamp <= cutoff:
            trim += 1
        if trim:
            del self.frames[:trim]
        if len(self.frames) > MAX_FRAMES:
            raise ValueError('Camera history exceeded its bounded frame capacity.')
        if frame.timestamp >= self.next_update:
            skipped = math.floor((frame.timestamp - self.next_update) / PARTIAL_MS) + 1
            self.next_update += skipped * PARTIAL_MS
            return self._snapshot(False)
        return None

    def finish(self) -> Clip | None:
        # Stop flushes the whole retained context, including the previous window's
        # tail, instead of an isolated fragment after a six-second reset.
        clip = self._snapshot(True) if len(self.frames) >= 10 else None
        self.frames = []
        self.origin = None
        self.next_update = None
        self.previous = None
        return clip

"""Bounded video history and whole-utterance caption replacement.

An active utterance has one provisional hypothesis. Its final decode replaces
that hypothesis completely before it becomes stable; no guessed word times or
text-overlap anchors can make an early word irreversible.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import deque
import math

from backend.stream import Clip, Frame
from backend.limits import (MAX_UTTERANCE_MS, HISTORY_RETENTION_MS, MAX_HISTORY_FRAMES,
                            MAX_HISTORY_BYTES)


class UtteranceTranscript:
    """Completed utterances plus one replaceable draft, owned by one session."""

    def __init__(self):
        self._completed: list[str] = []
        self._draft = ''
        self._active = False
        self._revision = 0
        self._final = False

    @property
    def active(self) -> bool:
        return self._active

    @staticmethod
    def _text(text: str) -> str:
        if not isinstance(text, str):
            raise ValueError('Transcript must be text.')
        return ' '.join(text.split())

    def snapshot(self) -> dict:
        stable = ' '.join(self._completed)
        return {'text': ' '.join(part for part in (stable, self._draft) if part),
                'stable_text': stable, 'draft_text': self._draft,
                'utterances': list(self._completed),
                'revision': self._revision, 'final': self._final}

    def revise(self, text: str) -> dict:
        """Open an utterance or replace its complete provisional hypothesis."""
        cleaned = self._text(text)
        if self._final:
            return self.snapshot()
        self._draft = cleaned
        self._active = True
        self._revision += 1
        return self.snapshot()

    def commit(self, text: str | None = None) -> dict:
        """Finalize the current utterance once, optionally replacing its draft.

        ``None`` keeps the most recent preview, useful after a failed final pass.
        An explicit empty string is a valid final decode that removes the draft.
        Call ``revise('')`` to open an utterance before any preview is available.
        """
        cleaned = self._text(text) if text is not None else None
        if self._final or not self._active:
            return self.snapshot()
        if cleaned is not None:
            self._draft = cleaned
        if self._draft:
            self._completed.append(self._draft)
        self._draft = ''
        self._active = False
        self._revision += 1
        return self.snapshot()

    def hold(self) -> dict:
        """Retain text while waiting for speech or final-refinement context."""
        if not self._final:
            self._revision += 1
        return self.snapshot()

    def finish(self) -> dict:
        """End the session without inventing or repeating an utterance."""
        if not self._final:
            if self._active:
                if self._draft:
                    self._completed.append(self._draft)
                self._draft = ''
                self._active = False
            self._final = True
            self._revision += 1
        return self.snapshot()


class VideoHistory:
    """Immutable source frames retained for a bounded utterance final pass.

    This shares ``Frame`` objects with the rolling activity buffer. Capacity
    pressure evicts old frames, and ``slice`` refuses to silently substitute a
    shorter interval if a requested utterance has fallen out of retention.
    """

    def __init__(self, *, retention_ms: float = HISTORY_RETENTION_MS, max_frames: int = MAX_HISTORY_FRAMES,
                 max_bytes: int = MAX_HISTORY_BYTES, max_clip_ms: float = MAX_UTTERANCE_MS):
        if (not math.isfinite(retention_ms) or not 0 < retention_ms <= HISTORY_RETENTION_MS
                or type(max_frames) is not int or not 2 <= max_frames <= MAX_HISTORY_FRAMES
                or type(max_bytes) is not int or not 1 <= max_bytes <= MAX_HISTORY_BYTES
                or not math.isfinite(max_clip_ms) or not 0 < max_clip_ms <= MAX_UTTERANCE_MS
                or max_clip_ms > retention_ms):
            raise ValueError('Invalid video history bounds.')
        self.retention_ms = retention_ms
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self.max_clip_ms = max_clip_ms
        self.frames: deque[Frame] = deque()
        self.origin: float | None = None
        self.total_bytes = 0

    @property
    def oldest_ms(self) -> float | None:
        return self.frames[0].timestamp if self.frames else None

    @property
    def latest_ms(self) -> float | None:
        return self.frames[-1].timestamp if self.frames else None

    def clear(self) -> None:
        self.frames.clear()
        self.origin = None
        self.total_bytes = 0

    def _discard_first(self) -> None:
        self.total_bytes -= len(self.frames.popleft().jpeg)

    def append(self, frame: Frame) -> None:
        if not isinstance(frame, Frame) or not math.isfinite(frame.timestamp) or frame.timestamp < 0:
            raise ValueError('Video history needs a valid camera frame.')
        if self.frames:
            previous = self.frames[-1]
            if frame is previous or frame.timestamp == previous.timestamp:
                return  # The camera rate limiter can return its prior last frame.
            if frame.timestamp < previous.timestamp:
                raise ValueError('Video history timestamps must increase.')
        if len(frame.jpeg) > self.max_bytes:
            raise ValueError('A camera frame exceeds the video history capacity.')
        if self.origin is None:
            self.origin = frame.timestamp
        self.frames.append(frame)
        self.total_bytes += len(frame.jpeg)
        cutoff = frame.timestamp - self.retention_ms
        # Keep the nearest predecessor of the retained interval for resampling.
        while len(self.frames) > 1 and self.frames[1].timestamp <= cutoff:
            self._discard_first()
        while len(self.frames) > self.max_frames or self.total_bytes > self.max_bytes:
            self._discard_first()

    def slice(self, start_ms: float, end_ms: float, segment_id: int,
              final: bool = False) -> Clip:
        if (not math.isfinite(start_ms) or not math.isfinite(end_ms)
                or start_ms < 0 or end_ms <= start_ms):
            raise ValueError('Invalid utterance interval.')
        if end_ms - start_ms > self.max_clip_ms + 1e-6:
            raise ValueError('The utterance exceeds the bounded final-refinement duration.')
        if not self.frames:
            raise ValueError('No camera history is available for this utterance.')
        if start_ms < self.frames[0].timestamp - 1e-6:
            raise ValueError('The beginning of this utterance is no longer in camera history.')
        if end_ms > self.frames[-1].timestamp + 1e-6:
            raise ValueError('The end of this utterance has not arrived yet.')
        frames = tuple(self.frames)
        times = [frame.timestamp for frame in frames]
        first = max(0, bisect_right(times, start_ms) - 1)
        last = bisect_right(times, end_ms)
        selected = frames[first:last]
        if len(selected) < 2 or selected[-1].timestamp <= start_ms:
            raise ValueError('Capture a little more video before refining this utterance.')
        return Clip(segment_id, selected, final, sample_origin=self.origin,
                    window_start=start_ms)

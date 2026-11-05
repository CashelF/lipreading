"""Session-local decisions from the frozen Light-ASD visual classifier.

The original activity threshold was chosen using first-take recordings only.
Separate start-confirmation metrics implement a stricter trial rule without
changing that activity decision. Both saved takes informed the trial review;
they are regression examples, not held-out validation of the new rule.
Inputs are raw speaking scores and their
absolute timestamps on the app's 25 fps sample grid.
The gate consumes each fully contextualized score once, including all evidence
between coalesced updates. It never changes FLARE's clip or decoding settings.

Twelve frames at either edge lack the combined face-crop and encoder context.
The initial left edge and the final right edge therefore cannot be classified
reliably by this policy. A short or unfinished observation abstains; inventing
future frames or interpreting uncertainty as speech would hide that limitation.
"""
from __future__ import annotations

from collections import deque

import numpy as np


CHECKPOINT = 'talkset'
SCORE_THRESHOLD = .4
START_SCORE_THRESHOLD = .8
START_CONFIRM_FRAMES = 3
START_RESET_SILENCE_MS = 800.
MEAN_FRAMES = 25
CONTEXT_FRAMES = 12
FPS = 25
FRAME_INTERVAL_MS = 1000. / FPS
TIMING_TOLERANCE_MS = 1e-3


class _SpeechRegions:
    """Ordered confirmed episodes; observed quiet, never missing time, closes one."""

    def __init__(self):
        self.current = None
        self.strong_run = 0
        self.quiet_frames = 0

    def interrupt(self):
        # A tracking interruption cannot manufacture a pause or erase an
        # already confirmed utterance. Unconfirmed starts need fresh evidence.
        if self.current is not None and self.current['confirmed_ms'] is None:
            self.current = None
        self.strong_run = 0
        self.quiet_frames = 0

    def _touch(self, regions, *, active=False, closed_at=None):
        if regions and regions[-1]['start_ms'] == self.current['start_ms']:
            region = regions[-1]
        else:
            region = {**self.current, 'closed_at_ms': None, 'fresh_active': False}
            regions.append(region)
        region['last_active_ms'] = self.current['last_active_ms']
        region['fresh_active'] = region['fresh_active'] or active
        region['closed_at_ms'] = closed_at

    def update(self, mean, timestamp, regions):
        if mean >= SCORE_THRESHOLD:
            self.quiet_frames = 0
            if self.current is None:
                self.current = {'start_ms': timestamp, 'last_active_ms': timestamp,
                                'confirmed_ms': None}
            self.current['last_active_ms'] = timestamp
        else:
            self.quiet_frames += 1
        self.strong_run = (min(self.strong_run + 1, START_CONFIRM_FRAMES)
                           if mean >= START_SCORE_THRESHOLD else 0)
        if self.current is None:
            return
        if self.strong_run >= START_CONFIRM_FRAMES and self.current['confirmed_ms'] is None:
            self.current['confirmed_ms'] = timestamp
        if self.quiet_frames * FRAME_INTERVAL_MS >= START_RESET_SILENCE_MS:
            if self.current['confirmed_ms'] is not None:
                self._touch(regions, closed_at=timestamp)
            self.current = None
            self.strong_run = 0
            self.quiet_frames = 0
        elif self.current['confirmed_ms'] is not None and mean >= SCORE_THRESHOLD:
            self._touch(regions, active=True)

    def finish_update(self, regions):
        if self.current is not None and self.current['confirmed_ms'] is not None:
            self._touch(regions)


class SpeechGateState:
    """One reading session's already examined evidence and mean-filter history.

``update`` returns an activity payload suitable for the existing transport.
Silence, duplicate evidence and uncertainty all preserve previous captions.
A gap is reported explicitly and establishes a new continuous history, so the
following valid update can recover rather than remaining stuck forever.
"""

    def __init__(self):
        self.history: deque[float] = deque(maxlen=MEAN_FRAMES - 1)
        self.last_timestamp_ms: float | None = None
        self._start_candidate_ms: float | None = None
        self._start_last_active_ms: float | None = None
        self._start_run = 0
        self._regions = _SpeechRegions()

    def _reset_start(self) -> None:
        self._start_candidate_ms = None
        self._start_last_active_ms = None
        self._start_run = 0

    def reset(self) -> None:
        self.history.clear()
        self.last_timestamp_ms = None
        self._reset_start()
        self._regions = _SpeechRegions()

    def abstain(self, detail: str = 'Speaking detection was unavailable. Waiting for a clear, continuous view.') -> dict:
        """Report missing/invalid observations without replaying checked evidence.

        Discard mean-filter history, but do not move or rewind the watermark.
        A later valid overlapping window can examine evidence that was never
        checked; if it cannot cover that interval, ``update`` reports the gap.
        """
        return self._invalid(detail)

    def _activity(self, state: str, detail: str, *, new_frames: int = 0,
                  means: list[float] | None = None, gap: bool = False,
                  invalid: bool = False,
                  start_confirmation: tuple[float, float] | None = None,
                  regions: list[dict] | None = None) -> dict:
        metrics = {
            'speech_score_threshold': SCORE_THRESHOLD,
            'speech_mean_frames': MEAN_FRAMES,
            'speech_context_frames': CONTEXT_FRAMES,
            'speech_new_frames': new_frames,
            'speech_mean_count': len(means) if means is not None else 0,
            'speech_history_frames': len(self.history),
            'speech_gap': int(gap),
            'speech_invalid': int(invalid),
            'speech_start_threshold': START_SCORE_THRESHOLD,
            'speech_start_confirm_frames': START_CONFIRM_FRAMES,
            'speech_start_confirmed': int(start_confirmation is not None),
            'speech_regions_version': 1,
        }
        # A confirmation belongs to this update's fresh evidence only. Preserve
        # its original low-threshold onset even if a later cluster in the same
        # coalesced update has replaced the internal pending candidate.
        candidate = self._start_candidate_ms
        if start_confirmation is not None:
            candidate, confirmed_at = start_confirmation
            metrics['speech_start_confirmed_ms'] = confirmed_at
        if candidate is not None:
            metrics['speech_start_candidate_ms'] = candidate
        if means:
            metrics['speech_max_mean'] = round(max(means), 6)
            metrics['speech_latest_mean'] = round(means[-1], 6)
        if self.last_timestamp_ms is not None:
            metrics['speech_evidence_end_ms'] = self.last_timestamp_ms
        return {'state': state, 'suppress': state != 'motion',
                'preserve_transcript': state != 'motion',
                'detail': detail, 'metrics': metrics, 'speech_regions': regions or []}

    def _invalid(self, detail: str) -> dict:
        # Keep the watermark: invalid input must not allow an old speech event
        # to be replayed when a duplicate window is subsequently retried.
        self.history.clear()
        self._reset_start()
        self._regions.interrupt()
        return self._activity('uncertain', detail, invalid=True)

    def update(self, scores, timestamps_ms) -> dict:
        """Consume new valid evidence, returning motion/still/uncertain.

Timestamps must describe every supplied score at 40 ms intervals. They are
sampling times, not JPEG arrival times or the fractional source-window start.
No fixed last-second truncation is used: all new evidence is checked so a short
utterance between coalesced requests is not skipped.
"""
        try:
            scores = np.asarray(scores, dtype=np.float64)
            timestamps = np.asarray(timestamps_ms, dtype=np.float64)
        except (TypeError, ValueError):
            return self._invalid('Speaking detection was unavailable. Waiting for a clear, continuous view.')
        if (scores.ndim != 1 or timestamps.ndim != 1 or len(scores) != len(timestamps)
                or not len(scores) or not np.isfinite(scores).all()
                or np.any(scores < 0) or np.any(scores > 1)
                or not np.isfinite(timestamps).all() or np.any(timestamps < 0)):
            return self._invalid('Speaking detection was unavailable. Waiting for a clear, continuous view.')
        if not np.all(np.abs(np.diff(timestamps) - FRAME_INTERVAL_MS) <= TIMING_TOLERANCE_MS):
            return self._invalid('Camera sampling was interrupted. Waiting for continuous video.')
        if len(scores) <= 2 * CONTEXT_FRAMES:
            return self._activity('uncertain', 'Keep your face in view a little longer to check for speech.')

        mature_scores = scores[CONTEXT_FRAMES:-CONTEXT_FRAMES]
        mature_times = timestamps[CONTEXT_FRAMES:-CONTEXT_FRAMES]
        if self.last_timestamp_ms is not None:
            fresh = mature_times > self.last_timestamp_ms + TIMING_TOLERANCE_MS
            mature_scores, mature_times = mature_scores[fresh], mature_times[fresh]
        if not len(mature_scores):
            return self._activity('still', 'Waiting for new visual speech; previous words are kept.')

        gap = (self.last_timestamp_ms is not None
               and abs(float(mature_times[0]) - self.last_timestamp_ms - FRAME_INTERVAL_MS)
               > TIMING_TOLERANCE_MS)
        if gap:
            self.history.clear()
            self._reset_start()
            self._regions.interrupt()
        means = []
        active_times = []
        start_confirmation = None
        regions = []
        for score, timestamp in zip(mature_scores, mature_times):
            value = float(score)
            if len(self.history) == MEAN_FRAMES - 1:
                means.append((sum(self.history) + value) / MEAN_FRAMES)
                mean, current_time = means[-1], float(timestamp)
                if not gap:
                    self._regions.update(mean, current_time, regions)
                if mean >= SCORE_THRESHOLD:
                    active_times.append(current_time)
                    if self._start_candidate_ms is None:
                        self._start_candidate_ms = current_time
                    self._start_last_active_ms = current_time
                elif (self._start_last_active_ms is not None
                      and current_time - self._start_last_active_ms
                      >= START_RESET_SILENCE_MS):
                    self._reset_start()
                if mean >= START_SCORE_THRESHOLD:
                    self._start_run = min(self._start_run + 1, START_CONFIRM_FRAMES)
                    if (self._start_run >= START_CONFIRM_FRAMES
                            and start_confirmation is None and not gap):
                        start_confirmation = (self._start_candidate_ms, current_time)
                else:
                    self._start_run = 0
            self.history.append(value)
            self.last_timestamp_ms = float(timestamp)
        if gap:
            return self._activity('uncertain', 'Some camera evidence was missed. Waiting for continuous video.',
                                  new_frames=len(mature_scores), means=means, gap=True)
        if not means:
            return self._activity('uncertain', 'Gathering enough visual context to check for speech.',
                                  new_frames=len(mature_scores))
        self._regions.finish_update(regions)
        if max(means) >= SCORE_THRESHOLD:
            result = self._activity('motion', 'Visual speech detected.',
                                    new_frames=len(mature_scores), means=means,
                                    start_confirmation=start_confirmation, regions=regions)
            result['metrics'].update(speech_first_active_ms=active_times[0],
                                     speech_last_active_ms=active_times[-1])
            return result
        return self._activity('still', 'Waiting for speech; previous words are kept.',
                              new_frames=len(mature_scores), means=means, regions=regions)

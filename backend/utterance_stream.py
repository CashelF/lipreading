"""Gate rolling video, but recognize and finalize whole bounded utterances."""
from __future__ import annotations

import asyncio
import copy
import time

from backend.preprocessing import FrameCache, PreprocessCache
from backend.stream import DecodeCache, decode_frame, resample, sample_times
from backend.utterance import UtteranceTranscript, VideoHistory
from backend.limits import (DEFAULT_UTTERANCE_SECONDS, MAX_PREPROCESS_FRAMES,
                            UTTERANCE_OPTIONS)

PRE_ROLL_MS = 1600.
END_CONTEXT_MS = 500.
END_SILENCE_MS = 800.


class UtteranceLimitError(RuntimeError):
    """The full utterance cannot safely be finalized within the input bound."""


class UtteranceReader:
    """One session; only previously detected speech authorizes final decoding.

    The speech classifier's timestamps are evidence times, not word alignments.
    Full utterance hypotheses replace one another until final refinement. A
    bounded full-utterance limit caps inference. Exceeding it stops the session with
    an explicit message and retains the latest draft; no arbitrary word splice
    is substituted for a full-utterance result.
    """

    def __init__(self, model):
        self.model = model
        self.history = VideoHistory()
        self.transcript = UtteranceTranscript()
        # Reuse immutable pixels, detector observations and exactly keyed ROIs.
        # Each consumer still owns independent timestamps and speech state.
        frames = FrameCache(max_frames=MAX_PREPROCESS_FRAMES)
        self.gate_decode = self.utterance_decode = DecodeCache(max_frames=MAX_PREPROCESS_FRAMES)
        self.gate_preprocess = PreprocessCache(frames=frames)
        self.utterance_preprocess = PreprocessCache(frames=frames)
        self.prefetch_preprocess = PreprocessCache(frames=frames)
        self.start_ms = None
        self.last_speech_ms = None
        self.last_gate_end_ms = None
        self.committed_end_ms = None
        self.verify_visual = True
        self.max_utterance_seconds = DEFAULT_UTTERANCE_SECONDS
        self._regions_interrupted = False
        self._refinement_warning = None
        self._frame_decode_ms = 0.
        self._last_decode_key = None
        self._last_decode_result = None

    def prepare_frame(self, source, speech_gate):
        frame = decode_frame(source, self.gate_decode)
        self.model.prepare_frame(frame, cache=self.prefetch_preprocess, speech_gate=speech_gate)

    def clear(self):
        self.history.clear()
        self.gate_decode.clear()
        self.gate_preprocess.clear()
        self.utterance_decode.clear()
        self.utterance_preprocess.clear()
        self.prefetch_preprocess.clear()
        self._last_decode_key = self._last_decode_result = None

    def set_utterance_limit(self, seconds):
        if type(seconds) is not int or seconds not in UTTERANCE_OPTIONS:
            raise ValueError('Utterance length must be 30 or 60 seconds.')
        if self.start_ms is not None:
            raise ValueError('Stop reading before changing the utterance length.')
        self.max_utterance_seconds = seconds

    async def _decode(self, start, end, clip, final, send):
        try:
            source = self.history.slice(start, end, clip.segment_id, final=final)
        except ValueError as exc:
            raise UtteranceLimitError('Video context was lost before final refinement. Your latest draft was kept; start a new session.') from exc
        await send({'type': 'utterance', 'state': 'refining' if final else 'draft',
                    'detail': 'Refining the completed utterance…' if final
                    else 'Live draft · the whole utterance may change until you pause.'})
        # Exact same source, sample phase, bounds and policy: e.g. Stop immediately
        # after the last preview. Do not search and verify identical video twice.
        # Absolute monotonic timestamps prevent object-ID reuse across old frames;
        # the key retains no additional camera images beyond the bounded caches.
        key = (id(self.model), getattr(self.model, 'beam', None),
               source.start_ms, source.end_ms, source.sample_origin,
               tuple((frame.timestamp, id(frame)) for frame in source.frames), self.verify_visual)
        can_reuse = bool(getattr(self.model, 'supports_exact_result_reuse', False))
        if can_reuse and key == self._last_decode_key:
            result = copy.deepcopy(self._last_decode_result)
            result['timings_ms'] = {'preprocessing': 0., 'upload': 0., 'inference': 0.,
                                    'grounding': 0., 'recognition_reused': 1}
            result['latency_ms'] = 0
            return result
        decode_started = time.perf_counter()
        frames, _ = await asyncio.to_thread(resample, source, cache=self.utterance_decode)
        self._frame_decode_ms += (time.perf_counter() - decode_started) * 1000
        decode = self.model.refine_utterance if final else self.model.decode_utterance
        options = {'cache': self.utterance_preprocess}
        if getattr(self.model, 'supports_visual_grounding', False):
            options['verify_visual'] = self.verify_visual
        result = await asyncio.to_thread(decode, frames, 25, **options)
        if can_reuse:
            self._last_decode_key, self._last_decode_result = key, copy.deepcopy(result)
        return result

    def _accept_recognition(self, result, final, timings):
        verification = result.get('recognition')
        if verification is not None and not verification.get('accepted'):
            # A failed replacement must not erase an earlier supported draft.
            # It also cannot introduce text into an empty/new utterance.
            self._refinement_warning = verification['detail']
            timings['grounding_withheld'] = timings.get('grounding_withheld', 0) + 1
            if final:
                self.transcript.commit()
        elif final:
            self.transcript.commit(result['text'])
        else:
            self.transcript.revise(result['text'])

    async def _process_regions(self, clip, inspection, send):
        """Consume ordered episodes so a pause survives request coalescing."""
        activity = inspection['activity']
        metrics = activity['metrics']
        timings = {**inspection.get('timings_ms', {}), 'inference': 0., 'upload': 0.}
        intervals = []
        refinements = 0
        valid = (not metrics.get('speech_invalid') and not metrics.get('speech_gap')
                 and activity.get('state') != 'uncertain')
        if not valid:
            self._regions_interrupted = True
            if clip.final and self.start_ms is not None:
                self._refinement_warning = 'Tracking was interrupted. Kept the last draft without final refinement.'
            return activity, timings, refinements, intervals
        regions = activity.get('speech_regions', [])
        evidence_end = metrics.get('speech_evidence_end_ms', clip.end_ms - 480.)

        async def recognize(final, *, closed=False):
            nonlocal refinements
            # Keep the full right edge when speech reaches the mature frontier:
            # capture timestamps can lie almost one sample beyond that grid.
            trailing_quiet = self.last_speech_ms < evidence_end
            end = (min(clip.end_ms, self.last_speech_ms + END_CONTEXT_MS)
                   if closed or trailing_quiet else clip.end_ms)
            if end - self.start_ms > self.max_utterance_seconds * 1000:
                raise UtteranceLimitError(f'This utterance exceeded {self.max_utterance_seconds} seconds. Your latest draft was kept. Start again and pause briefly between phrases.')
            if end - self.start_ms >= 800.:
                try:
                    result = await self._decode(self.start_ms, end, clip, final, send)
                except ValueError as exc:
                    if final:
                        # The gate already consumed this region's closure. A
                        # recoverable quality message would lose that boundary
                        # and let a later phrase attach to the unfinished draft.
                        raise UtteranceLimitError('Final refinement failed. Your latest draft was kept; start a new session.') from exc
                    raise
                intervals.append({'start_ms': self.start_ms, 'end_ms': end,
                                  'last_active_ms': self.last_speech_ms, 'final': final})
                for name, value in result.get('timings_ms', {}).items():
                    timings[name] = timings.get(name, 0.) + value
                if final:
                    refinements += 1
                self._accept_recognition(result, final, timings)
            elif final:
                self.transcript.commit()
            if final:
                self.committed_end_ms = end
                self.start_ms = None
                self.last_speech_ms = None
                await send({'type': 'utterance', 'state': 'committed',
                            'detail': self._refinement_warning or 'Utterance refined. Waiting for your next words.'})

        for region in regions:
            if (self.committed_end_ms is not None
                    and region['last_active_ms'] <= self.committed_end_ms):
                continue  # A duplicate region cannot reopen completed speech.
            if region['fresh_active']:
                self._regions_interrupted = False
            if self.start_ms is None:
                self.start_ms = max(self.history.origin, region['start_ms'] - PRE_ROLL_MS,
                                    self.committed_end_ms or self.history.origin)
                self.transcript.revise('')
            self.last_speech_ms = max(self.last_speech_ms or region['last_active_ms'],
                                      region['last_active_ms'])
            closed = region['closed_at_ms'] is not None
            if (closed or clip.final) and self._regions_interrupted:
                # Recovered quiet proves a pause now, not where speech stopped
                # during the missing observation. Do not trim the unknown span
                # to an old endpoint and present a purported full refinement.
                self.transcript.commit()
                # This marks only A's supported context, not the later quiet
                # confirmation time. That time can lie inside B's early mouth
                # movements because the detector is smoothed; B needs pre-roll.
                self.committed_end_ms = min(clip.end_ms, self.last_speech_ms + END_CONTEXT_MS)
                self.start_ms = None
                self.last_speech_ms = None
                self._refinement_warning = 'Tracking was interrupted. Kept the last draft without final refinement.'
                timings['preserved_drafts'] = timings.get('preserved_drafts', 0) + 1
                await send({'type': 'utterance', 'state': 'committed',
                            'detail': self._refinement_warning})
                continue
            if closed or clip.final or region['fresh_active']:
                await recognize(closed or clip.final, closed=closed)

        if clip.final and self.start_ms is not None and not self._regions_interrupted:
            # Stop can share the last camera frame; no fresh region is expected.
            await recognize(True)
        elif clip.final and self.start_ms is not None:
            self._refinement_warning = 'Tracking was interrupted. Kept the last draft without final refinement.'
        if self.start_ms is None and not activity.get('suppress') and not regions:
            activity = {**activity, 'state': 'uncertain', 'suppress': True,
                        'preserve_transcript': True,
                        'detail': 'Waiting for clearer speech before starting a new phrase.'}
        return activity, timings, refinements, intervals

    def _result(self, clip, inspection, activity, timings, refinements, capture_fps,
                begin, intervals=None):
        if clip.final:
            snapshot = self.transcript.finish()
            self.start_ms = None
        else:
            snapshot = self.transcript.hold()
        timings['refinement_passes'] = refinements
        timings['frame_decode'] = round(self._frame_decode_ms, 1)
        timings['activity_preprocessing'] = inspection.get('timings_ms', {}).get('preprocessing', 0.)
        timings['recognition_preprocessing'] = round(max(0., timings.get('preprocessing', 0.)
                                                    - timings['activity_preprocessing']), 1)
        return {'text': snapshot['text'], 'snapshot': snapshot, 'activity': activity,
                'quality': inspection.get('quality', {}),
                'detail': self._refinement_warning or f'{capture_fps:.1f} fps captured · full utterance refinement enabled',
                'timings_ms': timings, 'decoded_intervals': intervals or [],
                'latency_ms': round((time.perf_counter() - begin) * 1000)}

    async def process(self, clip, speech_gate, send):
        begin = time.perf_counter()
        self._refinement_warning = None
        # Reconstruct any skipped activity interval from retained source frames.
        # A slow full-utterance decode must not discard speech between requests.
        gate_clip = clip
        needed_start = (self.history.origin if self.last_gate_end_ms is None else
                        max(self.history.origin, self.last_gate_end_ms - 1000.))
        if needed_start < clip.start_ms:
            try:
                gate_clip = self.history.slice(needed_start, clip.end_ms, clip.segment_id)
            except ValueError as exc:
                raise UtteranceLimitError('Processing fell behind the retained video. Your latest draft was kept; start a new session.') from exc
        decode_started = time.perf_counter()
        frames, capture_fps = await asyncio.to_thread(resample, gate_clip, cache=self.gate_decode)
        self._frame_decode_ms = (time.perf_counter() - decode_started) * 1000
        self.gate_preprocess.set_sample_times(sample_times(gate_clip))
        inspection = await asyncio.to_thread(self.model.inspect_activity, frames, 25,
                                             cache=self.gate_preprocess, speech_gate=speech_gate)
        self.last_gate_end_ms = gate_clip.end_ms
        activity = inspection['activity']
        metrics = activity.get('metrics', {})
        if speech_gate and metrics.get('speech_regions_version') == 1:
            activity, timings, refinements, intervals = await self._process_regions(clip, inspection, send)
            return self._result(clip, inspection, activity, timings, refinements,
                                capture_fps, begin, intervals)
        detected = not activity.get('suppress', False)
        evidence_end = metrics.get('speech_evidence_end_ms', clip.end_ms - 480.)
        # A permissive continuation score alone must not authorize a fresh
        # transcription. The trained detector confirms starts using consecutive
        # stronger scores, while retaining the original low-threshold onset.
        # Legacy model adapters without start metadata keep their old contract.
        needs_confirmation = speech_gate and 'speech_start_threshold' in metrics
        confirmed = (not needs_confirmation or bool(metrics.get('speech_start_confirmed')))
        valid_evidence = not metrics.get('speech_invalid') and not metrics.get('speech_gap')
        if self.start_ms is None and detected and (not confirmed or not valid_evidence):
            detected = False
            activity = {**activity, 'state': 'uncertain', 'suppress': True,
                        'preserve_transcript': True,
                        'detail': 'Waiting for clearer speech before starting a new phrase.'}
        if detected:
            first_active = metrics.get('speech_first_active_ms', gate_clip.start_ms + PRE_ROLL_MS)
            last_active = metrics.get('speech_last_active_ms', evidence_end)
            if self.start_ms is None and needs_confirmation:
                first_active = metrics.get('speech_start_candidate_ms', first_active)
            if not speech_gate:
                first_active, last_active = gate_clip.start_ms, clip.end_ms
            if self.start_ms is None:
                self.start_ms = max(self.history.origin, first_active - PRE_ROLL_MS,
                                    self.committed_end_ms or self.history.origin)
                self.transcript.revise('')
            self.last_speech_ms = max(self.last_speech_ms or last_active, last_active)

        timings = {**inspection.get('timings_ms', {}), 'inference': 0., 'upload': 0.}
        refinements = 0
        intervals = []
        # A quiet session cannot enter this block. Stop authorizes only the final
        # pass for an utterance whose speech detector already opened it.
        if self.start_ms is not None:
            # A coalesced interval is marked motion if *any* matured mean was
            # active. Its explicit last-active time also proves every later
            # matured mean was quiet, so a pause need not wait for another clip.
            reliable_pause = (not metrics.get('speech_invalid')
                              and not metrics.get('speech_gap')
                              and (activity.get('state') == 'still'
                                   or (activity.get('state') == 'motion' and detected
                                       and 'speech_last_active_ms' in metrics)))
            ended = (reliable_pause
                     and evidence_end - self.last_speech_ms >= END_SILENCE_MS)
            trailing_quiet = speech_gate and reliable_pause and self.last_speech_ms < evidence_end
            requested_end = (min(clip.end_ms, self.last_speech_ms + END_CONTEXT_MS)
                             if ended or trailing_quiet else clip.end_ms)
            if requested_end - self.start_ms > self.max_utterance_seconds * 1000:
                raise UtteranceLimitError(f'This utterance exceeded {self.max_utterance_seconds} seconds. Your latest draft was kept. Start again and pause briefly between phrases.')
            final_pass = clip.final or ended
            if requested_end - self.start_ms >= 800. and (final_pass or detected):
                result = await self._decode(self.start_ms, requested_end, clip, final_pass, send)
                intervals.append({'start_ms': self.start_ms, 'end_ms': requested_end,
                                  'last_active_ms': self.last_speech_ms, 'final': final_pass})
                for name, value in result.get('timings_ms', {}).items():
                    timings[name] = timings.get(name, 0.) + value
                if final_pass:
                    refinements += 1
                    self._accept_recognition(result, True, timings)
                    self.committed_end_ms = requested_end
                    self.start_ms = None
                    self.last_speech_ms = None
                    await send({'type': 'utterance', 'state': 'committed',
                                'detail': self._refinement_warning or 'Utterance refined. Waiting for your next words.'})
                else:
                    self._accept_recognition(result, False, timings)

        return self._result(clip, inspection, activity, timings, refinements, capture_fps, begin, intervals)

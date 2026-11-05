"""25 fps, stable-face alignment and dual face/lip evaluation transforms.

Lip alignment follows AV-HuBERT's five stable points and 12-frame smoothing.
FLARE does not publish its exact full-face crop recipe; the aligned 256px face
is resized to 96px here. This is an explicit deployment approximation.
"""
from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from backend.limits import MAX_PREPROCESS_FRAMES, HISTORY_SECONDS


class FaceQualityError(ValueError):
    """Capture needs adjustment; no transcript should be generated."""

    def __init__(self, detail, *, camera_hint=None):
        super().__init__(detail)
        self.camera_hint = camera_hint


@dataclass
class Preprocessed:
    face: np.ndarray | None
    lips: np.ndarray | None
    quality: dict
    activity: dict | None = None


@dataclass
class _FrameCacheEntry:
    # A strong reference makes object-id reuse impossible while this entry lives.
    frame: np.ndarray
    landmarks: np.ndarray | None = None
    detected: bool = False
    box: np.ndarray | None = None
    box_detected: bool = False
    brightness: np.float64 | None = None
    # A resampled camera frame can occupy several time positions, each with a
    # different exact alignment. Retain a few variants instead of thrashing a
    # single slot whenever that frame is sampled twice.
    rois: OrderedDict = field(default_factory=OrderedDict)
    speech_crops: OrderedDict = field(default_factory=OrderedDict)


class FrameCache:
    """Bounded immutable observations/ROIs, independent of speech timestamps."""

    def __init__(self, max_frames=400):
        if type(max_frames) is not int or not 1 <= max_frames <= MAX_PREPROCESS_FRAMES:
            raise ValueError(f'Preprocessing cache must hold between 1 and {MAX_PREPROCESS_FRAMES} frames.')
        self.max_frames = max_frames
        self.entries = OrderedDict()
        self.detector = None

    def clear(self):
        self.entries.clear()
        self.detector = None

    def bind(self, detector):
        if self.detector is not detector:
            self.clear()
            self.detector = detector

    def entry(self, frame):
        key = id(frame)
        entry = self.entries.get(key)
        if entry is None or entry.frame is not frame:
            entry = _FrameCacheEntry(frame)
            self.entries[key] = entry
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_frames:
            self.entries.popitem(last=False)
        return entry


class PreprocessCache:
    """Session-local memoization for immutable decoded frames.

    Only detector observations and exactly matching transformed ROIs are reused.
    Interpolation and temporal smoothing still run on the entire current clip,
    including its newly available context. Do not mutate frames while cached.
    """

    def __init__(self, max_frames: int = 400, *, frames: FrameCache | None = None):
        if type(max_frames) is not int or not 1 <= max_frames <= MAX_PREPROCESS_FRAMES:
            raise ValueError(f'Preprocessing cache must hold between 1 and {MAX_PREPROCESS_FRAMES} frames.')
        self.frames = frames if frames is not None else FrameCache(max_frames)
        self.max_frames = self.frames.max_frames
        self._entries = self.frames.entries
        self._detector = None
        self.sample_times_ms: np.ndarray | None = None
        self.speech_state = None
        self.detector_calls = 0
        self.detector_hits = 0
        self.warp_calls = 0
        self.warp_hits = 0

    def clear(self):
        """Release all video/landmark/ROI references; retain cumulative counters."""
        self.frames.clear()
        self._detector = None
        self.sample_times_ms = None
        if self.speech_state is not None:
            self.speech_state.reset()
        self.speech_state = None

    def set_sample_times(self, timestamps_ms):
        """Attach the current window's actual fixed-grid sample times."""
        times = np.asarray(timestamps_ms, dtype=np.float64)
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or np.any(times < 0) or np.any(np.diff(times) <= 0)):
            raise ValueError('Sample timestamps must be finite and strictly increasing.')
        self.sample_times_ms = times.copy()
        self.sample_times_ms.setflags(write=False)

    def stats(self) -> dict:
        return {'entries': len(self._entries), 'detector_calls': self.detector_calls,
                'detector_hits': self.detector_hits, 'warp_calls': self.warp_calls,
                'warp_hits': self.warp_hits}

    def bind(self, detector):
        if self._detector is not detector or self.frames.detector is not detector:
            # The app attaches timestamps before the first preprocessing call.
            # Detector invalidation must not erase that current-window metadata.
            if self.speech_state is not None:
                self.speech_state.reset()
            self.speech_state = None
            self._detector = detector
        self.frames.bind(detector)

    def entry(self, frame: np.ndarray) -> _FrameCacheEntry:
        return self.frames.entry(frame)


def interpolate_landmarks(landmarks: list, max_missing: int = 5) -> np.ndarray:
    valid = [i for i, points in enumerate(landmarks) if points is not None]
    if len(valid) < len(landmarks) * .8:
        raise FaceQualityError('Keep one face fully visible and well lit. If you are farther away, move closer.', camera_hint='face_not_visible')
    gaps = np.diff([-1, *valid, len(landmarks)]) - 1
    if not valid or gaps.max(initial=0) > max_missing:
        raise FaceQualityError('Face tracking was lost. Face the camera and try again.', camera_hint='face_not_visible')
    values = np.stack([landmarks[i] for i in valid])
    positions = np.arange(len(landmarks))
    result = np.empty((len(landmarks), 68, 2), dtype=np.float32)
    for point in range(68):
        for axis in range(2):
            result[:, point, axis] = np.interp(positions, valid, values[:, point, axis])
    return result


def normalize_roi(gray: np.ndarray, mean: float = .421, std: float = .165, crop_size: int = 88) -> np.ndarray:
    import cv2
    resized = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_LINEAR)
    offset = (96 - crop_size) // 2
    return (resized[offset:offset + crop_size, offset:offset + crop_size].astype(np.float32) / 255. - mean) / std


def detection_view(frame):
    """Use the established detector scale, retaining original pixels for FAN.

    SFD detections are scale-sensitive. Increasing camera detail must not also
    double the detector's input scale. Box coordinates are mapped with the
    actual rounded dimensions, separately on each axis.
    """
    import cv2
    height, width = frame.shape[:2]
    scale = min(640 / width, 480 / height, 1.)
    if scale == 1.:
        return frame
    return cv2.resize(frame, (round(width * scale), round(height * scale)),
                      interpolation=cv2.INTER_LINEAR)


def aligned_rois(frame, transform, x, y, mean, std, crop_size):
    """Independent CPU work; preserve the original warp and conversion order."""
    import cv2
    from skimage.transform import warp
    aligned = (warp(frame, inverse_map=transform.inverse, output_shape=(256, 256)) * 255).astype(np.uint8)
    gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
    lip = normalize_roi(gray[y-48:y+48, x-48:x+48], mean, std, crop_size)
    face = normalize_roi(gray, mean, std, crop_size)
    return face, lip


class FacePreprocessor:
    def __init__(self, mean_face_path: Path, device: str = 'cuda', mean: float = .421, std: float = .165, crop_size: int = 88, *, visual_speech=None):
        import face_alignment
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False, face_detector='sfd')
        self.mean_face = np.load(mean_face_path, allow_pickle=False)
        self.mean, self.std, self.crop_size = mean, std, crop_size
        self.visual_speech = visual_speech
        if self.mean_face.shape != (68, 2):
            raise ValueError('Expected the AV-HuBERT 68-point reference face.')
        self.roi_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='lipreading-roi')

    def observe_frame(self, frame, cache, *, boxes=False):
        """Immutable observations only: no temporal gate/alignment state advances."""
        entry = cache.entry(frame) if cache is not None else None
        if entry is not None and entry.detected and (not boxes or entry.box_detected):
            cache.detector_hits += 1
            return entry.landmarks, entry.box
        view = detection_view(frame)
        options = {}
        if view is not frame:
            detections = self.detector.face_detector.detect_from_image(view)
            mapped = []
            for detection in detections:
                box = np.array(detection, dtype=np.float64, copy=True)
                box[:4] *= [frame.shape[1] / view.shape[1], frame.shape[0] / view.shape[0],
                            frame.shape[1] / view.shape[1], frame.shape[0] / view.shape[0]]
                mapped.append(box)
            # An empty list is intentional: None would rerun SFD at full scale.
            options['detected_faces'] = mapped
        if boxes:
            found, _, found_boxes = self.detector.get_landmarks_from_image(frame, return_bboxes=True, **options)
            box = np.asarray(found_boxes[0]).copy() if found_boxes is not None and len(found_boxes) == 1 else None
        else:
            found = self.detector.get_landmarks_from_image(frame, **options)
            box = None
        if cache is not None:
            cache.detector_calls += 1
        if found is not None and len(found) > 1:
            raise FaceQualityError('Keep only one person in the camera view. Faces on pictures or screens can also be detected.', camera_hint='multiple_faces')
        observation = found[0][:, :2] if found else None
        if entry is not None:
            observation = observation.copy() if observation is not None else None
            entry.landmarks, entry.detected = observation, True
            entry.box, entry.box_detected = box, boxes
        return observation, box

    def prepare_frame(self, frame, cache, *, speech_gate):
        cache.bind(self.detector)
        self.observe_frame(frame, cache, boxes=speech_gate and self.visual_speech is not None)
        entry = cache.entry(frame)
        if entry.brightness is None:
            entry.brightness = frame.mean()

    def warmup(self):
        """Initialize detector kernels and Numba helpers before camera admission.

        Blank frames and a supplied dummy box exercise the normal code paths;
        all results are discarded and no session cache is populated.
        """
        from skimage.transform import SimilarityTransform, warp
        for width, height in ((640, 360), (640, 480), (1280, 720), (1280, 960)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            box = np.array([160., 60., 480., 340., 1.], dtype=np.float64)
            for _ in range(2):
                self.detector.face_detector.detect_from_image(detection_view(frame))
                self.detector.get_landmarks_from_image(frame, detected_faces=[box])
        gray = np.zeros((256, 256), dtype=np.uint8)
        transform = SimilarityTransform()
        points = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
        transform.estimate(points, points)
        warp(frame, inverse_map=transform.inverse, output_shape=(256, 256))
        normalize_roi(gray, self.mean, self.std, self.crop_size)
        visual = getattr(self, 'visual_speech', None)
        if visual is not None:
            # Exercise the frozen visual model without populating session state.
            for count in (51, 52):
                visual.predict_crops(np.zeros((count, 112, 112), dtype=np.uint8))

    def __call__(self, frames: list[np.ndarray], fps: float = 25., *, cache: PreprocessCache | None = None, speech_gate: bool = False, activity_only: bool = False) -> Preprocessed:
        visual = getattr(self, 'visual_speech', None) if speech_gate else None
        try:
            return self._prepare(frames, fps, cache=cache, speech_gate=speech_gate,
                                 visual_speech=visual, activity_only=activity_only)
        except (FaceQualityError, ValueError) as exc:
            if visual is None:
                raise
            activity = self._visual_uncertain(str(exc), cache)
            if getattr(exc, 'camera_hint', None):
                activity['camera_hint'] = exc.camera_hint
                activity['metrics']['capture_issue'] = exc.camera_hint
            return Preprocessed(None, None, {'frames': len(frames), 'fps': 25},
                                activity)

    @staticmethod
    def _visual_uncertain(detail: str, cache=None) -> dict:
        if cache is not None and cache.speech_state is not None:
            return cache.speech_state.abstain(detail)
        return {'state': 'uncertain', 'suppress': True, 'preserve_transcript': True,
                'detail': detail, 'metrics': {'speech_detector': 'Light-ASD visual'}}

    def _visual_activity(self, frames, boxes, fps, cache, visual_speech):
        from .speech_gate import SpeechGateState
        explicit_times = cache.sample_times_ms if cache is not None else None
        if explicit_times is not None:
            if fps != 25 or len(explicit_times) != len(frames):
                raise ValueError('Speech detection needs matching 25 fps sample timestamps.')
            if cache.speech_state is None:
                cache.speech_state = SpeechGateState()
            state, times = cache.speech_state, explicit_times
        else:
            # Offline calls without capture timestamps are independent clips.
            state, times = SpeechGateState(), np.arange(len(frames), dtype=np.float64) * 40.
        if any(box is None for box in boxes):
            return self._visual_uncertain('Keep one face continuously visible for the speech check.', cache)
        frozen = all(np.array_equal(frame, frames[0]) for frame in frames[1:])
        try:
            # Identical input pixels contain no visible articulation. Advance the
            # same temporal gate with zeros so frozen video cannot reuse speech.
            options = {'cache': cache} if getattr(visual_speech, 'supports_crop_cache', False) else {}
            scores = (np.zeros(len(frames), dtype=np.float32) if frozen
                      else visual_speech.predict(frames, np.asarray(boxes), **options))
        except ValueError:
            return self._visual_uncertain('The visual speech check could not read this clip. Keep your face clearly visible and try again.', cache)
        except RuntimeError as exc:
            import logging
            logging.getLogger(__name__).warning('Visual speaking detector failed: %s', exc)
            return self._visual_uncertain('Visual speech detection is temporarily unavailable. Stop reading and retry.', cache)
        result = state.update(scores, times)
        result['metrics']['frozen_video'] = frozen
        return result

    def _prepare(self, frames, fps, *, cache, speech_gate, visual_speech, activity_only=False):
        from skimage.transform import SimilarityTransform
        if not frames or fps <= 0:
            raise FaceQualityError('No video frames received.')
        if any(f.ndim != 3 or f.shape[2] != 3 or f.dtype != np.uint8 for f in frames):
            raise ValueError('Expected uint8 RGB frames.')
        # Preserve real duration: callers supply measured fps or timestamp-resample first.
        count = round(len(frames) * 25 / fps)
        if not 20 <= count <= MAX_PREPROCESS_FRAMES:
            raise FaceQualityError(f'Use a video segment between 0.8 and {HISTORY_SECONDS} seconds.')
        indices = np.minimum((np.arange(count) * fps / 25).round().astype(int), len(frames) - 1)
        frames = [frames[i] for i in indices]
        if cache is not None:
            cache.bind(self.detector)
        landmarks, boxes = [], []
        for frame in frames:
            observation, box = self.observe_frame(frame, cache, boxes=visual_speech is not None)
            landmarks.append(observation)
            boxes.append(box)
        track = interpolate_landmarks(landmarks)
        span = np.ptp(track[:, :, 0], axis=1)
        if np.median(span) < 100:
            raise FaceQualityError('Move closer — your face is too small to read clearly. Keep your mouth visible and well lit.', camera_hint='move_closer')
        brightness = []
        for frame in frames:
            entry = cache.entry(frame) if cache is not None else None
            if entry is None:
                brightness.append(frame.mean())
            else:
                if entry.brightness is None:
                    entry.brightness = frame.mean()
                brightness.append(entry.brightness)
        if np.mean(brightness) < 25:
            raise FaceQualityError('Add more light in front of your face.')
        quality = {'tracked_fraction': sum(p is not None for p in landmarks) / len(landmarks),
                   'frames': len(frames), 'fps': 25, 'face_width_px': round(float(np.median(span)))}
        activity = None
        if visual_speech is not None:
            activity = self._visual_activity(frames, boxes, fps, cache, visual_speech)
            if activity['suppress']:
                return Preprocessed(None, None, quality, activity)
        elif speech_gate:
            from .activity import analyze_activity, confirm_image_stillness
            from .registered_activity import confirm_registered_stillness
            observed = np.array([p is not None for p in landmarks])
            activity = analyze_activity(track, observed, fps=25)
            activity['metrics']['landmark_state'] = activity['state']
            # Full motion metrics exist only after all tracking/pose checks pass.
            # Check raw mouth pixels as well: detector jitter can invent motion,
            # while insensitive landmarks can miss real, subtle articulation.
            if 'shape_motion' in activity['metrics']:
                pixels = confirm_image_stillness(frames, track, observed, fps=25)
                activity['metrics'].update(pixels['metrics'])
                if pixels['confirmed']:
                    activity.update(state='still', suppress=True,
                                    detail='Waiting for speech — mouth image appears still.')
                else:
                    activity['suppress'] = False
            # Independently register non-mouth image features so head movement
            # and landmark jitter do not masquerade as articulation. A failed
            # measurement now abstains instead of asking FLARE to guess words.
            if not activity['suppress']:
                registered = confirm_registered_stillness(frames, track, observed, fps=25)
                activity['metrics'].update(registered['metrics'])
                activity['metrics'].update(registration_reliable=registered['reliable'],
                                           registration_reason=registered['detail'])
                if registered['confirmed']:
                    activity.update(state='still', suppress=True,
                                    detail='Waiting for speech — lips are still after accounting for head movement.')
                elif not registered['reliable']:
                    # Imperfect image registration is not evidence of silence.
                    # Sustained independent lip articulation can still support
                    # reading even when skin pixels exceed the strict noise gate.
                    from .articulation import detect_articulation
                    articulation = detect_articulation(track, observed, fps=25)
                    activity['metrics'].update(articulation['metrics'])
                    activity['metrics']['articulation_active'] = articulation['active']
                    if articulation['active']:
                        activity.update(state='motion', suppress=False,
                                        detail='Clear lip movement detected; reading continues despite imperfect image alignment.')
                    else:
                        activity.update(state='uncertain', suppress=True,
                                        detail='Hold your face toward the camera in steady light; mouth movement is not clear enough to read.')
                else:
                    activity.update(state='motion', suppress=False,
                                    detail='Mouth movement detected independently of head movement.')
            if activity['suppress']:
                # Skip still clips or clips without reliable motion evidence.
                return Preprocessed(None, None, quality, activity)
        if activity_only:
            return Preprocessed(None, None, quality, activity)
        stable = [33, 36, 39, 42, 45]
        faces, lips = [], []
        # Match AV-HuBERT forward smoothing; reuse the last transform for the tail.
        margin = min(12, len(frames))
        transform = None
        pool = getattr(self, 'roi_pool', None)
        jobs, pending_rois = [], {}

        def finish_jobs():
            # Only this thread mutates caches. Keep queued work bounded even for
            # minute-long utterances; independent image work can finish in any order.
            for index, entry, key, future, fresh in jobs:
                face_roi, lip_roi = future.result()
                faces[index], lips[index] = face_roi, lip_roi
                if entry is not None:
                    entry.rois[key] = (face_roi, lip_roi)
                    entry.rois.move_to_end(key)
                    while len(entry.rois) > 4:
                        entry.rois.popitem(last=False)
                    cache.warp_calls += int(fresh)
                    cache.warp_hits += int(not fresh)
            jobs.clear()
            pending_rois.clear()

        for index, frame in enumerate(frames):
            if index <= len(frames) - margin:
                smoothed = track[index:index + margin].mean(axis=0)
                transform = SimilarityTransform()
                if not transform.estimate(smoothed[stable], self.mean_face[stable]):
                    raise FaceQualityError('Could not align your face. Look toward the camera.')
            aligned_points = transform(track[index])
            x, y = np.round(aligned_points[48:68].mean(axis=0)).astype(int)
            x, y = np.clip([x, y], 48, 208)
            entry = cache.entry(frame) if cache is not None else None
            roi_key = (transform.params.dtype.str, transform.params.tobytes(), int(x), int(y),
                       self.mean, self.std, self.crop_size) if entry is not None else None
            if entry is not None and roi_key in entry.rois:
                face_roi, lip_roi = entry.rois[roi_key]
                entry.rois.move_to_end(roi_key)
                cache.warp_hits += 1
            else:
                if pool is not None:
                    key = (id(frame), roi_key) if entry is not None else index
                    future = pending_rois.get(key)
                    fresh = future is None
                    if fresh:
                        future = pool.submit(aligned_rois, frame, transform, x, y,
                                             self.mean, self.std, self.crop_size)
                        pending_rois[key] = future
                    jobs.append((index, entry, roi_key, future, fresh))
                    faces.append(None)
                    lips.append(None)
                    if len(jobs) >= 16:
                        finish_jobs()
                    continue
                face_roi, lip_roi = aligned_rois(frame, transform, x, y,
                                                self.mean, self.std, self.crop_size)
                if entry is not None:
                    entry.rois[roi_key] = (face_roi, lip_roi)
                    while len(entry.rois) > 4:
                        entry.rois.popitem(last=False)
                    cache.warp_calls += 1
            lips.append(lip_roi)
            faces.append(face_roi)
        finish_jobs()
        # [B, C, T, H, W], matching the released frontend.
        face = np.stack(faces)[None, None]
        lip = np.stack(lips)[None, None]
        return Preprocessed(face, lip, quality, activity)

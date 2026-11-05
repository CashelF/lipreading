"""Conservative visual evidence for rejecting clearly motionless video clips.

This is a hand-tuned inactivity heuristic, not a trained speaking classifier or
speech probability. These helpers report uncertain geometry and movement; the
preprocessor uses independent image registration before admitting such clips.
Landmark thresholds are fractions of inter-eye distance after per-frame
similarity alignment; pixel thresholds use gray levels. None adapt upward to
observed speech or head movement. These are uncalibrated heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActivityThresholds:
    window_seconds: float = .5
    min_frames: int = 20
    # Low fixed cutoffs: below these values in every local window is clearly still.
    still_shape: float = .006
    still_opening: float = .004
    still_width: float = .006
    # Catch slow/subtle changes that accumulate across more than one local window.
    still_global_shape: float = .010
    still_global_opening: float = .008
    still_global_width: float = .010
    # Keep single-frame/very brief changes that the 3-frame median could remove.
    still_raw_shape_step: float = .012
    still_raw_opening_step: float = .006
    still_raw_width_step: float = .008
    # These label clear motion only. Everything between still and motion passes.
    motion_shape: float = .012
    motion_opening: float = .010
    motion_width: float = .012
    max_anchor_residual: float = .025
    max_nose_offset: float = .45
    max_eye_width_ratio: float = 2.5
    min_eye_distance_px: float = 5.


DEFAULT_THRESHOLDS = ActivityThresholds()
_ANCHORS = np.array([33, 36, 39, 42, 45])


def _result(state: str, detail: str, metrics: dict) -> dict:
    return {'state': state, 'suppress': state == 'still', 'detail': detail, 'metrics': metrics}


def _uncertain(detail: str, metrics: dict | None = None) -> dict:
    return _result('uncertain', detail, metrics or {})


def _median3(values: np.ndarray) -> np.ndarray:
    # Only suppress isolated measurement spikes; raw-change guards below keep
    # brief genuine movement eligible for decoding. Endpoints are preserved.
    result = values.copy()
    result[1:-1] = np.median(np.stack((values[:-2], values[1:-1], values[2:])), axis=0)
    return result


def _shape_range(points: np.ndarray) -> float:
    span = np.ptp(points, axis=0)
    return float(np.sqrt(np.mean(np.sum(span * span, axis=-1))))


def _maximum_window_range(values: np.ndarray, window: int, *, shape: bool = False) -> float:
    # Max over fixed-size local intervals: appending silence cannot dilute earlier
    # activity, unlike a variance/mean computed across an increasingly long clip.
    best = 0.
    for begin in range(len(values) - window + 1):
        part = values[begin:begin + window]
        score = _shape_range(part) if shape else float(np.ptp(part))
        best = max(best, score)
    return best


def analyze_activity(track: np.ndarray, observed: np.ndarray, fps: float = 25.,
                     *, thresholds: ActivityThresholds = DEFAULT_THRESHOLDS) -> dict:
    """Decide whether an entire *source clip* is clearly motionless.

    ``track`` contains interpolated image-space 68-point landmarks. ``observed``
    identifies actual detections, before interpolation. Any detection gap is kept
    uncertain rather than treating interpolated stillness as evidence of silence.
    The input is never modified, and no state carries between clips or sessions.
    """
    points = np.asarray(track, dtype=np.float64)
    detected = np.asarray(observed, dtype=bool)
    if (points.ndim != 3 or points.shape[1:] != (68, 2)
            or detected.shape != (len(points),)):
        return _uncertain('Landmark geometry is uncertain; reading continues.')
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(points).all():
        return _uncertain('Landmark timing or geometry is uncertain; reading continues.')
    metrics = {'frames': len(points), 'fps': float(fps),
               'tracked_fraction': round(float(detected.mean()), 6) if len(points) else 0.}
    if len(points) < thresholds.min_frames:
        return _uncertain('Too little motion history; reading continues.', metrics)
    if not detected.all():
        return _uncertain('Tracking had a gap; reading continues.', metrics)

    left = points[:, [36, 39]].mean(axis=1)
    right = points[:, [42, 45]].mean(axis=1)
    eye_vector = right - left
    eye_distance = np.linalg.norm(eye_vector, axis=1)
    metrics['eye_distance_px'] = round(float(np.median(eye_distance)), 3)
    if np.min(eye_distance) < thresholds.min_eye_distance_px:
        return _uncertain('Face geometry is too small or uncertain; reading continues.', metrics)
    eye_unit = eye_vector / eye_distance[:, None]
    nose_offset = np.abs(np.sum((points[:, 33] - (left + right) / 2) * eye_unit, axis=1)) / eye_distance
    left_width = np.linalg.norm(points[:, 39] - points[:, 36], axis=1)
    right_width = np.linalg.norm(points[:, 45] - points[:, 42], axis=1)
    if np.min(np.minimum(left_width, right_width)) < 1e-6:
        return _uncertain('Eye landmarks are uncertain; reading continues.', metrics)
    eye_ratio = np.maximum(left_width / right_width, right_width / left_width)
    metrics['nose_offset'] = round(float(np.max(nose_offset)), 6)
    metrics['eye_width_ratio'] = round(float(np.max(eye_ratio)), 6)
    if (np.max(nose_offset) > thresholds.max_nose_offset
            or np.max(eye_ratio) > thresholds.max_eye_width_ratio):
        return _uncertain('Head pose is uncertain; reading continues.', metrics)

    # Per-frame Procrustes fit uses only nose/eye anchors, never the moving lips.
    # Retaining the first reference keeps prior intervals comparable as clips grow.
    anchors = points[:, _ANCHORS]
    center = anchors.mean(axis=1, keepdims=True)
    centered = anchors - center
    reference = centered[0] / eye_distance[0]
    covariance = np.einsum('tni,nj->tij', centered, reference)
    try:
        u, singular, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return _uncertain('Head alignment is uncertain; reading continues.', metrics)
    rotation = u @ vt
    denominator = np.sum(centered * centered, axis=(1, 2))
    if (np.any(denominator < 1e-9) or np.any(np.linalg.det(rotation) < 0)
            or np.any(singular[:, -1] < singular[:, 0] * .01)):
        return _uncertain('Head alignment is uncertain; reading continues.', metrics)
    scale = singular.sum(axis=1) / denominator
    aligned = np.einsum('tni,tij->tnj', points - center, rotation) * scale[:, None, None]
    residual = np.sqrt(np.mean(np.sum((aligned[:, _ANCHORS] - reference) ** 2, axis=-1), axis=1))
    metrics['anchor_residual'] = round(float(np.max(residual)), 6)
    if np.max(residual) > thresholds.max_anchor_residual:
        return _uncertain('Head pose or tracking changed; reading continues.', metrics)

    lips = aligned[:, 48:68]
    # Mean of three inner-lip distances detects articulation even with a mostly
    # closed mouth. Absolute openness is deliberately never used as a threshold.
    opening = np.linalg.norm(aligned[:, [61, 62, 63]] - aligned[:, [67, 66, 65]], axis=2).mean(axis=1)
    width = np.linalg.norm(aligned[:, 54] - aligned[:, 48], axis=1)
    smooth_lips, smooth_opening, smooth_width = map(_median3, (lips, opening, width))
    window = min(len(points), max(3, round(fps * thresholds.window_seconds) + 1))
    values = {
        'shape_motion': _maximum_window_range(smooth_lips, window, shape=True),
        'opening_motion': _maximum_window_range(smooth_opening, window),
        'width_motion': _maximum_window_range(smooth_width, window),
        'global_shape_motion': _shape_range(smooth_lips),
        'global_opening_motion': float(np.ptp(smooth_opening)),
        'global_width_motion': float(np.ptp(smooth_width)),
        'raw_shape_step': float(np.max(np.sqrt(np.mean(np.sum(np.diff(lips, axis=0) ** 2, axis=2), axis=1)))),
        'raw_opening_step': float(np.max(np.abs(np.diff(opening)))),
        'raw_width_step': float(np.max(np.abs(np.diff(width)))),
    }
    metrics.update({key: round(value, 6) for key, value in values.items()})
    metrics.update({'window_frames': window,
                    'still_shape_threshold': thresholds.still_shape,
                    'still_opening_threshold': thresholds.still_opening,
                    'still_width_threshold': thresholds.still_width})
    still_limits = {
        'shape_motion': thresholds.still_shape,
        'opening_motion': thresholds.still_opening,
        'width_motion': thresholds.still_width,
        'global_shape_motion': thresholds.still_global_shape,
        'global_opening_motion': thresholds.still_global_opening,
        'global_width_motion': thresholds.still_global_width,
        'raw_shape_step': thresholds.still_raw_shape_step,
        'raw_opening_step': thresholds.still_raw_opening_step,
        'raw_width_step': thresholds.still_raw_width_step,
    }
    if all(values[key] <= limit for key, limit in still_limits.items()):
        return _result('still', 'Waiting for speech — lips appear still.', metrics)
    if (values['shape_motion'] >= thresholds.motion_shape
            or values['opening_motion'] >= thresholds.motion_opening
            or values['width_motion'] >= thresholds.motion_width):
        return _result('motion', 'Visible lip movement detected.', metrics)
    return _uncertain('Small or uncertain lip movement; reading continues.', metrics)


@dataclass(frozen=True)
class PixelStillnessThresholds:
    # These are gray levels on the original 0..255 scale, not probabilities.
    max_tail_change: float = 2.5
    max_pixel_change: float = 8.
    tail_fraction: float = .01
    padding_fraction: float = .25
    min_padding_px: int = 3
    min_patch_width: int = 16
    min_patch_height: int = 10
    min_frames: int = 20
    window_seconds: float = .5


DEFAULT_PIXEL_THRESHOLDS = PixelStillnessThresholds()


def confirm_image_stillness(frames: list[np.ndarray], track: np.ndarray,
                            observed: np.ndarray, fps: float = 25., *,
                            thresholds: PixelStillnessThresholds = DEFAULT_PIXEL_THRESHOLDS) -> dict:
    """Corroborate stillness using a fixed raw mouth patch, independent of jitter.

    The same image-space rectangle is used in every frame. We do not warp using
    noisy landmarks, normalize brightness, or smooth over time. A light spatial
    blur reduces one-gray-level sensor noise; the top 1% and an absolute-pixel
    guard retain sparse changes. Comparisons cover every frame against references
    spaced by a fixed short window, as well as adjacent frames. This retains early
    and slow accumulated evidence when a clip is extended.

    This deliberately returns false for incomplete tracking, unreliable bounds,
    head translation within the patch, lighting changes, and ambiguous pixels.
    A true result is a low-motion heuristic, never a calibrated speech decision.
    Callers should not use it to override landmark geometry/pose failures.
    """
    import cv2

    def result(confirmed=False, reason='Image motion is uncertain.', metrics=None):
        return {'confirmed': bool(confirmed), 'detail': reason, 'metrics': metrics or {}}

    points = np.asarray(track, dtype=np.float64)
    detections = np.asarray(observed, dtype=bool)
    if (len(frames) < thresholds.min_frames or points.shape != (len(frames), 68, 2)
            or detections.shape != (len(frames),) or not detections.all()
            or not np.isfinite(points).all() or not np.isfinite(fps) or fps <= 0):
        return result(reason='Image stillness needs complete, valid tracking.')
    if any(not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
           or frame.dtype != np.uint8 for frame in frames):
        return result(reason='Image format is uncertain.')
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        return result(reason='Image dimensions changed.')
    height, width = shape[:2]
    left = points[:, [36, 39]].mean(axis=1)
    right = points[:, [42, 45]].mean(axis=1)
    if np.min(np.linalg.norm(right - left, axis=1)) < DEFAULT_THRESHOLDS.min_eye_distance_px:
        return result(reason='Face geometry is uncertain.')
    # Median coordinates define the fixed box; EVERY frame must still fit inside.
    # Otherwise a moving speaker could leave a static background patch behind.
    median_lips = np.median(points[:, 48:68], axis=0)
    low = median_lips.min(axis=0)
    high = median_lips.max(axis=0)
    extent = high - low
    if np.any(extent <= 0):
        return result(reason='Mouth geometry is degenerate.')
    padding = np.maximum(extent * thresholds.padding_fraction, thresholds.min_padding_px)
    x0, y0 = np.floor(low - padding).astype(int)
    x1, y1 = np.ceil(high + padding).astype(int) + 1
    if (x0 < 0 or y0 < 0 or x1 > width or y1 > height
            or x1 - x0 < thresholds.min_patch_width or y1 - y0 < thresholds.min_patch_height):
        return result(reason='Mouth image is too small or outside the frame.')
    all_lips = points[:, 48:68]
    if (np.any(all_lips[:, :, 0] < x0) or np.any(all_lips[:, :, 0] >= x1)
            or np.any(all_lips[:, :, 1] < y0) or np.any(all_lips[:, :, 1] >= y1)):
        return result(reason='Mouth moved outside the fixed image patch.')

    patches = []
    for frame in frames:
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
        patches.append(cv2.GaussianBlur(gray, (3, 3), 0, borderType=cv2.BORDER_REPLICATE).reshape(-1))
    pixels = np.stack(patches)
    tail_count = max(4, int(np.ceil(pixels.shape[1] * thresholds.tail_fraction)))
    tail_count = min(tail_count, pixels.shape[1])
    stride = max(1, round(fps * thresholds.window_seconds))
    largest_tail, largest_pixel = 0., 0.

    def measure(changes):
        nonlocal largest_tail, largest_pixel
        changes = np.abs(changes)
        largest_pixel = max(largest_pixel, float(changes.max()))
        tail = np.partition(changes, changes.shape[1] - tail_count, axis=1)[:, -tail_count:]
        largest_tail = max(largest_tail, float(tail.mean(axis=1).max()))

    measure(np.diff(pixels, axis=0))
    # References never disappear as more frames arrive. Global comparisons also
    # catch gradual motion whose consecutive-frame difference stays below noise.
    for reference in range(0, len(pixels), stride):
        measure(pixels - pixels[reference])
    metrics = {
        'pixel_tail_change': round(largest_tail, 6),
        'pixel_peak_change': round(largest_pixel, 6),
        'pixel_still_threshold': thresholds.max_tail_change,
        'pixel_peak_threshold': thresholds.max_pixel_change,
        'pixel_tail_fraction': thresholds.tail_fraction,
        'pixel_patch_width': int(x1 - x0),
        'pixel_patch_height': int(y1 - y0),
        'pixel_reference_stride': stride,
    }
    # Exact boundary values remain uncertain instead of being suppressed.
    confirmed = (largest_tail < thresholds.max_tail_change
                 and largest_pixel < thresholds.max_pixel_change)
    return result(confirmed,
                  'Mouth pixels are consistently still.' if confirmed else 'Mouth pixels changed; reading continues.',
                  metrics)

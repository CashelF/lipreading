"""Independent mouth motion after rigid, non-mouth image registration.

This is a conservative abstention heuristic, not a trained speaking probability.
Landmarks locate the face and mouth; only non-mouth image features fit motion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class RegistrationThresholds:
    min_frames: int = 20
    min_features: int = 12
    max_features: int = 60
    min_inlier_fraction: float = .8
    max_tracking_error: float = .35
    max_registration_error: float = .4
    # Residual above this fixed bound can be resampling/pose error, including
    # small rotations. It is uncertainty, not evidence of articulation.
    max_control_residual: float = 2.
    max_brightness_offset: float = 20.
    mouth_tail: float = 2.5
    mouth_peak: float = 8.


DEFAULT_THRESHOLDS = RegistrationThresholds()


def confirm_registered_stillness(frames, track, observed, fps=25., *,
                                 thresholds=DEFAULT_THRESHOLDS):
    """Return ``confirmed``, registration ``reliable``, detail and scalar metrics.

    Every frame is compared with the first so brief and slow mouth changes cannot
    disappear in a temporal average. Eyes and the entire lower mouth/jaw region
    are excluded from fitting. Unreliable registration does not mean stillness.
    A bounded brightness offset is measured only on non-mouth control pixels.
    """
    metrics = {'registration_frames': len(frames)}

    def result(reliable=False, confirmed=False, detail='Keep your face steady and clearly visible.'):
        return {'reliable': bool(reliable), 'confirmed': bool(confirmed),
                'detail': detail, 'metrics': metrics}

    points = np.asarray(track, dtype=np.float64)
    detections = np.asarray(observed, dtype=bool)
    if (len(frames) < thresholds.min_frames or points.shape != (len(frames), 68, 2)
            or detections.shape != (len(frames),) or not detections.all()
            or not np.isfinite(points).all() or not math.isfinite(fps) or fps <= 0):
        return result(detail='Mouth motion needs complete, valid face tracking.')
    if any(not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
           or frame.dtype != np.uint8 for frame in frames):
        return result(detail='Camera image format is uncertain.')
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        return result(detail='Camera image dimensions changed.')
    left = points[:, [36, 39]].mean(axis=1)
    right = points[:, [42, 45]].mean(axis=1)
    eyes = right - left
    distance = np.linalg.norm(eyes, axis=1)
    if np.min(distance) < 5:
        return result(detail='Face geometry is too small or uncertain.')
    widths = np.stack((np.linalg.norm(points[:, 39] - points[:, 36], axis=1),
                       np.linalg.norm(points[:, 45] - points[:, 42], axis=1)))
    if np.min(widths) < 1e-6:
        return result(detail='Eye geometry is uncertain.')
    nose_offset = np.abs(np.sum((points[:, 33] - (left + right) / 2) * eyes, axis=1)) / distance ** 2
    if np.max(widths.max(axis=0) / widths.min(axis=0)) > 2.5 or nose_offset.max() > .45:
        return result(detail='Face the camera so mouth motion can be checked.')

    height, width = shape[:2]
    # A shared source crop bounds optical-flow work without resizing subtle motion.
    margin = max(8., float(np.median(distance)) * .2)
    low = np.floor(points.min(axis=(0, 1)) - margin).astype(int)
    high = np.ceil(points.max(axis=(0, 1)) + margin).astype(int) + 1
    low = np.maximum(low, 0)
    high = np.minimum(high, [width, height])
    if np.any(high - low < 32) or np.any(high - low > 640):
        return result(detail='Face image is outside the usable tracking area.')
    local = points - low
    first = local[0]
    crop_width, crop_height = high - low
    mouth_low, mouth_high = first[48:68].min(axis=0), first[48:68].max(axis=0)
    extent = mouth_high - mouth_low
    if extent[0] < 10 or extent[1] <= 0:
        return result(detail='Mouth geometry is too small or uncertain.')
    padding = np.maximum(extent * .35, 5)
    x0, y0 = np.floor(mouth_low - padding).astype(int)
    x1, y1 = np.ceil(mouth_high + padding).astype(int) + 1
    if x0 < 0 or y0 < 0 or x1 > crop_width or y1 > crop_height:
        return result(detail='Keep the whole mouth in the camera view.')

    control = np.zeros((crop_height, crop_width), dtype=np.uint8)
    cv2.fillConvexPoly(control, cv2.convexHull(np.rint(first[:48]).astype(np.int32)), 255)
    control[max(0, int(mouth_low[1]) - 6):] = 0
    for eye in (first[36:42], first[42:48]):
        eye_low = np.floor(eye.min(axis=0) - 3).astype(int)
        eye_high = np.ceil(eye.max(axis=0) + 3).astype(int) + 1
        control[max(0, eye_low[1]):eye_high[1], max(0, eye_low[0]):eye_high[0]] = 0
    # Avoid crop edges and hull boundaries where warp interpolation mixes background.
    control = cv2.erode(control, np.ones((3, 3), np.uint8))
    control_indices = control > 0
    if int(control_indices.sum()) < 150:
        return result(detail='Too little stable face texture to check mouth motion.')
    gray = [cv2.cvtColor(frame[low[1]:high[1], low[0]:high[0]], cv2.COLOR_RGB2GRAY)
            for frame in frames]
    reference = gray[0]
    features = cv2.goodFeaturesToTrack(reference, thresholds.max_features, .01, 4,
                                      mask=control, blockSize=5)
    if features is None or len(features) < thresholds.min_features:
        return result(detail='Too little stable face texture to check mouth motion.')
    metrics['registration_features'] = len(features)
    reference_float = reference.astype(np.float32)
    reference_blurred = cv2.GaussianBlur(reference_float, (3, 3), 0)
    ref_patch = reference_blurred[y0:y1, x0:x1]
    ref_control = reference_blurred[control_indices]
    min_inliers, max_control, max_offset = 1., 0., 0.
    max_tail, max_peak = 0., 0.
    flow_args = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, .01))

    for index, current in enumerate(gray[1:], start=1):
        found, status, _ = cv2.calcOpticalFlowPyrLK(reference, current, features, None, **flow_args)
        if found is None or not np.isfinite(found).all():
            return result(detail='Head movement could not be registered reliably.')
        back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(current, reference, found, None, **flow_args)
        if back is None:
            return result(detail='Head movement could not be registered reliably.')
        good = ((status[:, 0] > 0) & (reverse_status[:, 0] > 0)
                & (np.linalg.norm(back[:, 0] - features[:, 0], axis=1) < thresholds.max_tracking_error))
        if int(good.sum()) < thresholds.min_features:
            return result(detail='Head movement could not be registered reliably.')
        transform, inliers = cv2.estimateAffinePartial2D(features[good, 0], found[good, 0],
            method=cv2.RANSAC, ransacReprojThreshold=thresholds.max_registration_error,
            maxIters=500, confidence=.99, refineIters=10)
        if transform is None or inliers is None or not np.isfinite(transform).all():
            return result(detail='Head movement could not be registered reliably.')
        fraction = float(inliers.mean())
        min_inliers = min(min_inliers, fraction)
        metrics['registration_min_inlier_fraction'] = round(min_inliers, 4)
        anchor_points = features[good, 0][inliers[:, 0] > 0]
        if (fraction < thresholds.min_inlier_fraction or len(anchor_points) < thresholds.min_features
                or np.ptp(anchor_points[:, 0]) < distance[0] * .5
                or np.ptp(anchor_points[:, 1]) < distance[0] * .2):
            return result(detail='Head movement could not be registered reliably.')
        scale = float(np.linalg.norm(transform[0, :2]))
        angle = abs(math.degrees(math.atan2(transform[1, 0], transform[0, 0])))
        if not .85 <= scale <= 1.15 or angle > 12:
            return result(detail='Keep your face facing the camera with gentle movement.')
        inverse = cv2.invertAffineTransform(transform)
        registered_lips = cv2.transform(local[index, 48:68][None].astype(np.float32), inverse)[0]
        if (np.any(registered_lips < [x0, y0]) or np.any(registered_lips >= [x1, y1])):
            return result(detail='Mouth moved outside its registered image region.')
        warped = cv2.warpAffine(current.astype(np.float32), transform, (crop_width, crop_height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        warped = cv2.GaussianBlur(warped, (3, 3), 0)
        control_difference = warped[control_indices] - ref_control
        offset = float(np.median(control_difference))
        residual = float(np.percentile(np.abs(control_difference - offset), 95))
        max_control, max_offset = max(max_control, residual), max(max_offset, abs(offset))
        metrics.update(registration_control_residual=round(max_control, 4),
                       registration_brightness_offset=round(max_offset, 4))
        if abs(offset) > thresholds.max_brightness_offset or residual > thresholds.max_control_residual:
            return result(detail='Lighting or face registration is uncertain; improve the camera view.')
        difference = np.abs(warped[y0:y1, x0:x1] - offset - ref_patch).ravel()
        tail_count = max(4, int(np.ceil(len(difference) * .01)))
        tail = float(np.partition(difference, -tail_count)[-tail_count:].mean())
        max_tail, max_peak = max(max_tail, tail), max(max_peak, float(difference.max()))

    metrics.update(registered_mouth_tail=round(max_tail, 4), registered_mouth_peak=round(max_peak, 4),
                   registered_tail_threshold=thresholds.mouth_tail, registered_peak_threshold=thresholds.mouth_peak)
    still = max_tail < thresholds.mouth_tail and max_peak < thresholds.mouth_peak
    return result(True, still, 'Waiting for speech — mouth is still after head-motion correction.'
                  if still else 'Mouth movement remains after head-motion correction.')

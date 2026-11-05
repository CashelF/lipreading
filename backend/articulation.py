"""Positive evidence of sustained articulation when image registration is unsure.

This is not a speaking probability or a silence classifier. Small/ambiguous
motion returns false; independently confirmed still pixels must take priority.
"""
from __future__ import annotations

import numpy as np


def detect_articulation(track: np.ndarray, observed: np.ndarray, fps: float = 25.) -> dict:
    """Find coherent mouth changes within a short, fully tracked frontal interval.

    Distances are normalized per frame, removing image translation, rotation and
    uniform scale. Two three-frame levels must differ coherently: one bad frame
    cannot trigger admission. Pose changes are checked over the same interval.
    """
    points = np.asarray(track, dtype=np.float64)
    detected = np.asarray(observed, dtype=bool)
    metrics = {'articulation_active': False, 'articulation_opening': 0.,
               'articulation_width': 0., 'articulation_opening_threshold': .04,
               'articulation_width_threshold': .06}

    def result(active=False):
        metrics['articulation_active'] = bool(active)
        return {'active': bool(active), 'metrics': metrics}

    if (points.ndim != 3 or points.shape[1:] != (68, 2)
            or detected.shape != (len(points),) or len(points) < 20
            or not np.isfinite(points).all() or not np.isfinite(fps) or fps <= 0):
        return result()
    left, right = points[:, [36, 39]].mean(axis=1), points[:, [42, 45]].mean(axis=1)
    eye_vector = right - left
    eye_distance = np.linalg.norm(eye_vector, axis=1)
    if np.min(eye_distance) < 5:
        return result()
    axis = eye_vector / eye_distance[:, None]
    normal = np.stack((-axis[:, 1], axis[:, 0]), axis=1)
    nose = (points[:, 33] - (left + right) / 2) / eye_distance[:, None]
    offset = np.sum(nose * axis, axis=1)
    height = np.sum(nose * normal, axis=1)
    eye_widths = np.stack((np.linalg.norm(points[:, 39] - points[:, 36], axis=1),
                           np.linalg.norm(points[:, 45] - points[:, 42], axis=1)), axis=1)
    if np.min(eye_widths) < 1e-6:
        return result()
    eye_ratio = np.log(eye_widths[:, 0] / eye_widths[:, 1])
    valid = detected & (np.abs(offset) <= .45) & (np.abs(eye_ratio) <= np.log(2.5))
    pose = np.stack((offset, height, eye_ratio), axis=1)
    # Nose height guards symmetric foreshortening that eye-width ratio misses.
    pose_limits = np.array([.06, .08, .15])
    gaps = np.linalg.norm(points[:, [61, 62, 63]] - points[:, [67, 66, 65]], axis=2)
    widths = np.stack((np.linalg.norm(points[:, 54] - points[:, 48], axis=1),
                        np.linalg.norm(points[:, 64] - points[:, 60], axis=1)), axis=1)
    measurements = np.concatenate((gaps, widths), axis=1) / eye_distance[:, None]
    sustain = max(3, round(.12 * fps))
    horizon = max(2 * sustain, round(.8 * fps))
    if len(points) < 2 * sustain:
        return result()
    low = np.array([measurements[i:i + sustain].min(axis=0)
                    for i in range(len(points) - sustain + 1)])
    high = np.array([measurements[i:i + sustain].max(axis=0)
                     for i in range(len(points) - sustain + 1)])
    best_opening, best_width = 0., 0.
    for first in range(len(low)):
        for second in range(first + sustain, min(len(low), first + horizon - sustain + 1)):
            end = second + sustain
            if not valid[first:end].all() or np.any(np.ptp(pose[first:end], axis=0) > pose_limits):
                continue
            # Every frame of one level must exceed every frame of the other.
            # Evaluate opening and closing separately so incoherent pairs cannot
            # combine one gap opening with another gap closing into evidence.
            forward, reverse = low[second] - high[first], low[first] - high[second]
            opening = max(float(np.sort(forward[:3])[1]), float(np.sort(reverse[:3])[1]), 0.)
            width = max(float(np.min(forward[3:])), float(np.min(reverse[3:])), 0.)
            best_opening, best_width = max(best_opening, opening), max(best_width, width)
            if opening >= .04 or width >= .06:
                metrics.update(articulation_opening=round(best_opening, 6),
                               articulation_width=round(best_width, 6),
                               articulation_start_frame=first, articulation_end_frame=end - 1,
                               articulation_sustain_frames=sustain)
                return result(True)
    metrics.update(articulation_opening=round(best_opening, 6), articulation_width=round(best_width, 6),
                   articulation_sustain_frames=sustain)
    return result()

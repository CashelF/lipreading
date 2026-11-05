"""Detection scale must not discard the detail used by FAN/FLARE/Light-ASD."""
from types import SimpleNamespace
import numpy as np
import pytest

from backend.preprocessing import FacePreprocessor, FaceQualityError, PreprocessCache, detection_view


class Detector:
    def __init__(self, detections):
        self.detections = detections
        self.views, self.inputs = [], []
        self.face_detector = SimpleNamespace(detect_from_image=self.detect)

    def detect(self, image):
        self.views.append(image)
        assert image.shape[1] <= 640 and image.shape[0] <= 480
        return self.detections

    def get_landmarks_from_image(self, image, *, detected_faces=None, return_bboxes=False):
        self.inputs.append((image, detected_faces))
        if detected_faces is None:
            detected_faces = self.detect(image)
        points = [np.tile(box[:2], (68, 1)) for box in detected_faces] or None
        return (points, None, detected_faces) if return_bboxes else points


@pytest.mark.parametrize('shape', [(720, 1280), (960, 1280), (901, 1201), (960, 540)])
def test_detector_boxes_map_back_to_native_pixels_and_confidence_is_unchanged(shape):
    source_box = np.array([50., 40., 150., 180., .93], dtype=np.float32)
    detector = Detector([source_box])
    proc = FacePreprocessor.__new__(FacePreprocessor)
    proc.detector = detector
    frame = np.full((*shape, 3), 100, dtype=np.uint8)
    points, box = proc.observe_frame(frame, None, boxes=True)
    view = detector.views[0]
    assert detector.inputs[0][0] is frame  # FAN uses original image, not resized.
    expected = source_box.astype(np.float64)
    expected[:4] *= [shape[1] / view.shape[1], shape[0] / view.shape[0]] * 2
    np.testing.assert_array_equal(box, expected)
    np.testing.assert_array_equal(points[0], expected[:2])
    np.testing.assert_array_equal(source_box, np.array([50., 40., 150., 180., .93], dtype=np.float32))


def test_legacy_resolution_keeps_original_image_and_detection_path():
    frame = np.full((360, 640, 3), 100, dtype=np.uint8)
    assert detection_view(frame) is frame
    proc = FacePreprocessor.__new__(FacePreprocessor)
    proc.detector = Detector([np.array([50., 40., 150., 180., .93])])
    proc.observe_frame(frame, None, boxes=True)
    assert proc.detector.inputs[0][1] is None
    assert proc.detector.views[0] is frame


def test_no_detection_does_not_retry_at_the_problematic_full_resolution():
    proc = FacePreprocessor.__new__(FacePreprocessor)
    proc.detector = Detector([])
    assert proc.observe_frame(np.zeros((720, 1280, 3), dtype=np.uint8), None, boxes=True) == (None, None)
    assert len(proc.detector.views) == 1
    assert proc.detector.inputs[0][1] == []


def test_two_detected_faces_still_abstain_instead_of_arbitrarily_picking_one():
    proc = FacePreprocessor.__new__(FacePreprocessor)
    proc.detector = Detector([np.array([50., 40., 150., 180., .93]),
                              np.array([250., 40., 350., 180., .94])])
    with pytest.raises(FaceQualityError) as error:
        proc.observe_frame(np.zeros((720, 1280, 3), dtype=np.uint8), None, boxes=True)
    assert error.value.camera_hint == 'multiple_faces'


def test_scaled_observations_and_boxes_reuse_the_immutable_frame_cache():
    proc = FacePreprocessor.__new__(FacePreprocessor)
    proc.detector = Detector([np.array([50., 40., 150., 180., .93])])
    cache = PreprocessCache()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    first = proc.observe_frame(frame, cache, boxes=True)
    second = proc.observe_frame(frame, cache, boxes=True)
    assert first[0] is second[0] and first[1] is second[1]
    assert len(proc.detector.inputs) == len(proc.detector.views) == 1

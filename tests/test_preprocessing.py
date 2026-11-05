"""Geometry and capture quality tests; no downloaded face detector is required."""
import numpy as np
import pytest

from backend.preprocessing import FacePreprocessor, FaceQualityError, interpolate_landmarks, normalize_roi


def points(value):
    return np.full((68, 2), value, dtype=np.float32)


def test_short_tracking_dropout_is_interpolated_at_actual_frame_positions():
    track = [points(i * 10) for i in range(10)]
    track[3] = None
    track[4] = None
    interpolated = interpolate_landmarks(track)
    assert interpolated.shape == (10, 68, 2)
    assert interpolated.dtype == np.float32
    np.testing.assert_array_equal(interpolated[3], points(30))
    np.testing.assert_array_equal(interpolated[4], points(40))
    assert track[3] is None, 'Interpolation must not mutate the original observations'


def test_brief_missing_endpoints_use_nearest_detected_face():
    track = [points(i) for i in range(10)]
    track[0] = track[-1] = None
    interpolated = interpolate_landmarks(track)
    np.testing.assert_array_equal(interpolated[0], points(1))
    np.testing.assert_array_equal(interpolated[-1], points(8))


def test_rejects_poor_overall_tracking_even_when_gaps_are_short():
    track = [None if i % 3 == 0 else points(i) for i in range(30)]
    with pytest.raises(FaceQualityError, match='fully visible'):
        interpolate_landmarks(track)


def test_rejects_long_tracking_loss_even_when_overall_coverage_is_good():
    track = [points(i) for i in range(40)]
    track[10:16] = [None] * 6
    with pytest.raises(FaceQualityError, match='tracking was lost'):
        interpolate_landmarks(track)


@pytest.mark.parametrize('track', [[], [None] * 20])
def test_no_valid_landmarks_produces_capture_guidance(track):
    with pytest.raises(FaceQualityError):
        interpolate_landmarks(track)


@pytest.mark.parametrize('level', [0, 128, 255])
def test_roi_normalization_preserves_grayscale_dynamic_range(level):
    output = normalize_roi(np.full((96, 96), level, dtype=np.uint8))
    assert output.shape == (88, 88)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output, (level / 255 - .421) / .165, atol=1e-6)


def test_center_crop_removes_four_pixel_border_without_flipping_image():
    image = np.zeros((96, 96), dtype=np.uint8)
    image[:4] = 255
    image[-4:] = 255
    image[:, :4] = 255
    image[:, -4:] = 255
    image[4, 4] = 100
    image[91, 91] = 200
    output = normalize_roi(image, mean=0, std=1)
    assert output[0, 0] == pytest.approx(100 / 255)
    assert output[-1, -1] == pytest.approx(200 / 255)
    assert np.count_nonzero(output) == 2


def reference_face():
    # Nondegenerate five-point similarity fit and a 160px-wide face.
    theta = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    result = np.stack([128 + 80 * np.cos(theta), 128 + 85 * np.sin(theta)], axis=1).astype(np.float32)
    result[[33, 36, 39, 42, 45]] = [[128, 145], [90, 100], [110, 100], [146, 100], [166, 100]]
    mouth_angle = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    result[48:68] = np.stack([128 + 25 * np.cos(mouth_angle), 180 + 10 * np.sin(mouth_angle)], axis=1)
    return result


class Detector:
    def __init__(self, observations):
        self.observations = iter(observations)

    def get_landmarks_from_image(self, frame):
        return next(self.observations)


def preprocessor(observations):
    # Bypass only detector construction/download. All production geometry runs.
    proc = FacePreprocessor.__new__(FacePreprocessor)
    proc.detector = Detector(observations)
    proc.mean_face = reference_face()
    proc.mean, proc.std, proc.crop_size = .421, .165, 88
    return proc


def test_dual_stream_layout_and_rgb_grayscale_conversion():
    proc = preprocessor([[reference_face()]] * 20)
    red = np.full((256, 256, 3), [255, 0, 0], dtype=np.uint8)
    result = proc([red] * 20)
    assert result.face.shape == result.lips.shape == (1, 1, 20, 88, 88)
    assert result.face.dtype == result.lips.dtype == np.float32
    # RGB red converts to luminance76; mistaken BGR treatment would produce29.
    expected = (76 / 255 - .421) / .165
    np.testing.assert_allclose(result.lips, expected, atol=.025)
    assert result.quality['frames'] == 20
    assert result.quality['tracked_fraction'] == 1
    assert result.quality['fps'] == 25


def test_lip_stream_is_cropped_around_mouth_and_differs_from_full_face():
    proc = preprocessor([[reference_face()]] * 20)
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[132:228, 80:176] = 255  # 96px patch centered on the mouth.
    result = proc([frame] * 20)
    assert np.mean(result.lips) > 3.4
    assert np.mean(result.face) < 0


def test_rejects_multiple_people_before_attempting_alignment():
    proc = preprocessor([[reference_face(), reference_face()]] * 20)
    with pytest.raises(FaceQualityError, match='only one person'):
        proc([np.full((256, 256, 3), 128, dtype=np.uint8)] * 20)


def test_rejects_lost_face_track():
    proc = preprocessor([None] * 20)
    with pytest.raises(FaceQualityError, match='fully visible'):
        proc([np.full((256, 256, 3), 128, dtype=np.uint8)] * 20)


def test_rejects_face_too_small_for_readable_lip_detail():
    tiny = reference_face() * .4
    proc = preprocessor([[tiny]] * 20)
    with pytest.raises(FaceQualityError, match='Move closer'):
        proc([np.full((256, 256, 3), 128, dtype=np.uint8)] * 20)


def test_rejects_dark_capture():
    proc = preprocessor([[reference_face()]] * 20)
    with pytest.raises(FaceQualityError, match='more light'):
        proc([np.zeros((256, 256, 3), dtype=np.uint8)] * 20)


def test_native_detail_can_pass_without_relaxing_face_width_requirement():
    # Same framing, two capture resolutions: a native 144px face becomes 72px
    # when the browser halves the image. Landmark locations are explicit here;
    # this tests admission/geometry, not detector accuracy on synthetic pixels.
    low_points = reference_face() * .45
    low = np.full((360, 640, 3), 128, dtype=np.uint8)
    with pytest.raises(FaceQualityError) as error:
        preprocessor([[low_points]] * 20)([low] * 20)
    assert error.value.camera_hint == 'move_closer'
    high = np.full((720, 1280, 3), 128, dtype=np.uint8)
    # Landmark predictions are explicit; exercise the new detector handoff too.
    from types import SimpleNamespace
    proc = preprocessor([[low_points * 2]] * 20)
    proc.detector.face_detector = SimpleNamespace(detect_from_image=lambda frame: [np.array([0., 0., 200., 200., 1.])])
    proc.detector.get_landmarks_from_image = lambda frame, **options: [low_points * 2]
    result = proc([high] * 20)
    assert result.quality['face_width_px'] == 144
    assert result.lips.shape == result.face.shape == (1, 1, 20, 88, 88)
    assert np.isfinite(result.lips).all()


def test_short_partial_does_not_reach_detector():
    proc = preprocessor([])
    with pytest.raises(FaceQualityError, match='0.8 and 80 seconds'):
        proc([np.zeros((256, 256, 3), dtype=np.uint8)] * 10)


def test_20fps_clip_is_resampled_without_shortening_model_duration():
    proc = preprocessor([[reference_face()]] * 25)
    result = proc([np.full((256, 256, 3), 128, dtype=np.uint8)] * 20, fps=20)
    assert result.face.shape[2] == 25
    assert result.lips.shape[2] == 25

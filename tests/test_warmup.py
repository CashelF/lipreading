"""Cold initialization must happen before readiness, using no camera data."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from backend.model import FlareModel
from backend.preprocessing import FacePreprocessor


def test_landmark_warmup_uses_only_blank_frames_and_explicit_boxes():
    preprocessor = FacePreprocessor.__new__(FacePreprocessor)
    preprocessor.mean, preprocessor.std, preprocessor.crop_size = .421, .165, 88
    detector = Mock()
    preprocessor.detector = detector
    preprocessor.warmup()
    assert detector.face_detector.detect_from_image.call_count == 8
    assert detector.get_landmarks_from_image.call_count == 8
    assert all(call.args[0].shape[1] <= 640 and call.args[0].shape[0] <= 480
               for call in detector.face_detector.detect_from_image.call_args_list)
    assert [call.args[0].shape for call in detector.get_landmarks_from_image.call_args_list] == [
        (360, 640, 3), (360, 640, 3), (480, 640, 3), (480, 640, 3),
        (720, 1280, 3), (720, 1280, 3), (960, 1280, 3), (960, 1280, 3)]
    for call in detector.get_landmarks_from_image.call_args_list:
        assert call.args[0].dtype == np.uint8
        assert not call.args[0].any()
        box = call.kwargs['detected_faces'][0]
        assert box.dtype == np.float64
        np.testing.assert_array_equal(box, [160., 60., 480., 340., 1.])


def test_loaded_weights_are_not_ready_during_warmup():
    runtime = FlareModel()
    runtime.model = object()
    runtime._loading = True
    assert runtime.status()['ready'] is False
    assert runtime.status()['state'] == 'loading'
    runtime._loading = False
    assert runtime.status()['ready'] is True


def test_model_warmup_preserves_search_settings_and_discards_results(monkeypatch):
    monkeypatch.setenv('FLARE_BEAM_SIZE', '30')
    runtime = FlareModel()
    runtime.model = object()
    runtime.preprocessor = SimpleNamespace(warmup=Mock())
    runtime._search = Mock(return_value=(['never display warmup'], []))
    torch = SimpleNamespace(
        inference_mode=nullcontext, zeros=Mock(return_value=object()),
        ones=Mock(return_value=object()), bfloat16='bf16', bool='bool',
        cuda=SimpleNamespace(synchronize=Mock()))
    assert runtime._warmup(torch) is None
    runtime.preprocessor.warmup.assert_called_once()
    args, kwargs = runtime._search.call_args
    assert args[0] == [runtime.model]
    assert kwargs == dict(beam_size=30, audio_only=False, video_only=True, temp=1.2)
    assert runtime.beam == 30
    torch.zeros.assert_any_call((1, 1, 51, 88, 88), device='cuda', dtype='bf16')
    torch.zeros.assert_any_call((1, 1, 52, 88, 88), device='cuda', dtype='bf16')
    assert runtime._search.call_count == 2
    torch.cuda.synchronize.assert_called_once()

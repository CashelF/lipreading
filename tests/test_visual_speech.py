"""Contract and official-path parity for the offline visual ASD adapter."""
import numpy as np
import pytest

from backend.visual_speech import (
    PROJECT_ROOT, VisualSpeechDetector, crop_faces, _verify_digest,
)


@pytest.mark.parametrize('offset', [(0, 0), (-80, -40), (100, 90)])
def test_crop_matches_official_geometry_without_codec_roundtrip(offset):
    cv2 = pytest.importorskip('cv2')
    from scipy.signal import medfilt

    rng = np.random.default_rng(81)
    rgb = rng.integers(0, 256, (220, 260, 3), dtype=np.uint8)
    frames = [rgb] * 25
    boxes = np.array([[80 + i / 3, 30, 170 + i / 4, 145, .99]
                      for i in range(25)])
    boxes[:, :4] += [*offset, *offset]
    sizes = medfilt(np.maximum(boxes[:, 3] - boxes[:, 1],
                              boxes[:, 2] - boxes[:, 0]) / 2, kernel_size=13)
    xs = medfilt((boxes[:, 0] + boxes[:, 2]) / 2, kernel_size=13)
    ys = medfilt((boxes[:, 1] + boxes[:, 3]) / 2, kernel_size=13)
    expected = []
    # Independent direct transcription of the upstream BGR demo crop, including
    # its asymmetric vertical extent and its final center crop (not resize).
    for bs, x, y in zip(sizes, xs, ys):
        cs = .4
        bsi = int(bs * (1 + 2 * cs))
        frame = np.pad(rgb[:, :, ::-1], ((bsi, bsi), (bsi, bsi), (0, 0)),
                       'constant', constant_values=(110, 110))
        my, mx = y + bsi, x + bsi
        face = frame[int(my-bs):int(my+bs*(1+2*cs)),
                     int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        face = cv2.resize(face, (224, 224))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        expected.append(gray[56:168, 56:168])
    np.testing.assert_array_equal(crop_faces(frames, boxes), np.stack(expected))


@pytest.mark.parametrize('boxes', [
    np.zeros((13, 4)),
    np.ones((12, 4)),
    np.full((13, 4), np.nan),
    np.tile([10, 10, 9, 20], (13, 1)),
])
def test_crop_rejects_missing_or_invalid_tracking(boxes):
    frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 13
    with pytest.raises(ValueError):
        crop_faces(frames, boxes)


def test_artifact_integrity_is_checked(tmp_path):
    artifact = tmp_path / 'encoder.py'
    artifact.write_bytes(b'wrong model')
    with pytest.raises(ValueError, match='checksum'):
        _verify_digest(artifact, '0' * 64)
    with pytest.raises(FileNotFoundError, match='setup_visual_speech'):
        _verify_digest(tmp_path / 'missing', '0' * 64)


@pytest.mark.parametrize('checkpoint', ['ava', 'talkset'])
def test_frozen_visual_adapter_matches_official_forward_path(checkpoint):
    torch = pytest.importorskip('torch')
    source = PROJECT_ROOT / 'vendor' / 'light-asd'
    if not (source / 'model' / 'Model.py').is_file():
        pytest.skip('Run scripts/setup_visual_speech.py to install official artifacts.')
    # Isolate the official import names in a subprocess: the app vendors other
    # repositories whose generic ``model`` modules must never be shadowed.
    import subprocess
    import sys
    code = '''
import sys
import numpy as np
import torch
from backend.visual_speech import VisualSpeechDetector, PROJECT_ROOT, CHECKPOINTS
torch.set_num_threads(2)
sys.path.insert(0, str(PROJECT_ROOT / 'vendor' / 'light-asd'))
from model.Model import ASD_Model
from loss import lossV
name = sys.argv[1]
model_modules = {k: v for k, v in sys.modules.items() if k == 'model' or k.startswith('model.')}
detector = VisualSpeechDetector(device='cpu', checkpoint=name)
assert model_modules == {k: v for k, v in sys.modules.items()
                         if k == 'model' or k.startswith('model.')}
state = torch.load(PROJECT_ROOT / 'models' / 'visual_speech' / CHECKPOINTS[name][0],
                   weights_only=True, map_location='cpu')
model = ASD_Model().eval()
head = lossV().eval()
model.load_state_dict({k[len('model.'):]: v for k, v in state.items()
                       if k.startswith('model.')}, strict=True)
head.load_state_dict({k[len('lossV.'):]: v for k, v in state.items()
                      if k.startswith('lossV.')}, strict=True)
crops = np.random.default_rng(42).integers(0, 256, (19, 112, 112), dtype=np.uint8)
with torch.inference_mode():
    visual = model.forward_visual_frontend(torch.tensor(crops, dtype=torch.float32)[None])
    out = model.forward_visual_backend(visual)
    expected = head.FC(out).softmax(dim=-1)[:, 1].numpy()
actual = detector.predict_crops(crops)
np.testing.assert_array_equal(actual, expected)
assert len(actual) == len(crops) and np.isfinite(actual).all()
assert (actual >= 0).all() and (actual <= 1).all()
assert not detector.encoder.training and not detector.classifier.training
assert all(not p.requires_grad for p in detector.encoder.parameters())
assert all(not p.requires_grad for p in detector.classifier.parameters())
'''
    subprocess.run([sys.executable, '-c', code, checkpoint], cwd=PROJECT_ROOT,
                   check=True, capture_output=True, text=True, timeout=60)

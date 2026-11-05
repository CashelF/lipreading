"""Frozen visual speaking detector used by the live gate and offline checks.

Uses Light-ASD's actually supervised visual-only head. Unlike TalkNet's lossV,
Light-ASD's lossV is trained before any audio fusion. Published Light-ASD ASD
benchmark scores describe its audiovisual model, not this auxiliary head.

Inputs are RGB frames already sampled at 25 fps, plus one SFD face box per
frame. The crop follows the authors' Columbia_test.py demo. No microphone,
landmark-motion fallback or FLARE decoding is used here. Session-level caption
admission is implemented separately in speech_gate.py.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REPOSITORY = 'https://github.com/Junhua-Liao/Light-ASD'
UPSTREAM_COMMIT = 'ed38c232de5efe0261dbd68627c0ade7cdfe14eb'
ENCODER_SHA256 = 'c4a85b8aa5eaab428f79836c5d9511a01ac45bfeb6dccc75e06c1e9eff25520c'
CHECKPOINTS = {
    'talkset': ('finetuning_TalkSet.model',
                'efc375833887eefa9d209dc92810e18519b04c3c73ea35a549f2a7f40b7d94d5'),
    'ava': ('pretrain_AVA_CVPR.model',
            'd44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317'),
}


def _verify_digest(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} is missing. Run .venv-model/bin/python scripts/setup_visual_speech.py.')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Unexpected Light-ASD artifact checksum: {path.name}')


def crop_faces(frames: list[np.ndarray], boxes: np.ndarray, *, cache=None) -> np.ndarray:
    """Official demo face geometry, grayscale, and center 112px crop.

    Detection rows are [x1,y1,x2,y2] (an optional score is ignored). Frames and
    detections must correspond one-to-one at 25 fps. Track interpolation belongs
    to the caller; missing/multiple faces must not silently become predictions.
    The official demo writes and rereads an AVI between cropping and grayscale;
    this implementation keeps pixels in memory without adding codec loss.
    """
    import cv2
    from scipy.signal import medfilt

    if len(frames) < 13:
        raise ValueError('Light-ASD needs at least 13 frames for its face crop smoothing.')
    if any(frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
           for frame in frames):
        raise ValueError('Expected uint8 RGB frames.')
    boxes = np.asarray(boxes, dtype=np.float64)
    if (boxes.ndim != 2 or boxes.shape[0] != len(frames)
            or boxes.shape[1] not in (4, 5) or not np.isfinite(boxes).all()
            or np.any(boxes[:, 2:4] <= boxes[:, :2])):
        raise ValueError('Expected one finite, nonempty SFD face box per frame.')

    sizes = medfilt(np.maximum(boxes[:, 3] - boxes[:, 1],
                              boxes[:, 2] - boxes[:, 0]) / 2, kernel_size=13)
    centers_x = medfilt((boxes[:, 0] + boxes[:, 2]) / 2, kernel_size=13)
    centers_y = medfilt((boxes[:, 1] + boxes[:, 3]) / 2, kernel_size=13)
    result = []
    scale = .40
    for frame, size, cx, cy in zip(frames, sizes, centers_x, centers_y):
        pad = int(size * (1 + 2 * scale))
        # Malformed detections should not request unbounded image padding.
        if size < 1 or pad > max(frame.shape[:2]) * 2:
            raise ValueError('Face box is too small or outside the supported image extent.')
        mx, my = cx + pad, cy + pad
        x1, x2 = int(mx - size * (1 + scale)), int(mx + size * (1 + scale))
        y1, y2 = int(my - size), int(my + size * (1 + 2 * scale))
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] + 2 * pad or y2 > frame.shape[0] + 2 * pad:
            raise ValueError('Face crop extends beyond the official padded image.')
        entry = cache.entry(frame) if cache is not None else None
        key = (pad, x1, y1, x2, y2)
        if entry is not None and key in entry.speech_crops:
            result.append(entry.speech_crops[key])
            entry.speech_crops.move_to_end(key)
            continue
        # Materialize only the requested rectangle of the padded image. Pixel
        # values and resize geometry are unchanged, including out-of-frame crops.
        left, right, top, bottom = x1 - pad, x2 - pad, y1 - pad, y2 - pad
        height, width = frame.shape[:2]
        if left >= 0 and top >= 0 and right <= width and bottom <= height:
            rectangle = frame[top:bottom, left:right]
        else:
            rectangle = np.full((y2-y1, x2-x1, 3), 110, dtype=frame.dtype)
            sx, ex, sy, ey = max(0, left), min(width, right), max(0, top), min(height, bottom)
            if ex > sx and ey > sy:
                rectangle[sy-top:ey-top, sx-left:ex-left] = frame[sy:ey, sx:ex]
        face = cv2.resize(rectangle, (224, 224))
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        roi = gray[56:168, 56:168]
        if entry is not None:
            roi = roi.copy()
            roi.setflags(write=False)
            entry.speech_crops[key] = roi
            while len(entry.speech_crops) > 4:
                entry.speech_crops.popitem(last=False)
        result.append(roi)
    return np.stack(result)


class VisualSpeechDetector:
    """Frozen float32 Light-ASD visual encoder and its trained 2-class head.

    ``predict`` returns one raw softmax speaking score per input frame. Scores
    are not calibrated probabilities for this webcam and do not make a gating
    decision. Compare checkpoints and thresholds on separate labeled recordings.
    The three 5-frame temporal convolutions have a radius of six frames. Face
    crop smoothing needs another six: raw-frame chunks require twelve frames of
    context on either side to avoid either operation's boundary effects.
    """

    temporal_radius_frames = 6
    crop_radius_frames = 6
    fps = 25
    supports_crop_cache = True

    def __init__(self, device: str = 'cpu', checkpoint: str = 'ava', *,
                 source_dir: Path | None = None, weights_dir: Path | None = None):
        import torch

        if checkpoint not in CHECKPOINTS:
            raise ValueError(f'Unknown Light-ASD checkpoint: {checkpoint}')
        source_dir = Path(source_dir or PROJECT_ROOT / 'vendor' / 'light-asd')
        weights_dir = Path(weights_dir or PROJECT_ROOT / 'models' / 'visual_speech')
        filename, digest = CHECKPOINTS[checkpoint]
        encoder_path = source_dir / 'model' / 'Encoder.py'
        weight_path = weights_dir / filename
        _verify_digest(encoder_path, ENCODER_SHA256)
        _verify_digest(weight_path, digest)

        # Load just the self-contained encoder file, never a generic top-level
        # ``model`` package that could collide with FLARE or other vendored code.
        spec = importlib.util.spec_from_file_location('_still_light_asd_encoder', encoder_path)
        if spec is None or spec.loader is None:
            raise ImportError('Could not load the pinned Light-ASD visual encoder.')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        encoder = module.visual_encoder()
        classifier = torch.nn.Linear(128, 2)
        state = torch.load(weight_path, map_location='cpu', weights_only=True)
        prefix = 'model.visualEncoder.'
        encoder_state = {key[len(prefix):]: value for key, value in state.items()
                         if key.startswith(prefix)}
        head_state = {key[len('lossV.FC.'):]: value for key, value in state.items()
                      if key.startswith('lossV.FC.')}
        # Strict loads reject missing/random visual head or encoder parameters.
        encoder.load_state_dict(encoder_state, strict=True)
        classifier.load_state_dict(head_state, strict=True)
        self.device = torch.device(device)
        self.encoder = encoder.to(device=self.device, dtype=torch.float32).eval()
        self.classifier = classifier.to(device=self.device, dtype=torch.float32).eval()
        self.encoder.requires_grad_(False)
        self.classifier.requires_grad_(False)
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = digest

    def predict_crops(self, crops: np.ndarray) -> np.ndarray:
        """Score raw grayscale uint8 T×112×112 crops at 25 fps."""
        import torch

        crops = np.asarray(crops)
        if (crops.ndim != 3 or crops.shape[1:] != (112, 112)
                or not 1 <= len(crops) <= 400 or crops.dtype != np.uint8):
            raise ValueError('Expected 1–400 uint8 grayscale crops of shape T×112×112.')
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            x = torch.from_numpy(np.ascontiguousarray(crops)).to(
                device=self.device, dtype=torch.float32)[None, None]
            x = (x / 255 - .4161) / .1688
            embeddings = self.encoder(x)
            logits = self.classifier(embeddings.reshape(-1, 128))
            return logits.softmax(dim=-1)[:, 1].cpu().numpy()

    def predict(self, frames: list[np.ndarray], boxes: np.ndarray, *, cache=None) -> np.ndarray:
        return self.predict_crops(crop_faces(frames, boxes, cache=cache))

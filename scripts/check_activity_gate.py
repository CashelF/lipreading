#!/usr/bin/env python3
"""Diagnostic fixtures for the heuristic gate, not a speech accuracy benchmark."""
from __future__ import annotations
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from PIL import Image
    from skimage import data
    import cv2
    from backend.preprocessing import FacePreprocessor

    torch.set_num_threads(8)
    processor = FacePreprocessor(ROOT / 'models/20words_mean_face.npy')
    processor.warmup()
    source = cv2.resize(data.astronaut()[:360, 90:450], (512, 512), interpolation=cv2.INTER_LINEAR)
    rng = np.random.default_rng(17)

    def jpeg(frame):
        buffer = BytesIO()
        Image.fromarray(frame).save(buffer, format='JPEG', quality=92)
        buffer.seek(0)
        with Image.open(buffer) as image:
            return np.asarray(image.convert('RGB'))

    fixtures = {
        'still_photo': [jpeg(source) for _ in range(51)],
        'translated_still_photo': [jpeg(np.roll(source, (i % 5 - 2, i % 7 - 3), axis=(0, 1))) for i in range(51)],
        'still_photo_with_sensor_noise': [jpeg(np.clip(source.astype(np.float32) + rng.normal(0, 1, source.shape), 0, 255).astype(np.uint8)) for _ in range(51)],
    }
    # A positive pipeline check: animate lip opening and add camera-like noise.
    # This is an artificial warp, not real speech or a word-accuracy benchmark.
    mouth = processor.detector.get_landmarks_from_image(source)[0][48:68, :2]
    center = mouth.mean(axis=0)
    extent = np.ptp(mouth, axis=0)
    x, y = np.meshgrid(np.arange(512, dtype=np.float32), np.arange(512, dtype=np.float32))
    envelope = np.exp(-((x - center[0]) / (extent[0] * .7)) ** 4
                      - ((y - center[1]) / (extent[1] + 8)) ** 4)
    animated = []
    for index in range(51):
        opening = 5 * np.sin(index * np.pi / 13) ** 2
        moved = cv2.remap(source, x, y - opening * np.tanh((y - center[1]) / 2) * envelope,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        noisy = np.clip(moved.astype(np.float32) + rng.normal(0, 3, source.shape), 0, 255).astype(np.uint8)
        animated.append(jpeg(noisy))
    fixtures['synthetic_articulation_with_noise'] = animated
    results = {}
    for name, frames in fixtures.items():
        prepared = processor(frames, speech_gate=True)
        results[name] = prepared.activity
        print(json.dumps({'fixture': name, 'activity': prepared.activity}), flush=True)
    output = ROOT / 'models/activity-gate-check.json'
    output.write_text(json.dumps({'description': 'Public-photo stillness and artificial-articulation diagnostics; speech recall must be tested separately.', 'results': results}, indent=2) + '\n')
    assert results['still_photo']['suppress'], 'The exact still-photo fixture must be rejected.'
    assert results['still_photo_with_sensor_noise']['suppress'], 'Low sensor noise should not invent speech.'
    assert results['translated_still_photo']['suppress'], 'Head translation must not invent speech.'
    assert not results['synthetic_articulation_with_noise']['suppress'], 'Visible articulation must reach decoding despite image noise.'


if __name__ == '__main__':
    main()

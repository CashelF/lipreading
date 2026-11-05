#!/usr/bin/env python3
"""Local comparison on consented recordings; never changes the live gate.

Replay the actual timestamped JPEG packets through ClipBuffer/resample. Scores
are descriptive: clip labels do not provide frame-level speech annotations and
the model's softmax is not a calibrated probability for this user's camera.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.stream import ClipBuffer, DecodeCache, resample, sample_times


def replay_windows(directory: Path, manifest: dict):
    """Use only frames that would have arrived at each live inference request."""
    names, timestamps = manifest['frame_files'], manifest['timestamps_ms']
    if not names or len(names) != len(timestamps) or len(names) != manifest['frames']:
        raise ValueError('Recording manifest has inconsistent frame counts.')
    buffer = ClipBuffer()
    previous_end = None
    for name, timestamp in zip(names, timestamps):
        if not isinstance(name, str) or not re.fullmatch(r'\d{6}\.jpg', name):
            raise ValueError('Unexpected frame filename.')
        packet = struct.pack('<d', timestamp) + (directory / name).read_bytes()
        clip = buffer.append(packet)
        if clip is not None:
            previous_end = clip.end_ms
            yield clip
    final = buffer.finish()
    if final is not None and final.end_ms != previous_end:
        yield final


def score_summary(values):
    import numpy as np
    return {name: round(float(value), 6) for name, value in {
        'min': np.min(values), 'p10': np.quantile(values, .1),
        'median': np.median(values), 'mean': np.mean(values),
        'p90': np.quantile(values, .9), 'max': np.max(values),
    }.items()}


def evaluate(directory, manifest, detectors, processor):
    import numpy as np
    from backend.preprocessing import PreprocessCache

    decoded = DecodeCache()
    preprocessing = PreprocessCache()
    # Keep a strong reference with each entry to prevent object-id reuse.
    detections = {}
    results = []
    origin = manifest['timestamps_ms'][0]
    for clip in replay_windows(directory, manifest):
        row = {'start_seconds': round((clip.start_ms - origin) / 1000, 3),
               'end_seconds': round((clip.end_ms - origin) / 1000, 3)}
        try:
            frames, capture_fps = resample(clip, cache=decoded)
            row.update(frames=len(frames), capture_fps=round(capture_fps, 2))
            row['sample_timestamps_ms'] = (sample_times(clip) - origin).tolist()
            boxes = []
            for frame in frames:
                entry = detections.get(id(frame))
                if entry is None or entry[0] is not frame:
                    found = processor.detector.face_detector.detect_from_image(frame)
                    entry = (frame, found)
                    detections[id(frame)] = entry
                if entry[1] is None or len(entry[1]) != 1:
                    raise ValueError('Exactly one detected face is required in every frame.')
                boxes.append(entry[1][0])
            row['candidates'] = {}
            for name, detector in detectors.items():
                scores = detector.predict(frames, np.asarray(boxes))
                if len(scores) != len(frames) or not np.isfinite(scores).all():
                    raise ValueError('Visual detector returned invalid scores.')
                radius = detector.temporal_radius_frames + detector.crop_radius_frames
                contextualized = scores[radius:-radius]
                row['candidates'][name] = {
                    'visual_scores': [round(float(value), 6) for value in scores],
                    'contextualized_window': score_summary(contextualized),
                    # Separate recent evidence from an old event in the six-second
                    # history. Both crop median smoothing and convolutions need
                    # six future frames, so exclude their combined boundary.
                    'latest_contextualized_second': score_summary(contextualized[-25:]),
                    'context_delay_seconds': radius / detector.fps,
                }
        except ValueError as exc:
            row['visual_error'] = str(exc)
        try:
            # Resampling errors must not leave a previous window in `frames`.
            frames, _ = resample(clip, cache=decoded)
            prepared = processor(frames, fps=25, cache=preprocessing, speech_gate=True)
            activity = prepared.activity
            row['current_gate'] = {key: activity[key] for key in ('state', 'suppress', 'detail')}
        except ValueError as exc:
            row['current_gate'] = {'error': str(exc)}
        results.append(row)
    return {'id': manifest['id'], 'label': manifest['label'], 'take': manifest['take'],
            'windows': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, default=ROOT / 'data/speaking-examples')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/visual_speech/example-report.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--checkpoint', choices=('both', 'ava', 'talkset'), default='both')
    parser.add_argument('--take', type=int, choices=(1, 2),
                        help='Evaluate one take only, keeping validation separate from tuning.')
    args = parser.parse_args()
    records = []
    for path in sorted(args.examples.glob('*/manifest.json')):
        if re.fullmatch(r'[0-9a-f]{32}', path.parent.name):
            manifest = json.loads(path.read_text())
            if args.take is None or manifest['take'] == args.take:
                records.append((path.parent, manifest))
    if not records:
        print('No saved examples yet. Open Record examples in the app first.')
        return 2

    import torch
    from backend.preprocessing import FacePreprocessor
    from backend.visual_speech import VisualSpeechDetector, UPSTREAM_COMMIT
    torch.set_num_threads(8)
    names = ('ava', 'talkset') if args.checkpoint == 'both' else (args.checkpoint,)
    detectors = {name: VisualSpeechDetector(device=args.device, checkpoint=name) for name in names}
    processor = FacePreprocessor(ROOT / 'models/20words_mean_face.npy', device=args.device)
    report = {'candidate': 'Light-ASD visual branch', 'upstream_commit': UPSTREAM_COMMIT,
              'take': args.take,
              'checkpoints': {name: detector.checkpoint_sha256 for name, detector in detectors.items()},
              'live_gate_changed': False,
              'limitations': [
                  'No automatic threshold selection or deployment.',
                  'Take 1 is for tuning; keep take 2 separate for validation.',
                  'Clip labels are instructions, not verified frame annotations.',
                  'Softmax scores are not calibrated webcam speech probabilities.',
                  'No FLARE captions or word error rates are measured.',
              ], 'recordings': []}
    for directory, manifest in records:
        result = evaluate(directory, manifest, detectors, processor)
        report['recordings'].append(result)
        windows = result['windows']
        print(json.dumps({'id': manifest['id'], 'label': manifest['label'], 'take': manifest['take'],
                          'windows': len(windows),
                          'visual_errors': sum('visual_error' in row for row in windows),
                          'current_gate_admitted': sum(row['current_gate'].get('suppress') is False for row in windows)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(f'Local report: {args.output}. Live inference was not changed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

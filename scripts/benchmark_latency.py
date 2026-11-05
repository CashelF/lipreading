#!/usr/bin/env python3
"""Compare exact outputs and latency of the original and optimized GPU paths.

Uses the public-domain astronaut photo bundled with scikit-image, with small
integer translations. This checks numerical/output equivalence and synthetic
workload speed, NOT speech accuracy, word error rate, or live camera latency.
No camera frames or user transcripts are read, saved, or sent anywhere.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOGGER = logging.getLogger('lipreading.benchmark')


def parse_lengths(value: str) -> list[int]:
    try:
        lengths = [int(n) for n in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated frame counts, e.g. 51,101,151.') from exc
    if not lengths or any(n < 20 or n > 400 for n in lengths):
        raise argparse.ArgumentTypeError('Frame counts must be between 20 and 400.')
    if lengths != sorted(set(lengths)):
        raise argparse.ArgumentTypeError('Frame counts must be unique and increasing for overlapping-clip reuse.')
    return lengths


def fixture_frames(count: int):
    """Unique immutable arrays, retaining object identity across clip prefixes."""
    import cv2
    import numpy as np
    from skimage import data
    # Crop away the astronaut's small suit patches; enlarge the real face enough
    # to pass the unchanged 100px face-width admission check.
    source = cv2.resize(data.astronaut()[:360, 90:450], (512, 512), interpolation=cv2.INTER_LINEAR)
    padded = np.pad(source, ((4, 4), (4, 4), (0, 0)), mode='edge')
    frames = []
    for index in range(count):
        dx = (index // 3) % 9 - 4
        dy = (index // 11) % 3 - 1
        frame = padded[4 + dy:516 + dy, 4 + dx:516 + dx].copy()
        frame.setflags(write=False)
        frames.append(frame)
    return frames


def digest_tree(value) -> str:
    """Hash exact tensor bytes, shapes, dtypes, and container boundaries."""
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            cpu = item.detach().cpu().contiguous()
            digest.update(f'tensor:{cpu.dtype}:{tuple(cpu.shape)}:'.encode())
            digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, (tuple, list)):
            digest.update(f'{type(item).__name__}:{len(item)}:['.encode())
            for child in item:
                visit(child)
            digest.update(b']')
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def synchronize():
    import torch
    torch.cuda.synchronize()


def timed(function):
    synchronize()
    begin = time.perf_counter()
    result = function()
    synchronize()
    return result, (time.perf_counter() - begin) * 1000


def gpu_sample(prepared):
    import torch
    # Copy the ndarray values exactly; no dtype or layout changes from production.
    face, lips = [torch.from_numpy(array).to(device='cuda', dtype=torch.bfloat16)
                  for array in (prepared.face, prepared.lips)]
    return {'net_input': {'source': {'audio': None, 'video': (face, lips),
                                    'face_lip_mask': (torch.ones(1, device='cuda'), torch.ones(1, device='cuda'))},
                          'padding_mask': torch.zeros((1, face.shape[2]), dtype=torch.bool, device='cuda')}}


@contextmanager
def trace_search(model, search):
    """Identical instrumentation for each path; record actual returned beam data."""
    metrics = {'encoder_ms': 0.0, 'projection_ms': 0.0, 'beam_ms': 0.0}
    traces = []
    encoder = model.encoder.forward
    projection = model.modality_projection.forward
    beam = search.__globals__['beam_search']

    def wrap(function, name):
        def call(*args, **kwargs):
            result, duration = timed(lambda: function(*args, **kwargs))
            metrics[name] += duration
            return result
        return call

    def traced_beam(*args, **kwargs):
        result, duration = timed(lambda: beam(*args, **kwargs))
        metrics['beam_ms'] += duration
        selected, score, sequences, scores = result
        # Hashing incurs GPU-to-CPU transfer outside the beam timer. It is the
        # same validation overhead for both paths and is reported separately.
        synchronize()
        start = time.perf_counter()
        traces.append({'tokens': digest_tree((selected, sequences)),
                       'scores': digest_tree((score, scores)),
                       'completed_candidates': len(sequences),
                       'selected_length': selected.numel()})
        metrics['validation_ms'] = metrics.get('validation_ms', 0.0) + (time.perf_counter() - start) * 1000
        return result

    model.encoder.forward = wrap(encoder, 'encoder_ms')
    model.modality_projection.forward = wrap(projection, 'projection_ms')
    search.__globals__['beam_search'] = traced_beam
    try:
        yield metrics, traces
    finally:
        model.encoder.forward = encoder
        model.modality_projection.forward = projection
        search.__globals__['beam_search'] = beam


def run_path(runtime, search, frames, cache):
    import torch
    prepared, preprocess_ms = timed(lambda: runtime.preprocessor(frames, 25, cache=cache))
    sample, transfer_ms = timed(lambda: gpu_sample(prepared))
    with trace_search(runtime.model, search) as (metrics, traces), torch.inference_mode():
        output, search_ms = timed(lambda: search([runtime.model], sample, beam_size=30,
                                                audio_only=False, video_only=True, temp=1.2))
    validation_ms = metrics.get('validation_ms', 0.0)
    durations = {'preprocess_ms': preprocess_ms, 'transfer_ms': transfer_ms,
                 'search_ms': search_ms - validation_ms, **metrics}
    durations['total_ms'] = preprocess_ms + transfer_ms + search_ms - validation_ms
    return prepared, output, traces, {key: round(value, 3) for key, value in durations.items()}


def identical_arrays(first, second) -> bool:
    return (first.dtype == second.dtype and first.shape == second.shape
            and first.tobytes() == second.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=parse_lengths, default=[51, 101, 151],
                        help='Increasing frame counts; default 51,101,151. Use 51 for a quicker check.')
    parser.add_argument('--output', type=Path, help='Write JSON report here, e.g. models/latency-benchmark.json.')
    parser.add_argument('--warmups', type=int, default=1,
                        help='Unreported full passes through EACH path before measurement (default 1).')
    parser.add_argument('--cpu-threads', type=int,
                        help='Explicit common PyTorch CPU thread count; default preserves runtime defaults.')
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error('--warmups must be nonnegative')
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')

    import numpy as np
    import torch
    from backend.decoding import build_search
    from backend.model import FlareModel
    from backend.preprocessing import PreprocessCache
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU is required for this benchmark.')
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    np.random.seed(0)
    logging.basicConfig(level=logging.WARNING)
    runtime = FlareModel()
    runtime.beam = 30
    print('Loading a separate model for comparison; live app is not modified.', file=sys.stderr, flush=True)
    runtime.load()
    source = runtime.runtime / 'decode_scripts/infer_s2s_beamsearch_s3_chime_llm.py'
    baseline = build_search(source, LOGGER, optimized=False)
    optimized = build_search(source, LOGGER, optimized=True)
    frames = fixture_frames(max(args.lengths))

    for number in range(args.warmups):
        print(f'Warmup {number + 1}/{args.warmups}: baseline and optimized.', file=sys.stderr, flush=True)
        run_path(runtime, baseline, frames[:args.lengths[0]], None)
        run_path(runtime, optimized, frames[:args.lengths[0]], PreprocessCache())

    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'fixture': 'scikit-image astronaut crop, deterministic integer translations; no speech ground truth',
        'limitations': ['Synthetic workload, not live camera latency or a speech accuracy benchmark.',
                        'Exact equality on these samples is regression evidence, not proof for every possible video.',
                        'Timing includes identical stage synchronization; validation hashing is subtracted.',
                        'Other processes may contend for the GPU.'],
        'device': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
        'cpu_threads': torch.get_num_threads(), 'beam_size': 30, 'temperature': 1.2,
        'fps': 25, 'warmups_per_path': args.warmups, 'measurements': [],
    }
    cache = PreprocessCache()
    for index, count in enumerate(args.lengths):
        print(f'Comparing {count} frames (unchanged beam 30).', file=sys.stderr, flush=True)
        # Alternate order to reduce one-sided thermal/cache bias. The optimized
        # cache persists across growing prefixes; the baseline has no reuse.
        paths = [('baseline', baseline, None), ('optimized', optimized, cache)]
        if index % 2:
            paths.reverse()
        runs = {name: run_path(runtime, search, frames[:count], state)
                for name, search, state in paths}
        old, new = runs['baseline'], runs['optimized']
        equality = {
            'face_tensor_exact': identical_arrays(old[0].face, new[0].face),
            'lip_tensor_exact': identical_arrays(old[0].lips, new[0].lips),
            'quality_exact': old[0].quality == new[0].quality,
            'beam_token_ids_exact': [item['tokens'] for item in old[2]] == [item['tokens'] for item in new[2]],
            'beam_scores_exact': [item['scores'] for item in old[2]] == [item['scores'] for item in new[2]],
            'captions_and_nbest_exact': old[1] == new[1],
        }
        row = {'frames': count, 'capture_duration_seconds': round((count - 1) / 25, 3),
               'baseline': old[3], 'optimized': new[3], 'equivalence': equality,
               'cache': cache.stats(), 'beam_trace': new[2]}
        row['speedup'] = round(old[3]['total_ms'] / new[3]['total_ms'], 3)
        report['measurements'].append(row)
        print(json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
    report['all_exact'] = all(all(row['equivalence'].values()) for row in report['measurements'])
    report['combined_speedup'] = round(sum(row['baseline']['total_ms'] for row in report['measurements']) /
                                       sum(row['optimized']['total_ms'] for row in report['measurements']), 3)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text, flush=True)
    return 0 if report['all_exact'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

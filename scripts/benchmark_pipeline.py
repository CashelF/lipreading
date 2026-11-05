#!/usr/bin/env python3
"""Compare a saved pre-change pipeline with the current code on local examples.

Only numeric timings/equality results are written. Captions, pixels and token IDs
stay transient. Identical-window regression evidence is not a word-error-rate
benchmark. The separate model leaves the live listener unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_latency import trace_search, identical_arrays
from scripts.check_speaking_gate import load_records, packets_for_record, write_private


def load_baseline(root, name):
    module_name = f'backend._baseline_{name}'
    spec = importlib.util.spec_from_file_location(module_name, root / 'backend' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class CapturePreprocessor:
    def __init__(self, preprocessor):
        self.inner = preprocessor
        self.inputs = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        result = self.inner(*args, **kwargs)
        if not kwargs.get('speech_gate', False) and result.face is not None:
            self.inputs.append((result.face, result.lips))
        return result


async def compare(args, runtime):
    from backend.utterance_stream import UtteranceReader
    from backend.stream import ClipBuffer, PreparationSampler
    import torch
    # The installed face-alignment package enables cuDNN autotuning, whose
    # selected plans are thread-local. Compare both paths on the same worker so
    # a new worker's independent autotuning cannot masquerade as a code change.
    # Real app scheduling is tested separately by benchmark_streaming.py.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    old_pre = load_baseline(args.baseline, 'preprocessing')
    old_model = load_baseline(args.baseline, 'model')
    old_reader = load_baseline(args.baseline, 'utterance_stream')
    old_decoder = load_baseline(args.baseline, 'decoding')
    old_visual = load_baseline(args.baseline, 'visual_speech')
    old_reader.PreprocessCache = old_pre.PreprocessCache
    previous = copy.copy(runtime)
    previous.__class__ = old_model.FlareModel
    pre = old_pre.FacePreprocessor.__new__(old_pre.FacePreprocessor)
    pre.__dict__.update(runtime.preprocessor.__dict__)
    pre.detector = copy.copy(pre.detector)
    pre.detector.face_alignment_net = getattr(pre.detector.face_alignment_net, 'network', pre.detector.face_alignment_net)
    pre.detector.face_detector = copy.copy(pre.detector.face_detector)
    sfd = pre.detector.face_detector
    sfd.face_detector = getattr(sfd.face_detector, 'network', sfd.face_detector)
    pre.visual_speech = copy.copy(pre.visual_speech)
    pre.visual_speech.__class__ = old_visual.VisualSpeechDetector
    previous.preprocessor = CapturePreprocessor(pre)
    runtime.preprocessor = CapturePreprocessor(runtime.preprocessor)
    for host in (previous, runtime):
        visual = host.preprocessor.inner.visual_speech
        original = visual.predict_crops
        visual._benchmark_trace = []
        def traced_crops(crops, original=original, visual=visual):
            result = original(crops)
            visual._benchmark_trace.append((hashlib.sha256(crops.tobytes()).hexdigest(), result.copy()))
            return result
        visual.predict_crops = traced_crops
    previous._search = old_decoder.build_search(runtime.runtime / 'decode_scripts/infer_s2s_beamsearch_s3_chime_llm.py', logging.getLogger('baseline'))
    report = {'same_input_windows': True, 'caption_text_saved': False,
              'comparison_worker_threads': 1,
              'early_preparation': args.prefetch,
              'limitations': ['Saved development examples, not held-out speech ground truth.',
                              'Real-time transport/scheduling are measured separately.',
                              'Trace hashing overhead is excluded equally from both paths.'], 'rows': []}
    records = list(load_records().values())
    if args.takes:
        records = [(d, m) for d, m in records if m['take'] in args.takes]
    async def send(message):
        pass
    for directory, manifest in records:
        with urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=3) as response:
            if json.load(response).get('busy'):
                raise RuntimeError('User session active; stopping the separate benchmark.')
        readers = {'before': old_reader.UtteranceReader(previous), 'after': UtteranceReader(runtime)}
        runtimes = {'before': previous, 'after': runtime}
        last_decoded = {}
        prefetched = set()
        buffer = ClipBuffer()
        sampler = PreparationSampler()
        early_sources = []
        windows = []
        for _, packet in packets_for_record(directory, manifest):
            clip = buffer.append(packet)
            for reader in readers.values():
                reader.history.append(buffer.frames[-1])
            early_sources.extend(sampler.append(buffer.frames[-1]))
            if clip:
                windows.append((clip, early_sources))
                early_sources = []
        windows.append((buffer.finish(), early_sources))
        for index, (clip, early_sources) in enumerate(windows):
            if clip is None:
                continue
            data = {}
            order = ['before', 'after'] if index % 2 == 0 else ['after', 'before']
            for name in order:
                host, reader = runtimes[name], readers[name]
                early_ms = 0.
                if name == 'after' and args.prefetch:
                    started = time.perf_counter()
                    for source in early_sources:
                        if id(source) not in prefetched:
                            await asyncio.to_thread(reader.prepare_frame, source, True)
                            prefetched.add(id(source))
                    early_ms = (time.perf_counter() - started) * 1000
                host.preprocessor.inputs = []
                host.preprocessor.inner.visual_speech._benchmark_trace = []
                torch.cuda.synchronize()
                with trace_search(host.model, host._search) as (metrics, traces):
                    begin = time.perf_counter()
                    result = await reader.process(clip, True, send)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - begin) * 1000 - metrics.get('validation_ms', 0.)
                captured_inputs = host.preprocessor.inputs
                if result['timings_ms'].get('recognition_reused'):
                    if traces:
                        raise RuntimeError('This benchmark needs a separate fixture for mixed cached/new decodes.')
                    captured_inputs, traces = last_decoded[name]
                elif traces:
                    last_decoded[name] = (captured_inputs, traces)
                data[name] = (result, captured_inputs, traces, elapsed, metrics,
                              host.preprocessor.inner.visual_speech._benchmark_trace, early_ms)
            old, new = data['before'], data['after']
            equal = {'snapshot': old[0]['snapshot'] == new[0]['snapshot'],
                     'activity': old[0]['activity'] == new[0]['activity'],
                     'intervals': old[0]['decoded_intervals'] == new[0]['decoded_intervals'],
                     'model_inputs': len(old[1]) == len(new[1]) and all(
                         identical_arrays(a, b) for pair_a, pair_b in zip(old[1], new[1])
                         for a, b in zip(pair_a, pair_b)),
                     'beam_tokens_and_scores': old[2] == new[2]}
            beam_detail = {'tokens_equal': [x['tokens'] for x in old[2]] == [x['tokens'] for x in new[2]],
                           'scores_equal': [x['scores'] for x in old[2]] == [x['scores'] for x in new[2]],
                           'before_lengths': [x['selected_length'] for x in old[2]],
                           'after_lengths': [x['selected_length'] for x in new[2]],
                           'before_candidates': [x['completed_candidates'] for x in old[2]],
                           'after_candidates': [x['completed_candidates'] for x in new[2]]}
            row = {'label': manifest['label'], 'take': manifest['take'], 'end_ms': clip.end_ms,
                   'final': clip.final, 'equivalent': equal,
                   'before_ms': old[3], 'after_ms': new[3],
                   'early_preparation_ms': new[6],
                   'beam_comparison': beam_detail,
                   'before_stages': old[0]['timings_ms'], 'after_stages': new[0]['timings_ms'],
                   'before_decoder_ms': old[4].get('beam_ms', 0), 'after_decoder_ms': new[4].get('beam_ms', 0)}
            if not equal['activity']:
                import numpy as np
                row['activity_differences'] = {key: [old[0]['activity'].get(key), new[0]['activity'].get(key)]
                                              for key in old[0]['activity'] if old[0]['activity'].get(key) != new[0]['activity'].get(key)}
                row['speech_crop_pixels_equal'] = [a[0] for a in old[5]] == [a[0] for a in new[5]]
                row['speech_score_max_diff'] = [float(np.max(np.abs(a[1] - b[1]))) for a, b in zip(old[5], new[5])]
            report['rows'].append(row)
            write_private(args.output, report)
            print(json.dumps({'label': row['label'], 'take': row['take'], 'end_ms': clip.end_ms,
                              'equivalent': equal, 'before_ms': round(old[3]), 'after_ms': round(new[3])}), flush=True)
            if not all(equal.values()):
                print(json.dumps({k: v for k, v in row.items() if k in ['activity_differences', 'speech_crop_pixels_equal', 'speech_score_max_diff']}), flush=True)
                raise RuntimeError('Exact pipeline comparison failed; see numeric report.')
        for reader in readers.values():
            reader.clear()
    report['passed'] = True
    write_private(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/visual_speech/pipeline-speed-equivalence.json')
    parser.add_argument('--takes', nargs='+', type=int)
    parser.add_argument('--prefetch', action='store_true', help='Prepare source frames before candidate windows; report this work separately.')
    args = parser.parse_args()
    with urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=3) as response:
        if json.load(response).get('busy'):
            raise SystemExit('Wait for the active reading session before benchmarking.')
    from backend.model import FlareModel
    runtime = FlareModel()
    runtime.load()
    asyncio.run(compare(args, runtime))


if __name__ == '__main__':
    main()

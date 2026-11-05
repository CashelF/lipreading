#!/usr/bin/env python3
"""Offline-only probe of visual evidence on consented local examples.

Loads a separate local model; never modifies the live server or its policy.
Forces quiet examples through recognition to study known non-speech errors.
Reports numeric diagnostics only, not videos, token IDs or recognized words.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import logging
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.grounding import (capture_candidate, control_sample, summarize_grounding,
                               token_log_probabilities, visual_prefix)
from backend.model import FlareModel
from backend.stream import Clip, Frame, resample
from scripts.benchmark_latency import gpu_sample
from scripts.check_speaking_gate import load_records, packets_for_record, write_private


def evaluate(runtime, directory, manifest, seconds, controls):
    import torch
    packets = packets_for_record(directory, manifest)
    frames = tuple(Frame(timestamp, packet[8:]) for timestamp, packet in packets
                   if timestamp <= seconds * 1000)
    clip = Clip(0, frames, True, sample_origin=0.)
    rgb, _ = resample(clip)
    began = time.perf_counter()
    prepared = runtime.preprocessor(rgb, 25, speech_gate=False)
    sample = gpu_sample(prepared)
    with torch.inference_mode():
        with capture_candidate(runtime._search) as captured:
            hypotheses, nbest = runtime._search([runtime.model], sample, beam_size=30,
                                                audio_only=False, video_only=True, temp=1.2)
        if len(captured) != 1:
            raise RuntimeError('Expected one candidate from the isolated decoder.')
        candidate = captured[0]
        # Detect prompt/projection drift before relying on a contrastive score.
        rebuilt = visual_prefix(runtime.model, sample)
        if not torch.equal(candidate.prefix, rebuilt):
            raise RuntimeError('The diagnostic prefix differs from production search.')
        search_seconds = time.perf_counter() - began
        scoring_start = time.perf_counter()
        real = token_log_probabilities(runtime.model, candidate.prefix, candidate.tokens)
        baselines = {}
        for kind in controls:
            prefix = visual_prefix(runtime.model, control_sample(sample, kind))
            baselines[kind] = token_log_probabilities(runtime.model, prefix, candidate.tokens)
        torch.cuda.synchronize()
        scoring_ms = (time.perf_counter() - scoring_start) * 1000
        row = summarize_grounding(real, baselines, ended=candidate.ended)
        # Both length-normalized scores include EOS when the search completed.
        search_rescore_delta = abs(float(real.mean()) - candidate.beam_score) if candidate.ended else None
        scores = sorted({score for text, score in nbest[0]}, reverse=True)
        row.update(label=manifest['label'], take=manifest['take'], seconds=seconds,
                   frames=len(rgb), beam_mean_logp=candidate.beam_score,
                   beam_margin=scores[0] - scores[1] if len(scores) > 1 else None,
                   caption_present=bool(hypotheses[0].strip()),
                   search_rescore_delta=search_rescore_delta,
                   preprocessing_search_ms=round(search_seconds * 1000, 1),
                   grounding_ms=round(scoring_ms, 1))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--take', type=int, choices=(1, 2), nargs='+', default=[1, 2])
    parser.add_argument('--seconds', type=float, nargs='+', default=[2., 8.])
    parser.add_argument('--controls', nargs='+', choices=('freeze_first', 'freeze_middle', 'freeze_last', 'shuffle'),
                        default=['freeze_first', 'freeze_middle', 'freeze_last', 'shuffle'])
    parser.add_argument('--output', type=Path, default=ROOT / 'models/visual_speech/grounding-experiment.json')
    args = parser.parse_args()
    if any(not .8 <= value <= 12. for value in args.seconds):
        parser.error('Durations must be between .8 and 12 seconds.')
    # Avoid starting a second workload in the middle of an active user session.
    with urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=5) as response:
        if json.load(response).get('busy'):
            raise SystemExit('A live session is active; wait until it is idle before this experiment.')
    logging.basicConfig(level=logging.WARNING)
    runtime = FlareModel()
    runtime.load()
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'caption_text_saved': False,
              'production_policy_changed': False, 'rows': [],
              'limitations': ['Quiet clips provide known non-speech cases, but no saved post-speech failure clip is available.',
                              'Speech clips have no reference words; their output is not assumed correct.',
                              'All saved takes are development examples; no suppression threshold is calibrated.',
                              'Frozen/shuffled inputs are artificial controls and can change features beyond speech content.']}
    for directory, manifest in sorted(load_records().values(), key=lambda item: (item[1]['take'], item[1]['label'])):
        if manifest['take'] not in args.take:
            continue
        for seconds in args.seconds:
            row = evaluate(runtime, directory, manifest, seconds, args.controls)
            report['rows'].append(row)
            write_private(args.output, report)
            print(json.dumps({key: row[key] for key in ('label', 'take', 'seconds', 'content_tokens',
                                                      'beam_mean_logp', 'controls', 'grounding_ms')}), flush=True)
            gc.collect()
    print(f'Local numeric report: {args.output}', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Local probe: score copied captions against single/repeated saved video.

Constructed repeats are diagnostic controls, not reference transcriptions or
proof of accuracy on naturally repeated speech. Save numeric summaries only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.grounding import FROZEN_CONTROLS, control_sample, token_log_probabilities, visual_prefix
from backend.model import FlareModel
from backend.repetition import repeated_token_spans, repetition_evidence
from backend.stream import Clip, Frame, resample
from scripts.benchmark_latency import gpu_sample
from scripts.check_speaking_gate import load_records, packets_for_record, write_private


def extend_sample(sample, repeat):
    import torch
    net = sample['net_input']
    videos = net['source']['video']
    if repeat:
        changed = tuple(torch.cat((video, video, video), dim=2) for video in videos)
    else:
        changed = tuple(torch.cat((video, video[:, :, -1:].expand(-1, -1, video.shape[2] * 2, -1, -1)), dim=2)
                        for video in videos)
    return {**sample, 'net_input': {**net, 'source': {**net['source'], 'video': changed},
            'padding_mask': torch.zeros((1, changed[0].shape[2]), device=videos[0].device, dtype=torch.bool)}}


def evaluate(runtime, directory, manifest, seconds):
    import torch
    packets = packets_for_record(directory, manifest)
    frames = tuple(Frame(timestamp, packet[8:]) for timestamp, packet in packets if timestamp <= seconds * 1000)
    rgb, _ = resample(Clip(0, frames, True, sample_origin=0.))
    sample = gpu_sample(runtime.preprocessor(rgb, 25, speech_gate=False))
    model = runtime.model
    with torch.inference_mode():
        hypotheses, _ = runtime._search([model], sample, beam_size=30, audio_only=False, video_only=True, temp=1.2)
        phrase = hypotheses[0].strip()
        if not phrase:
            return []
        repeated = ' '.join([phrase] * 3)
        encoded = model.llm_tokenizer(repeated, add_special_tokens=False, return_offsets_mapping=True)
        tokens = torch.tensor(encoded['input_ids'] + [model.llm_tokenizer.eos_token_id], device='cuda')
        spans = [[i for i, (a, b) in enumerate(encoded['offset_mapping'])
                  if a < (copy + 1) * (len(phrase) + 1) - 1 and b > copy * (len(phrase) + 1)]
                 for copy in range(3)]
        rows = []
        for kind, video in [('once', sample), ('once_then_frozen', extend_sample(sample, False)),
                            ('three_visual_copies', extend_sample(sample, True))]:
            prefix = visual_prefix(model, video)
            real = token_log_probabilities(model, prefix, tokens)
            controls = {name: token_log_probabilities(model, visual_prefix(model, control_sample(video, name)), tokens)
                        for name in FROZEN_CONTROLS}
            lifts = [[float((real[indices] - scores[indices]).mean()) for indices in spans]
                     for scores in controls.values()]
            rows.append({'take': manifest['take'], 'label': manifest['label'], 'seconds': seconds,
                         'kind': kind, 'base_words': len(phrase.split()),
                         'tokens_per_copy': [len(indices) for indices in spans],
                         'max_lift_per_copy': [max(values[i] for values in lifts) for i in range(3)],
                         'lift_by_control': dict(zip(FROZEN_CONTROLS, lifts)),
                         'policy_result': repetition_evidence(repeated_token_spans(model.llm_tokenizer, tokens[:-1].tolist()),
                                                              real, controls)})
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/visual_speech/repetition-experiment.json')
    args = parser.parse_args()
    with urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=5) as response:
        if json.load(response).get('busy'):
            raise SystemExit('Wait for the live session to finish before running this experiment.')
    runtime = FlareModel()
    runtime.load()
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'caption_text_saved': False,
              'production_policy_changed': False, 'rows': [],
              'limitations': ['Artificial repeated video and generated text are not naturally repeated speech.',
                              'Base captions have no reference transcription and may themselves contain errors.']}
    for directory, manifest in sorted(load_records().values(), key=lambda item: (item[1]['take'], item[1]['label'])):
        if not manifest['label'].startswith('speech_'):
            continue
        for row in evaluate(runtime, directory, manifest, 4.):
            report['rows'].append(row)
            write_private(args.output, report)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

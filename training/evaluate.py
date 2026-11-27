#!/usr/bin/env python3
"""Word error rate of a trained checkpoint on held-out shards, with beam search.

The report holds error counts only, never recognized text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from backend.word_errors import bootstrap_interval, corpus_wer, score
from training.data import ClipBatches, prepare
from training.model import LipReader
from training.text import Vocabulary


def load(checkpoint_path: Path, device) -> LipReader:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = LipReader(config['vocab_size'], dim=config['dim'], layers=config['layers'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--shards', type=Path, nargs='+', required=True)
    parser.add_argument('--beam', type=int, default=10)
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    device = torch.device('cuda')
    vocabulary = Vocabulary(args.vocabulary)
    model = load(args.checkpoint, device)
    dataset = ClipBatches(args.shards, args.vocabulary, frames_per_batch=6000, shuffle=False, with_text=True)
    rows = []
    for video, frames, _, _, texts in DataLoader(dataset, batch_size=None, num_workers=2):
        with torch.autocast('cuda', torch.bfloat16):
            features = model.front(prepare(video.to(device), frames))
        for index, text in enumerate(texts):
            with torch.autocast('cuda', torch.bfloat16):
                ids = model.beam_search(features[index, :frames[index]], frames[index].to(device), beam=args.beam)
            rows.append(score(text, vocabulary.decode(ids)))
        if len(rows) >= args.limit:
            break
    report = {'checkpoint': args.checkpoint.name, 'beam': args.beam, 'clips': len(rows),
              'reference_words': sum(row['reference_words'] for row in rows), 'wer': corpus_wer(rows),
              'wer_95_interval': bootstrap_interval(rows), 'captions_saved': False}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

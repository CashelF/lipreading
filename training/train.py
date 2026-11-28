#!/usr/bin/env python3
"""Train the small lipreader from the features extract_features.py saved. See docs/TRAINING.md.

Logs hold numbers only. The learning rate follows hours of video seen, so a
resumed run continues the same schedule.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from backend.word_errors import corpus_wer, score
from training.data import FeatureBatches, restore, time_mask
from training.model import FLARE_DEPTH, LipReader
from training.text import Vocabulary


def learning_rate(hours_seen: float, total_hours: float, warmup_hours: float, peak: float) -> float:
    if hours_seen < warmup_hours:
        return peak * hours_seen / warmup_hours
    progress = min(1.0, (hours_seen - warmup_hours) / max(1e-9, total_hours - warmup_hours))
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def save(path: Path, state: dict):
    pending = path.with_suffix('.pending')
    torch.save(state, pending)
    os.replace(pending, path)


@torch.no_grad()
def evaluate(model, loader, vocabulary, device, limit: int) -> dict:
    model.eval()
    rows, losses, clips = [], [], 0
    for front, frames, tokens, token_counts, teacher, texts in loader:
        features, frames = restore(front.to(device)), frames.to(device)
        flare = restore(teacher.to(device)) if model.flare.whole else None
        with torch.autocast('cuda', torch.bfloat16):
            loss = model(features, frames, tokens.to(device), token_counts.to(device), teacher=flare)[0]
            predictions = model.greedy(features, frames, flare=flare)
        losses.append(float(loss))
        rows += [score(text, vocabulary.decode(ids)) for text, ids in zip(texts, predictions)]
        clips += len(texts)
        if clips >= limit:
            break
    model.train()
    return {'dev_loss': round(sum(losses) / max(1, len(losses)), 4), 'dev_wer': corpus_wer(rows), 'dev_clips': clips}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True,
                        help='Folder extract_features.py wrote: one sub-folder per clip shard, and frontend.pt.')
    parser.add_argument('--train', nargs='+', required=True,
                        help='Quoted glob patterns of the sub-folders to train on, e.g. "avyt-talking-*".')
    parser.add_argument('--dev', nargs='+', required=True, help='Names of the sub-folders held out for dev WER.')
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, required=True, help='Hours of video in one pass of the training features.')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--frames-per-batch', type=int, default=14400)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.03)
    parser.add_argument('--warmup', type=float, default=0.04, help='Share of all video hours spent warming up.')
    parser.add_argument('--teacher-weight', type=float, default=5.0,
                        help='Weight of the loss for reproducing FLARE\'s encoder features, against the text losses.')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--keep-layers', type=int, nargs='+', default=list(range(FLARE_DEPTH)),
                        help='Which of FLARE\'s 18 encoder layers the model keeps (17M parameters each). With all of '
                             'them, the default, the encoder stays as it is and only the text side is trained.')
    parser.add_argument('--init', type=Path,
                        help='Checkpoint whose text side starts a new run: the first stage\'s, when layers are dropped.')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=12, help='Layers of the small Conformer on top.')
    parser.add_argument('--dev-clips', type=int, default=1500)
    parser.add_argument('--max-frames', type=int, default=400, help='Longest clip used, for training and dev.')
    parser.add_argument('--stop-after-steps', type=int, help='For smoke tests.')
    args = parser.parse_args()

    device = torch.device('cuda')
    args.output.mkdir(parents=True, exist_ok=True)
    vocabulary = Vocabulary(args.vocabulary)
    model = LipReader(vocabulary.size, keep=args.keep_layers, dim=args.dim, layers=args.layers).to(device)
    # The front end is FLARE's and stays as it is; training never runs it, but the checkpoint must hold it.
    model.frontend.load_state_dict(torch.load(args.features / 'frontend.pt', map_location=device))
    model.flare.load_flare(torch.load(args.features / 'encoder.pt', map_location=device))
    if args.init:
        text_side = {name: value for name, value in torch.load(args.init, map_location=device, weights_only=False)['model'].items()
                     if not name.startswith(('frontend.', 'flare.'))}
        missing = model.load_state_dict(text_side, strict=False).missing_keys
        assert all(name.startswith(('frontend.', 'flare.')) for name in missing), missing
    trained = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': [p for p in trained if p.ndim > 1], 'weight_decay': args.weight_decay},
                                   {'params': [p for p in trained if p.ndim <= 1], 'weight_decay': 0.0}],
                                  lr=args.lr, betas=(0.9, 0.98))
    state = {'step': 0, 'epoch': 0, 'hours_seen': 0.0, 'best_wer': None, 'restarts': 0}
    last = args.output / 'last.pt'
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        state.update(checkpoint['state'])
        state['restarts'] += 1
    total_hours = args.hours * args.epochs
    warmup_hours = total_hours * args.warmup
    dev_folders = [args.features / name for name in args.dev]

    def training_folders():
        # Listed again every pass, so shards whose features finish later join in.
        found = sorted(Path(index).parent for pattern in args.train
                       for index in glob.glob(str(args.features / pattern / 'index.json')))
        return [folder for folder in found if folder not in dev_folders]

    if not training_folders():
        raise SystemExit('No training features found.')
    log_path = args.output / 'train.jsonl'

    def log(row: dict):
        with open(log_path, 'a') as output:
            output.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def checkpoint_state():
        return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'state': state,
                'config': {'keep': model.flare.keep, 'dim': args.dim, 'layers': args.layers,
                           'vocab_size': vocabulary.size}}

    def check():
        """Dev WER and a checkpoint. A pass over thousands of hours is too long to wait for either."""
        dev = FeatureBatches(dev_folders, args.vocabulary, frames_per_batch=args.frames_per_batch,
                             max_frames=args.max_frames, shuffle=False, with_text=True)
        result = evaluate(model, DataLoader(dev, batch_size=None, num_workers=2), vocabulary, device, args.dev_clips)
        log({'event': 'epoch', 'epoch': state['epoch'], 'step': state['step'],
             'hours_seen': round(state['hours_seen'], 1), **result})
        if result['dev_wer'] is not None and (state['best_wer'] is None or result['dev_wer'] < state['best_wer']):
            state['best_wer'] = result['dev_wer']
            save(args.output / 'best.pt', checkpoint_state())
        save(last, checkpoint_state())

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log({'event': 'start', 'parameters_m': round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
         'trained_m': round(sum(p.numel() for p in trained) / 1e6, 1), 'train_shards': len(training_folders()),
         'dev_shards': len(dev_folders), 'total_hours': total_hours, 'time': round(time.time()), **state})
    model.train()
    while state['epoch'] < args.epochs and not stopping:
        epoch = state['epoch']
        folders = training_folders()
        log({'event': 'pass', 'epoch': epoch, 'train_shards': len(folders), 'max_frames': args.max_frames})
        dataset = FeatureBatches(folders, args.vocabulary, frames_per_batch=args.frames_per_batch,
                                 max_frames=args.max_frames, seed=epoch * 100 + state['restarts'])
        loader = DataLoader(dataset, batch_size=None, num_workers=args.workers, prefetch_factor=4, pin_memory=True)
        window_start, window_frames, window = time.perf_counter(), 0, []
        for front, frames, tokens, token_counts, teacher in loader:
            lr = learning_rate(state['hours_seen'], total_hours, warmup_hours, args.lr)
            for group in optimizer.param_groups:
                group['lr'] = lr
            features = time_mask(restore(front.to(device, non_blocking=True)), frames)
            with torch.autocast('cuda', torch.bfloat16):
                loss, ctc, attention, copied = model(
                    features, frames.to(device), tokens.to(device), token_counts.to(device),
                    teacher=restore(teacher.to(device, non_blocking=True)), teacher_weight=args.teacher_weight)
            optimizer.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trained, 5.0)
                optimizer.step()
            state['step'] += 1
            clip_frames = int(frames.sum())
            state['hours_seen'] += clip_frames / 25 / 3600
            window_frames += clip_frames
            window.append((float(loss.detach()), float(ctc), float(attention), float(copied)))
            if state['step'] % 100 == 0:
                elapsed = time.perf_counter() - window_start
                columns = [sum(column) / len(window) for column in zip(*window)]
                log({'step': state['step'], 'epoch': epoch, 'hours_seen': round(state['hours_seen'], 1),
                     'loss': round(columns[0], 4), 'ctc': round(columns[1], 4), 'attention': round(columns[2], 4),
                     'teacher': round(columns[3], 4), 'lr': round(lr, 7),
                     'gpu_gib': round(torch.cuda.max_memory_reserved() / 2 ** 30, 1),
                     'video_hours_per_hour': round(window_frames / 25 / elapsed)})
                window_start, window_frames, window = time.perf_counter(), 0, []
            if state['step'] % 5000 == 0:
                check()
            if stopping or (args.stop_after_steps and state['step'] >= args.stop_after_steps):
                stopping = True
                break
        del loader
        if stopping:
            break
        state['epoch'] += 1
    check()
    log({'event': 'finished' if state['epoch'] >= args.epochs else 'stopped', **state})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

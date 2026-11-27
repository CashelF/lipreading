#!/usr/bin/env python3
"""Train the pilot lipreader on shards of mouth clips. See docs/TRAINING.md.

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
from training.data import ClipBatches, prepare
from training.model import LipReader
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
    for video, frames, tokens, token_counts, texts in loader:
        video, frames = video.to(device), frames.to(device)
        x = prepare(video, frames, train=False)
        with torch.autocast('cuda', torch.bfloat16):
            loss, _, _ = model(x, frames, tokens.to(device), token_counts.to(device))
            predictions = model.greedy(x, frames)
        losses.append(float(loss))
        rows += [score(text, vocabulary.decode(ids)) for text, ids in zip(texts, predictions)]
        clips += len(texts)
        if clips >= limit:
            break
    model.train()
    return {'dev_loss': round(sum(losses) / max(1, len(losses)), 4), 'dev_wer': corpus_wer(rows), 'dev_clips': clips}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', nargs='+', required=True,
                        help='Quoted glob patterns of training shards, e.g. "data/train/lips/*.tar".')
    parser.add_argument('--dev', type=Path, nargs='+', required=True, help='Held-out shard files.')
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, required=True, help='Hours of video in one pass of the training shards.')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--frames-per-batch', type=int, default=14400)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.03)
    parser.add_argument('--warmup', type=float, default=0.04, help='Share of all video hours spent warming up.')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--dev-clips', type=int, default=1500)
    parser.add_argument('--memory-gib', type=float, help='Cap on GPU memory, so other jobs on the card keep theirs.')
    parser.add_argument('--stop-after-steps', type=int, help='For smoke tests.')
    args = parser.parse_args()

    device = torch.device('cuda')
    # Batch shapes change every step. Autotuning each new shape costs time and
    # makes cuDNN hold large workspaces, so leave it off.
    torch.backends.cudnn.benchmark = False
    if args.memory_gib:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gib * 2 ** 30 / total), 0)
    args.output.mkdir(parents=True, exist_ok=True)
    vocabulary = Vocabulary(args.vocabulary)
    model = LipReader(vocabulary.size, width=args.width, dim=args.dim, layers=args.layers).to(device)
    decay = [p for p in model.parameters() if p.ndim > 1]
    plain = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                                   {'params': plain, 'weight_decay': 0.0}], lr=args.lr, betas=(0.9, 0.98))
    state = {'step': 0, 'epoch': 0, 'hours_seen': 0.0, 'best_wer': None, 'restarts': 0, 'skipped': 0}
    last = args.output / 'last.pt'
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        state.update(checkpoint['state'])
        state['restarts'] += 1
    total_hours = args.hours * args.epochs
    warmup_hours = total_hours * args.warmup
    dev_shards = [str(path.resolve()) for path in args.dev]

    def training_shards():
        # Listed again every pass, so shards that finish converting later join in.
        found = [Path(shard).resolve() for pattern in args.train for shard in sorted(glob.glob(pattern))]
        return [shard for shard in found if str(shard) not in dev_shards]

    train_shards = training_shards()
    if not train_shards:
        raise SystemExit('No training shards found.')
    log_path = args.output / 'train.jsonl'

    def log(row: dict):
        with open(log_path, 'a') as output:
            output.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def checkpoint_state():
        return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'state': state,
                'config': {'width': args.width, 'dim': args.dim, 'layers': args.layers, 'vocab_size': vocabulary.size}}

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log({'event': 'start', 'parameters_m': round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
         'train_shards': len(train_shards), 'dev_shards': len(dev_shards), **state})
    model.train()
    while state['epoch'] < args.epochs and not stopping:
        epoch = state['epoch']
        # Curriculum: lipreading from scratch needs short clips first to learn alignment.
        max_frames = 100 if epoch == 0 else 400
        train_shards = training_shards()
        log({'event': 'pass', 'epoch': epoch, 'train_shards': len(train_shards), 'max_frames': max_frames})
        dataset = ClipBatches(train_shards, args.vocabulary, frames_per_batch=args.frames_per_batch,
                              max_frames=max_frames, seed=epoch * 100 + state['restarts'])
        loader = DataLoader(dataset, batch_size=None, num_workers=args.workers, prefetch_factor=4, pin_memory=True)
        window_start, window_frames, window = time.perf_counter(), 0, []
        for video, frames, tokens, token_counts in loader:
            lr = learning_rate(state['hours_seen'], total_hours, warmup_hours, args.lr)
            for group in optimizer.param_groups:
                group['lr'] = lr
            try:
                video = video.to(device, non_blocking=True)
                x = prepare(video, frames, train=True)
                with torch.autocast('cuda', torch.bfloat16):
                    loss, ctc, attention = model(x, frames.to(device), tokens.to(device), token_counts.to(device))
                optimizer.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            except torch.OutOfMemoryError:
                # A rare oversized batch under the memory cap: drop it and carry on.
                del video
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                state['skipped'] += 1
                continue
            state['step'] += 1
            clip_frames = int(frames.sum())
            state['hours_seen'] += clip_frames / 25 / 3600
            window_frames += clip_frames
            window.append((float(loss.detach()), float(ctc), float(attention)))
            if state['step'] % 100 == 0:
                elapsed = time.perf_counter() - window_start
                columns = list(zip(*window))
                log({'step': state['step'], 'epoch': epoch, 'hours_seen': round(state['hours_seen'], 1),
                     'loss': round(sum(columns[0]) / len(window), 4), 'ctc': round(sum(columns[1]) / len(window), 2),
                     'attention': round(sum(columns[2]) / len(window), 4), 'lr': round(lr, 7),
                     'skipped': state['skipped'], 'gpu_gib': round(torch.cuda.max_memory_reserved() / 2 ** 30, 1),
                     'video_hours_per_hour': round(window_frames / 25 / elapsed)})
                window_start, window_frames, window = time.perf_counter(), 0, []
            if state['step'] % 2000 == 0:
                save(last, checkpoint_state())
            if stopping or (args.stop_after_steps and state['step'] >= args.stop_after_steps):
                stopping = True
                break
        del loader
        if stopping:
            break
        state['epoch'] += 1
        dev = ClipBatches(dev_shards, args.vocabulary, frames_per_batch=args.frames_per_batch, shuffle=False,
                          with_text=True)
        result = evaluate(model, DataLoader(dev, batch_size=None, num_workers=2), vocabulary, device, args.dev_clips)
        log({'event': 'epoch', 'epoch': epoch, 'step': state['step'], 'hours_seen': round(state['hours_seen'], 1),
             **result})
        if result['dev_wer'] is not None and (state['best_wer'] is None or result['dev_wer'] < state['best_wer']):
            state['best_wer'] = result['dev_wer']
            save(args.output / 'best.pt', checkpoint_state())
        save(last, checkpoint_state())
    save(last, checkpoint_state())
    log({'event': 'finished' if state['epoch'] >= args.epochs else 'stopped', **state})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

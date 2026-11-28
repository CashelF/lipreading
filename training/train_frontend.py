#!/usr/bin/env python3
"""Teach the front end to work from smaller frames. See docs/TRAINING.md.

The front end is most of the model's compute, and its cost falls with the square of the frame
size. It starts from FLARE's weights and learns, from video at a lower resolution, to reproduce
what extract_features.py saved at full resolution: its own features, and what FLARE's encoder
made of them. The second target is the one that counts, because everything after the front end
was trained on the encoder's output. Logs hold numbers only.
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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.data import ClipBatches, prepare, restore
from training.model import FLARE_DEPTH, FlareStack, Frontend, shrink
from training.shards import read_samples
from training.video import decode_gray, frame_count


class ClipTargets(ClipBatches):
    """Batches of clips with what FLARE gave for them: uint8 video (B, T, 96, 96), frame counts, two
    unused token tensors, then the stored int8 features (B, T, 1536): the front end's, then the encoder's."""

    def __init__(self, shards, features: Path, vocabulary_path, **options):
        super().__init__(shards, vocabulary_path, **options)
        self.features = Path(features)

    def _read(self, shard, vocabulary, generator):
        folder = self.features / Path(shard).stem
        clips = json.loads((folder / 'index.json').read_text())['clips']
        front, deep = (np.load(folder / name, mmap_mode='r') for name in ('front.npy', 'teacher.npy'))
        starts = dict(zip((key for key, _, _ in clips), np.concatenate([[0], np.cumsum([frames for _, _, frames in clips])])))
        for sample in read_samples(shard):
            if sample['key'] not in starts:
                continue
            try:
                if frame_count(sample['video']) > self.max_frames:
                    continue
                video = decode_gray(sample['video'])
            except Exception:
                continue
            rows = slice(int(starts[sample['key']]), int(starts[sample['key']]) + len(video))
            if self.min_frames <= len(video) <= self.max_frames:
                yield video, [1], '', np.concatenate([front[rows], deep[rows]], 1)


def distance(features, wanted, frames):
    """L1 plus one minus cosine between standardized features, over real frames. Also returns the mean cosine."""
    real = torch.arange(features.shape[1], device=features.device)[None] < frames.to(features.device)[:, None]
    predicted, wanted = F.layer_norm(features.float(), features.shape[-1:])[real], wanted[real]
    cosine = F.cosine_similarity(predicted, wanted, dim=-1).mean()
    return F.l1_loss(predicted, wanted) + 1 - cosine, cosine


def both_distances(frontend, flare, video, wanted, frames):
    """Distance of the front end's features, and of FLARE's encoder output computed from them, to the stored ones."""
    features = F.layer_norm(frontend(video).float(), (Frontend.dim,))
    padding = torch.arange(features.shape[1], device=features.device)[None] >= frames.to(features.device)[:, None]
    near, cosine = distance(features, wanted[..., :Frontend.dim], frames)
    far, deep_cosine = distance(flare(features, padding), wanted[..., Frontend.dim:], frames)
    return near + far, cosine, deep_cosine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips', nargs='+', required=True, help='Quoted glob patterns of clip shards.')
    parser.add_argument('--features', type=Path, required=True, help='Folder extract_features.py wrote.')
    parser.add_argument('--dev', nargs='+', required=True, help='Clip shards held out for the dev check.')
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, required=True, help='Frame size the front end will run at (FLARE: 88).')
    parser.add_argument('--hours', type=float, required=True, help='Hours of video to train on.')
    parser.add_argument('--frames-per-batch', type=int, default=4800)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()

    device = torch.device('cuda')
    args.output.mkdir(parents=True, exist_ok=True)
    frontend = Frontend().to(device)
    frontend.load_state_dict(torch.load(args.features / 'frontend.pt', map_location=device))
    # FLARE's encoder, frozen: gradients pass through it to the front end, none of its weights change.
    flare = FlareStack(FLARE_DEPTH).to(device)
    flare.load_flare(torch.load(args.features / 'encoder.pt', map_location=device))
    optimizer = torch.optim.AdamW(frontend.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    dev_shards = [str(Path(shard).resolve()) for shard in args.dev]
    shards = [str(Path(shard).resolve()) for pattern in args.clips for shard in sorted(glob.glob(pattern))]
    shards = [shard for shard in shards if shard not in dev_shards and (args.features / Path(shard).stem / 'index.json').exists()]
    log_path = args.output / 'train.jsonl'

    def log(row: dict):
        with open(log_path, 'a') as output:
            output.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def save():
        pending = args.output / 'frontend.pending'
        torch.save({'frontend': frontend.state_dict(), 'resolution': args.resolution}, pending)
        os.replace(pending, args.output / 'frontend.pt')

    @torch.no_grad()
    def check(hours_seen):
        frontend.eval()
        loader = ClipTargets(dev_shards, args.features, args.vocabulary, frames_per_batch=args.frames_per_batch, shuffle=False)
        near, far, frames_seen = 0.0, 0.0, 0
        for video, frames, _, _, wanted in DataLoader(loader, batch_size=None, num_workers=2):
            with torch.autocast('cuda', torch.bfloat16):
                _, cosine, deep_cosine = both_distances(frontend, flare, shrink(prepare(video.to(device), frames), args.resolution),
                                                        restore(wanted.to(device)), frames)
            near, far = near + float(cosine) * int(frames.sum()), far + float(deep_cosine) * int(frames.sum())
            frames_seen += int(frames.sum())
            if frames_seen > 400000:
                break
        frontend.train()
        log({'event': 'dev', 'hours_seen': round(hours_seen, 1), 'dev_cosine': round(near / frames_seen, 4),
             'dev_encoder_cosine': round(far / frames_seen, 4)})
        save()

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log({'event': 'start', 'resolution': args.resolution, 'train_shards': len(shards), 'total_hours': args.hours,
         'parameters_m': round(sum(p.numel() for p in frontend.parameters()) / 1e6, 1), 'time': round(time.time())})
    step, hours_seen, epoch = 0, 0.0, 0
    frontend.train()
    while hours_seen < args.hours and not stopping:
        dataset = ClipTargets(shards, args.features, args.vocabulary, frames_per_batch=args.frames_per_batch, seed=epoch)
        loader = DataLoader(dataset, batch_size=None, num_workers=args.workers, prefetch_factor=4, pin_memory=True)
        window_start, window_frames, window = time.perf_counter(), 0, []
        for video, frames, _, _, wanted in loader:
            warm = min(1.0, hours_seen / (0.05 * args.hours))
            lr = args.lr * warm * (0.5 * (1 + math.cos(math.pi * min(1.0, hours_seen / args.hours))) if warm == 1.0 else 1.0)
            for group in optimizer.param_groups:
                group['lr'] = lr
            x = shrink(prepare(video.to(device, non_blocking=True), frames), args.resolution)
            with torch.autocast('cuda', torch.bfloat16):
                loss, cosine, deep_cosine = both_distances(frontend, flare, x, restore(wanted.to(device, non_blocking=True)), frames)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(frontend.parameters(), 5.0)
            optimizer.step()
            step += 1
            hours_seen += int(frames.sum()) / 90000
            window_frames += int(frames.sum())
            window.append((float(loss.detach()), float(cosine.detach()), float(deep_cosine.detach())))
            if step % 100 == 0:
                elapsed = time.perf_counter() - window_start
                log({'step': step, 'hours_seen': round(hours_seen, 1), 'loss': round(sum(w[0] for w in window) / len(window), 4),
                     'cosine': round(sum(w[1] for w in window) / len(window), 4),
                     'encoder_cosine': round(sum(w[2] for w in window) / len(window), 4), 'lr': round(lr, 7),
                     'gpu_gib': round(torch.cuda.max_memory_reserved() / 2 ** 30, 1),
                     'video_hours_per_hour': round(window_frames / 25 / elapsed)})
                window_start, window_frames, window = time.perf_counter(), 0, []
            if step % 2000 == 0:
                check(hours_seen)
            if stopping or hours_seen >= args.hours:
                break
        del loader
        epoch += 1
    check(hours_seen)
    log({'event': 'stopped' if stopping else 'finished', 'step': step, 'hours_seen': round(hours_seen, 1)})


if __name__ == '__main__':
    main()

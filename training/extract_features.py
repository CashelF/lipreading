#!/usr/bin/env python3
"""Run FLARE once over every clip and save what training needs. See docs/TRAINING.md.

For each shard of mouth clips this writes a folder with, per video frame:
  front.npy    FLARE's lip front-end output (the small model's input),
  teacher.npy  FLARE's encoder output (the small model's target),
both standardized per frame and stored as int8 (value x 24), plus index.json with each clip's
key, transcript and frame count. Training then never touches video or FLARE, and a pass over
the data costs a small fraction of the first one. Finished shards are skipped on a rerun.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from training.data import SCALE, prepare, quantize
from training.flare_teacher import FlareEncoder
from training.shards import read_samples
from training.video import decode_gray, frame_count

class ShardClips(IterableDataset):
    """Each worker decodes its shards and yields (shard number, batch of equal-length clips); None marks a shard's end.

    Equal length matters: FLARE's encoder lets padded frames leak into real ones, so a clip padded
    beside a longer one gets different features than it gets alone, which is how the app runs it.
    """

    def __init__(self, shards, frames_per_batch: int, max_frames: int, min_frames: int = 12, pool: int = 1000):
        self.shards, self.frames_per_batch, self.max_frames, self.min_frames = shards, frames_per_batch, max_frames, min_frames
        self.pool = pool

    def _batches(self, clips):
        clips.sort(key=lambda clip: len(clip[2]))
        batch = []
        for clip in clips:
            if batch and (len(clip[2]) != len(batch[0][2]) or len(clip[2]) * (len(batch) + 1) > self.frames_per_batch):
                yield self._collate(batch)
                batch = []
            batch.append(clip)
        if batch:
            yield self._collate(batch)

    def __iter__(self):
        info = get_worker_info()
        for number in range(0 if info is None else info.id, len(self.shards), 1 if info is None else info.num_workers):
            clips = []
            for sample in read_samples(self.shards[number]):
                try:
                    if frame_count(sample['video']) > self.max_frames:
                        continue
                    video = decode_gray(sample['video'])
                except Exception:
                    continue
                if self.min_frames <= len(video) <= self.max_frames and video.shape[1:] == (96, 96):
                    clips.append((sample['key'], sample['text'], video))
                # A shard can hold 20 hours of video, 16 GB decoded: hand it over in pieces. Every worker
                # holds one piece (about 1.6 GB), so pieces times workers must fit in memory beside other jobs.
                if len(clips) >= self.pool:
                    yield from ((number, batch) for batch in self._batches(clips))
                    clips = []
            yield from ((number, batch) for batch in self._batches(clips))
            yield number, None

    @staticmethod
    def _collate(batch):
        video = torch.from_numpy(np.stack([clip for _, _, clip in batch]))
        return [key for key, _, _ in batch], [text for _, text, _ in batch], torch.full((len(batch),), video.shape[1]), video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards', nargs='+', required=True, help='Quoted glob patterns of clip shards.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames-per-batch', type=int, default=9600)
    parser.add_argument('--max-frames', type=int, default=400)
    parser.add_argument('--workers', type=int, default=8,
                        help='Each holds about 2 GB, and the main process about 3 GB for every shard in progress.')
    args = parser.parse_args()
    shards = [Path(shard) for pattern in args.shards for shard in sorted(glob.glob(pattern))]
    shards = [shard for shard in shards if not (args.output / shard.stem / 'index.json').exists()]
    if not shards:
        return  # Everything is extracted; don't load FLARE for nothing.
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    flare = FlareEncoder(device)
    captured = {}
    # The lip front end runs inside FLARE's forward pass; keep its output instead of computing it twice.
    flare.encoder.feature_extractor_video_lip.resnet.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__('front', output.transpose(1, 2)))
    # Weights a trained model keeps or starts from, so training never needs FLARE itself.
    lip = flare.encoder.feature_extractor_video_lip
    torch.save({name: value.cpu() for name, value in lip.resnet.state_dict().items()}, args.output / 'frontend.pt')
    parts = {'lip_proj': lip.proj, 'mix_norm': flare.encoder.layer_norm, 'mix_proj': flare.encoder.post_extract_proj,
             'layers': flare.encoder.encoder.layers}
    torch.save({f'{part}.{name}': value.float().cpu() for part, module in parts.items()
                for name, value in module.state_dict().items()}, args.output / 'encoder.pt')
    loader = DataLoader(ShardClips([str(shard) for shard in shards], args.frames_per_batch, args.max_frames),
                        batch_size=None, num_workers=args.workers, prefetch_factor=2, in_order=False)
    pending = {}
    for number, batch in loader:
        store = pending.setdefault(number, {'clips': [], 'front': [], 'teacher': []})
        if batch is not None:
            keys, texts, frames, video = batch
            with torch.autocast('cuda', torch.bfloat16):
                teacher = flare(prepare(video.to(device), frames), frames)
            front, teacher = quantize(captured.pop('front')), quantize(teacher)  # One copy off the GPU per batch.
            for row, count in enumerate(frames.tolist()):
                store['clips'].append([keys[row], texts[row], count])
                store['front'].append(front[row, :count])
                store['teacher'].append(teacher[row, :count])
            continue
        del pending[number]
        folder = args.output / shards[number].stem
        scratch = args.output / ('.' + shards[number].stem + '.pending')
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir()
        np.save(scratch / 'front.npy', np.concatenate(store['front']) if store['front'] else np.zeros((0, 768), np.int8))
        np.save(scratch / 'teacher.npy', np.concatenate(store['teacher']) if store['teacher'] else np.zeros((0, 768), np.int8))
        (scratch / 'index.json').write_text(json.dumps({'scale': SCALE, 'clips': store['clips']}))
        scratch.rename(folder)  # Published whole, so a reader never sees a half-written shard.
        print(json.dumps({'shard': shards[number].stem, 'clips': len(store['clips']),
                          'hours': round(sum(clip[2] for clip in store['clips']) / 90000, 2)}), flush=True)


if __name__ == '__main__':
    main()

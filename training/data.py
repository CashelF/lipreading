"""Shard streaming, length-bucketed batches and GPU-side augmentation."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from training.shards import read_samples
from training.text import Vocabulary
from training.video import decode_gray, frame_count

MEAN, STD = 0.421, 0.165  # Same grayscale statistics as the live preprocessor.


class ClipBatches(IterableDataset):
    """Yields ready batches: uint8 video (B, T, 96, 96), frame counts, padded token ids, token counts."""

    def __init__(self, shards, vocabulary_path, *, frames_per_batch: int, max_frames: int = 400, min_frames: int = 12,
                 shuffle: bool = True, seed: int = 0, pool: int = 600, interleave: int = 4,
                 with_text: bool = False):
        self.shards = [str(path) for path in shards]
        self.vocabulary_path = str(vocabulary_path)
        self.frames_per_batch, self.max_frames, self.min_frames = frames_per_batch, max_frames, min_frames
        self.shuffle, self.seed, self.pool, self.with_text = shuffle, seed, pool, with_text
        self.interleave = interleave if shuffle else 1

    def _samples(self, shards, vocabulary):
        # Read a few shards at once so a batch is not a handful of videos from one shard.
        waiting, active = list(shards), []
        while waiting or active:
            while waiting and len(active) < self.interleave:
                active.append(read_samples(waiting.pop(0)))
            for reader in list(active):
                sample = next(reader, None)
                if sample is None:
                    active.remove(reader)
                    continue
                tokens = vocabulary.encode(sample['text'])
                if not tokens:
                    continue
                try:
                    # Skip over-long clips from the header alone; decoding is the costly step.
                    if frame_count(sample['video']) > self.max_frames:
                        continue
                    video = decode_gray(sample['video'])
                except Exception:
                    continue
                # CTC needs at least one frame per token (plus repeats); skip mislabelled clips.
                if not self.min_frames <= len(video) <= self.max_frames or len(tokens) * 2 > len(video):
                    continue
                if video.shape[1:] != (96, 96):
                    continue
                yield video, tokens, sample['text']

    def _batches(self, items):
        items.sort(key=lambda item: len(item[0]))
        batch, longest = [], 0
        for item in items:
            longest_next = max(longest, len(item[0]))
            if batch and longest_next * (len(batch) + 1) > self.frames_per_batch:
                yield batch
                batch, longest_next = [], len(item[0])
            batch.append(item)
            longest = longest_next
        if batch:
            yield batch

    def _collate(self, batch):
        frames = torch.tensor([len(video) for video, _, _ in batch])
        video = torch.zeros(len(batch), int(frames.max()), 96, 96, dtype=torch.uint8)
        token_counts = torch.tensor([len(tokens) for _, tokens, _ in batch])
        tokens = torch.zeros(len(batch), int(token_counts.max()), dtype=torch.long)
        for row, (clip, ids, _) in enumerate(batch):
            video[row, :len(clip)] = torch.from_numpy(clip)
            tokens[row, :len(ids)] = torch.tensor(ids)
        result = (video, frames, tokens, token_counts)
        return result + ([text for _, _, text in batch],) if self.with_text else result

    def __iter__(self):
        info = get_worker_info()
        shards = self.shards if info is None else self.shards[info.id::info.num_workers]
        generator = random.Random(self.seed * 1000 + (0 if info is None else info.id))
        if self.shuffle:
            shards = generator.sample(shards, len(shards))
        vocabulary = Vocabulary(self.vocabulary_path)
        pool = []
        for item in self._samples(shards, vocabulary):
            pool.append(item)
            if len(pool) >= self.pool:
                batches = list(self._batches(pool))
                if self.shuffle:
                    generator.shuffle(batches)
                yield from map(self._collate, batches)
                pool = []
        if pool:
            yield from map(self._collate, self._batches(pool))


def prepare(video: torch.Tensor, frames: torch.Tensor, *, train: bool) -> torch.Tensor:
    """uint8 (B, T, 96, 96) on the GPU to normalized float (B, T, 88, 88), augmented when training."""
    batch, length = video.shape[:2]
    if train:
        top, left = (int(value) for value in torch.randint(0, 9, (2,)))
    else:
        top = left = 4
    x = (video[:, :, top:top + 88, left:left + 88].float() / 255 - MEAN) / STD
    if train:
        flip = torch.rand(batch, device=video.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        # Time masking: per second of video, hide up to 0.4 s (set to the mean).
        hidden = torch.zeros(batch, length, dtype=torch.bool)
        for row, count in enumerate(frames.tolist()):
            for _ in range(max(1, count // 25)):
                span = random.randint(0, 10)
                start = random.randint(0, max(0, count - span))
                hidden[row, start:start + span] = True
        x = x.masked_fill(hidden.to(video.device)[:, :, None, None], 0)
    padding = torch.arange(length, device=video.device)[None] >= frames.to(video.device)[:, None]
    return x.masked_fill(padding[:, :, None, None], 0)


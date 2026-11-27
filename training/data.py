"""Length-bucketed batches of stored features (training) or of clips (extraction, evaluation)."""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from training.shards import read_samples
from training.text import Vocabulary
from training.video import decode_gray, frame_count

MEAN, STD = 0.421, 0.165  # Same grayscale statistics as the live preprocessor.
SCALE = 24  # Stored features are int8: the standardized value times this (+-5.3 deviations, 0.04 steps).


class ClipBatches(IterableDataset):
    """Yields ready batches: uint8 video (B, T, 96, 96), frame counts, padded token ids, token counts,
    then the transcripts with ``with_text``."""

    def __init__(self, shards, vocabulary_path, *, frames_per_batch: int, max_frames: int = 400, min_frames: int = 12,
                 shuffle: bool = True, seed: int = 0, pool: int = 600, interleave: int = 4, with_text: bool = False):
        self.shards = [str(path) for path in shards]
        self.vocabulary_path = str(vocabulary_path)
        self.frames_per_batch, self.max_frames, self.min_frames = frames_per_batch, max_frames, min_frames
        self.shuffle, self.seed, self.pool, self.with_text = shuffle, seed, pool, with_text
        self.interleave = interleave if shuffle else 1

    def _usable(self, frames: int, tokens: list[int]) -> bool:
        # CTC needs at least one frame per token (plus repeats); this also drops mislabelled clips.
        return bool(tokens) and self.min_frames <= frames <= self.max_frames and len(tokens) * 2 <= frames

    def _read(self, shard, vocabulary, generator):
        """One shard's usable clips as (frames array, token ids, transcript, extra)."""
        for sample in read_samples(shard):
            tokens = vocabulary.encode(sample['text'])
            try:
                # Skip over-long clips from the header alone; decoding is the costly step.
                if not tokens or frame_count(sample['video']) > self.max_frames:
                    continue
                video = decode_gray(sample['video'])
            except Exception:
                continue
            if self._usable(len(video), tokens) and video.shape[1:] == (96, 96):
                yield video, tokens, sample['text'], None

    def _samples(self, shards, vocabulary, generator):
        # Read a few shards at once so a batch is not a handful of videos from one shard.
        waiting, active = list(shards), []
        while waiting or active:
            while waiting and len(active) < self.interleave:
                active.append(self._read(waiting.pop(0), vocabulary, generator))
            for reader in list(active):
                sample = next(reader, None)
                if sample is None:
                    active.remove(reader)
                else:
                    yield sample

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
        frames = torch.tensor([len(item[0]) for item in batch])
        token_counts = torch.tensor([len(item[1]) for item in batch])
        tokens = torch.zeros(len(batch), int(token_counts.max()), dtype=torch.long)
        inputs = torch.zeros(len(batch), int(frames.max()), *batch[0][0].shape[1:], dtype=torch.from_numpy(batch[0][0]).dtype)
        extra = None if batch[0][3] is None else torch.zeros(len(batch), int(frames.max()), batch[0][3].shape[1], dtype=torch.int8)
        for row, (clip, ids, _, target) in enumerate(batch):
            inputs[row, :len(clip)] = torch.from_numpy(clip)
            tokens[row, :len(ids)] = torch.tensor(ids)
            if extra is not None:
                extra[row, :len(target)] = torch.from_numpy(target)
        result = (inputs, frames, tokens, token_counts) + (() if extra is None else (extra,))
        return result + ([item[2] for item in batch],) if self.with_text else result

    def __iter__(self):
        info = get_worker_info()
        shards = self.shards if info is None else self.shards[info.id::info.num_workers]
        generator = random.Random(self.seed * 1000 + (0 if info is None else info.id))
        if self.shuffle:
            shards = generator.sample(shards, len(shards))
        vocabulary = Vocabulary(self.vocabulary_path)
        pool = []
        for item in self._samples(shards, vocabulary, generator):
            pool.append(item)
            if len(pool) >= self.pool:
                batches = list(self._batches(pool))
                if self.shuffle:
                    generator.shuffle(batches)
                yield from map(self._collate, batches)
                pool = []
        if pool:
            yield from map(self._collate, self._batches(pool))


class FeatureBatches(ClipBatches):
    """Batches from the folders ``extract_features.py`` writes, one per clip shard: int8 front-end features
    (B, T, 768), frame counts, padded token ids, token counts, int8 teacher features (B, T, 768), then the
    transcripts with ``with_text``. No video is decoded, so a pass costs little."""

    def _read(self, folder, vocabulary, generator):
        folder = Path(folder)
        clips = json.loads((folder / 'index.json').read_text())['clips']
        front, teacher = (np.load(folder / name, mmap_mode='r') for name in ('front.npy', 'teacher.npy'))
        starts = np.concatenate([[0], np.cumsum([frames for _, _, frames in clips])])
        order = list(range(len(clips)))
        if self.shuffle:
            generator.shuffle(order)  # Clips are stored sorted by length.
        for index in order:
            _, text, frames = clips[index]
            tokens = vocabulary.encode(text)
            if self._usable(frames, tokens):
                rows = slice(int(starts[index]), int(starts[index]) + frames)
                yield np.array(front[rows]), tokens, text, np.array(teacher[rows])


def restore(stored: torch.Tensor) -> torch.Tensor:
    """int8 features from disk back to standardized floats."""
    return stored.float() / SCALE


def quantize(features: torch.Tensor) -> np.ndarray:
    """Features (..., D) to what is stored: standardized per frame, as int8."""
    standardized = torch.nn.functional.layer_norm(features.float(), features.shape[-1:])
    return (standardized * SCALE).round().clamp(-127, 127).to(torch.int8).cpu().numpy()


def prepare(video: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """uint8 video (B, T, 96, 96) on the GPU to the centre 88x88, normalized as FLARE expects, padding zeroed."""
    x = (video[:, :, 4:92, 4:92].float() / 255 - MEAN) / STD
    padding = torch.arange(video.shape[1], device=video.device)[None] >= frames.to(video.device)[:, None]
    return x.masked_fill(padding[:, :, None, None], 0)


def time_mask(features: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Hide up to 0.4 s per second of each clip's features (set to zero, their mean). Training only: the
    small model must fill the gap from context, while its target was computed from the whole clip."""
    hidden = torch.zeros(features.shape[:2], dtype=torch.bool)
    for row, count in enumerate(frames.tolist()):
        for _ in range(max(1, count // 25)):
            span = random.randint(0, 10)
            start = random.randint(0, max(0, count - span))
            hidden[row, start:start + span] = True
    return features.masked_fill(hidden.to(features.device)[:, :, None], 0)

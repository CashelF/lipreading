"""SentencePiece vocabulary shared by the CTC head and the attention decoder.

Id 0 is the CTC blank (SentencePiece's pad), 1 is unknown, and the id after the
last piece marks sentence start and end for the decoder.
"""
from __future__ import annotations

from pathlib import Path
import re

import sentencepiece as spm

_KEEP = re.compile(r"[^a-z' ]+")


def clean(text: str) -> str:
    """Lowercase letters, apostrophes and single spaces only."""
    return ' '.join(_KEEP.sub(' ', text.lower().replace('’', "'")).split())


class Vocabulary:
    def __init__(self, model: str | Path):
        self.pieces = spm.SentencePieceProcessor(model_file=str(model))
        self.blank = 0
        self.sos_eos = self.pieces.get_piece_size()
        self.size = self.sos_eos + 1

    def encode(self, text: str) -> list[int]:
        return self.pieces.encode(clean(text))

    def decode(self, ids) -> str:
        return self.pieces.decode([int(i) for i in ids if 0 < int(i) < self.sos_eos])


def train_vocabulary(lines, prefix: str | Path, size: int = 1000) -> Path:
    spm.SentencePieceTrainer.train(
        sentence_iterator=(clean(line) for line in lines), model_prefix=str(prefix), vocab_size=size,
        model_type='unigram', character_coverage=1.0, pad_id=0, pad_piece='<blank>', unk_id=1,
        bos_id=-1, eos_id=-1, input_sentence_size=2_000_000, shuffle_input_sentence=True)
    return Path(str(prefix) + '.model')


def main():
    import argparse
    import random
    from training.shards import read_samples

    parser = argparse.ArgumentParser(description='Train the SentencePiece vocabulary from shard transcripts.')
    parser.add_argument('--shards', type=Path, nargs='+', required=True, help='Shard files to read transcripts from.')
    parser.add_argument('--output', type=Path, required=True, help='Prefix; writes <prefix>.model and <prefix>.vocab.')
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--max-shards', type=int, default=60)
    args = parser.parse_args()
    shards = random.Random(0).sample(args.shards, min(args.max_shards, len(args.shards)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(train_vocabulary((sample['text'] for shard in shards for sample in read_samples(shard)),
                           args.output, args.size))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()

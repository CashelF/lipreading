#!/usr/bin/env python3
"""Shrink downloaded AVYT shards to compact grayscale clips, then delete the source.

Upstream ``avyt/talking-train`` clips are 96×96 colour at a high bitrate
(about 400 MB per hour). Training uses grayscale, so each shard is rewritten
as grayscale H.264 plus a small Opus soundtrack and the original tar is
removed. Shards still being downloaded are skipped; ``--watch`` keeps going
until the download unit stops.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.shards import ShardWriter, read_samples
from training.video import decode_gray, encode_gray, transcode_audio


def convert(source: str, destination: str, crf: int) -> dict:
    source, destination = Path(source), Path(destination)
    writer = ShardWriter(destination)
    clips = frames = failed = 0
    try:
        for sample in read_samples(source):
            try:
                images = decode_gray(sample['video'])
                writer.add(sample['key'], encode_gray(images, crf=crf), sample['text'],
                           transcode_audio(sample['video']))
                clips += 1
                frames += len(images)
            except Exception:  # One corrupt clip must not lose the shard.
                failed += 1
        if not clips:
            raise ValueError('No clips converted.')
        writer.close()
    except Exception:
        writer.abort()
        raise
    before, after = source.stat().st_size, destination.stat().st_size
    source.unlink()
    return {'shard': destination.name, 'clips': clips, 'failed': failed, 'hours': round(frames / 25 / 3600, 3),
            'source_mb': round(before / 1e6), 'compact_mb': round(after / 1e6)}


def downloading(unit: str) -> bool:
    return subprocess.run(['systemctl', '--user', 'is-active', '-q', unit]).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Folder of talking-train-*.tar shards.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--crf', type=int, default=17)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--watch', help='systemd user unit of the download; keep converting while it runs.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / 'prepare.jsonl'
    submitted = {}
    with ProcessPoolExecutor(args.workers) as pool:
        while True:
            still_downloading = bool(args.watch) and downloading(args.watch)
            for source in sorted(args.source.glob('talking-train-*.tar')):
                destination = args.output / source.name.replace('talking-train', 'avyt-talking')
                # A shard younger than a minute may still be moving into place.
                if source in submitted or destination.exists() or time.time() - source.stat().st_mtime < 60:
                    continue
                submitted[source] = pool.submit(convert, str(source), str(destination), args.crf)
            for source, future in list(submitted.items()):
                if future.done():
                    del submitted[source]
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {'shard': source.name, 'error': type(exc).__name__}
                    with open(log, 'a') as output:
                        output.write(json.dumps(row) + '\n')
                    print(json.dumps(row), flush=True)
            if not submitted and not still_downloading:
                break
            time.sleep(10)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Held-out word error rate on consented read-sentence recordings.

Recordings come from Help → Troubleshooting → Record sentences and stay in the
ignored data/eval-sentences/ folder. Two modes:

  live   Replay each clip in real time through a ready, idle server's
         /api/stream, exactly as the browser would (speech gate and visual
         checks on). Measures what a person sees after pressing Stop.
  model  Decode each whole clip once, in this process, with the speech gate
         and visual checks off. Measures FLARE alone; use it to compare models.

The report holds numeric error counts, prompt ids, speaker codes and
categorical failure reasons only: no recognized text and no pixels.
Never tune anything on these recordings; they are the held-out set.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.word_errors import bootstrap_interval, corpus_wer, score

ORIGIN = 'http://127.0.0.1:8000'


def load_recordings(root: Path, speaker: str | None = None):
    records = []
    for path in sorted(root.glob('*/manifest.json')):
        if not re.fullmatch(r'[0-9a-f]{32}', path.parent.name) or path.parent.is_symlink():
            continue
        manifest = json.loads(path.read_text())
        if manifest.get('label') != 'sentence' or (speaker and manifest['speaker'] != speaker):
            continue
        names, stamps = manifest['frame_files'], manifest['timestamps_ms']
        if len(names) != len(stamps) or not all(re.fullmatch(r'\d{6}\.jpg', name) for name in names):
            raise ValueError(f'Recording {manifest["id"]} has an inconsistent manifest.')
        records.append((path.parent, manifest))
    return records


def frames_of(directory: Path, manifest: dict):
    from backend.stream import Frame
    return [Frame(float(stamp), (directory / name).read_bytes())
            for name, stamp in zip(manifest['frame_files'], manifest['timestamps_ms'])]


def status(origin: str) -> dict:
    with urllib.request.urlopen(origin + '/api/status', timeout=5) as response:
        return json.load(response)


async def replay_live(origin: str, frames) -> dict:
    from websockets.asyncio.client import connect
    transcript, errors = None, []
    url = origin.replace('https:', 'wss:').replace('http:', 'ws:') + '/api/stream'
    async with connect(url, origin=origin, max_size=2 ** 24) as ws:
        await ws.send(json.dumps({'type': 'start', 'speech_gate': True, 'verify_visual': True}))
        first = json.loads(await ws.recv())
        if first.get('state') != 'capturing':
            return {'text': '', 'failure': 'session_refused', 'detail': first.get('detail', '')}
        stopped_at = None

        async def receive():
            nonlocal transcript
            async for raw in ws:
                message = json.loads(raw)
                if message['type'] == 'transcript':
                    transcript = message
                elif message['type'] == 'error':
                    errors.append(message.get('detail', ''))
                elif message['type'] == 'stopped':
                    return

        receiver = asyncio.create_task(receive())
        begin, origin_ms = time.perf_counter(), frames[0].timestamp
        for frame in frames:
            await asyncio.sleep(max(0., begin + (frame.timestamp - origin_ms) / 1000 - time.perf_counter()))
            await ws.send(struct.pack('<d', frame.timestamp) + frame.jpeg)
        stopped_at = time.perf_counter()
        await ws.send(json.dumps({'type': 'stop'}))
        await asyncio.wait_for(receiver, timeout=120)
    text = transcript['text'] if transcript else ''
    return {'text': text, 'failure': 'error' if errors else (None if text else 'no_text'),
            'detail': errors[-1] if errors else '',
            'stop_to_final_ms': round((time.perf_counter() - stopped_at) * 1000)}


def decode_model(runtime, frames) -> dict:
    from backend.preprocessing import PreprocessCache
    from backend.stream import Clip, resample, sample_times
    clip = Clip(0, tuple(frames), True)
    try:
        rgb, _ = resample(clip)
        cache = PreprocessCache(max_frames=len(rgb) + 1)
        cache.set_sample_times(sample_times(clip))
        begin = time.perf_counter()
        result = runtime.decode_utterance(rgb, 25, cache=cache, verify_visual=False)
    except ValueError as exc:
        # Quality rejections are fixed app messages (e.g. move closer), not captions.
        return {'text': '', 'failure': 'quality', 'detail': str(exc)}
    text = result['text']
    return {'text': text, 'failure': None if text else 'no_text', 'detail': '',
            'decode_ms': round((time.perf_counter() - begin) * 1000)}


def summarize(rows: list[dict]) -> dict:
    scored = [row for row in rows if row['reference_words']]
    by_speaker = defaultdict(list)
    for row in scored:
        by_speaker[row['speaker']].append(row)
    return {'recordings': len(rows), 'reference_words': sum(row['reference_words'] for row in scored),
            'wer': corpus_wer(scored), 'wer_95_interval': bootstrap_interval(scored),
            'substitutions': sum(row['substitutions'] for row in scored),
            'deletions': sum(row['deletions'] for row in scored),
            'insertions': sum(row['insertions'] for row in scored),
            'exact_sentences': sum(row['errors'] == 0 for row in scored),
            'failures': {reason: sum(row['failure'] == reason for row in rows)
                         for reason in sorted({row['failure'] for row in rows if row['failure']})},
            'speakers': {speaker: {'recordings': len(items), 'wer': corpus_wer(items)}
                         for speaker, items in sorted(by_speaker.items())}}


def write_private(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as output:
        json.dump(report, output, indent=2)
        output.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('live', 'model'), default='live')
    parser.add_argument('--recordings', type=Path, default=ROOT / 'data/eval-sentences')
    parser.add_argument('--speaker', help='Evaluate one speaker code only.')
    parser.add_argument('--origin', default=ORIGIN, help='Server for live mode.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    records = load_recordings(args.recordings, args.speaker)
    if not records:
        print('No sentence recordings yet. Open Help → Troubleshooting → Record sentences in the app.')
        return 2
    runtime = state = None
    if args.mode == 'live':
        state = status(args.origin)
        if not state.get('ready') or state.get('busy'):
            raise SystemExit('The server must be ready and idle; check /api/status.')
    else:
        from backend.model import FlareModel
        runtime = FlareModel()
        runtime.load()
    created = datetime.now(timezone.utc)
    output = args.output or ROOT / f'models/eval/sentences-{args.mode}-{created:%Y%m%dT%H%M%SZ}.json'
    rows = []
    for index, (directory, manifest) in enumerate(records, 1):
        frames = frames_of(directory, manifest)
        if args.mode == 'live':
            if status(args.origin).get('busy'):
                raise SystemExit('Someone started a session; stopping so the evaluation does not compete.')
            result = asyncio.run(replay_live(args.origin, frames))
        else:
            result = decode_model(runtime, frames)
        row = {'id': manifest['id'], 'prompt_id': manifest['prompt_id'], 'speaker': manifest['speaker'],
               'duration_seconds': manifest['duration_seconds'],
               **score(manifest['reference'], result.pop('text')), **result}
        rows.append(row)
        print(json.dumps({'done': f'{index}/{len(records)}', 'prompt_id': row['prompt_id'],
                          'wer': row['wer'], 'failure': row['failure']}), flush=True)
    model = ({'name': state.get('model'), 'beam_size': state.get('beam_size')} if runtime is None
             else {'name': runtime.name, 'beam_size': runtime.beam})
    report = {'created_at': created.isoformat(), 'mode': args.mode, 'model': model,
              'captions_saved': False, 'summary': summarize(rows), 'rows': rows}
    write_private(output, report)
    print(json.dumps(report['summary'], indent=2))
    print(f'Local report: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

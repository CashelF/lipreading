#!/usr/bin/env python3
"""Paced camera-protocol check with public photo fixtures, never user video.

Exercises real preprocessing and GPU decoding; does not measure speech accuracy.
"""
import argparse
import asyncio
from io import BytesIO
import json
from pathlib import Path
import struct
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from websockets.asyncio.client import connect
from scripts.benchmark_latency import fixture_frames


async def check(origin: str, seconds: float, speech_gate: bool, translated: bool = False):
    with urllib.request.urlopen(origin + '/api/status', timeout=5) as response:
        state = json.load(response)
    if not state['ready'] or state['busy']:
        raise RuntimeError('The model must be ready and idle before this check.')
    if not state.get('continuous_revisions'):
        raise RuntimeError('The server is not running the continuous-window version.')
    count = round(seconds * 25) + 1
    frames = fixture_frames(count)
    packets = []
    for index, frame in enumerate(frames):
        buffer = BytesIO()
        Image.fromarray(frames[0] if speech_gate and not translated else frame).save(buffer, format='JPEG', quality=92)
        packets.append(struct.pack('<d', index * 40.) + buffer.getvalue())
    messages = []
    async with connect(origin.replace('https:', 'wss:').replace('http:', 'ws:') + '/api/stream', origin=origin) as ws:
        await ws.send(json.dumps({'type': 'start', 'speech_gate': speech_gate}))
        first = json.loads(await ws.recv())
        assert first.get('state') == 'capturing', first

        async def receive():
            async for raw in ws:
                message = json.loads(raw)
                assert message['type'] != 'error', message
                messages.append(message)
                if message['type'] == 'stopped':
                    return

        receiver = asyncio.create_task(receive())
        start = time.perf_counter()
        for index, packet in enumerate(packets):
            await asyncio.sleep(max(0., start + index * .04 - time.perf_counter()))
            await ws.send(packet)
        await ws.send(json.dumps({'type': 'stop'}))
        await asyncio.wait_for(receiver, timeout=60)
    snapshots = [m for m in messages if m['type'] == 'transcript']
    assert len(snapshots) >= 4, 'Too few overlapping updates'
    assert snapshots[-1]['final'] and not any(m['final'] for m in snapshots[:-1])
    assert snapshots[-1]['window_end_ms'] == (count - 1) * 40
    assert snapshots[-1]['draft_text'] == ''
    assert snapshots[0]['window_start_ms'] == 0
    assert any(m['window_start_ms'] > 0 for m in snapshots)
    assert [m['revision'] for m in snapshots] == sorted({m['revision'] for m in snapshots})
    for previous, current in zip(snapshots, snapshots[1:]):
        assert previous['window_end_ms'] - current['window_start_ms'] >= 1000
    for snapshot in snapshots:
        assert snapshot['text'] == ' '.join(filter(None, [snapshot['stable_text'], snapshot['draft_text']]))
        assert snapshot['window_end_ms'] - snapshot['window_start_ms'] <= 6000
    if speech_gate:
        assert all(not m['text'] and m['timings_ms']['inference'] == 0 for m in snapshots)
    else:
        assert any(m['timings_ms']['inference'] > 0 for m in snapshots)
    # Captions are deliberately omitted: a static photograph is not speech.
    return {'passed': True, 'speech_gate': speech_gate, 'translated': translated, 'frames': count,
            'updates': len(snapshots), 'windows': [
                {key: m[key] for key in ('revision', 'window_start_ms', 'window_end_ms', 'final', 'timings_ms')}
                for m in snapshots]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', default='http://127.0.0.1:8000')
    parser.add_argument('--seconds', type=float, default=12.12)
    parser.add_argument('--filter', choices=('on', 'off'), default='off')
    parser.add_argument('--translated', action='store_true', help='Exercise head-movement rejection with a translated public photo.')
    args = parser.parse_args()
    if not 8 <= args.seconds <= 30:
        parser.error('--seconds must be between 8 and 30')
    print(json.dumps(asyncio.run(check(args.origin.rstrip('/'), args.seconds, args.filter == 'on', args.translated)), indent=2))

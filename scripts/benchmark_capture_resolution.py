#!/usr/bin/env python3
"""Local live cost check; resized legacy clips are NOT an accuracy benchmark.

Captions and frames stay in memory. Only numeric timings and presence flags are
saved. Replays the explicitly recorded normal-speech example at camera cadence.
"""
import argparse
import asyncio
from io import BytesIO
import json
from pathlib import Path
import statistics
import struct
import sys
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from scripts.check_speaking_gate import load_records, packets_for_record, write_private
from scripts.profile_pipeline import replay


async def run(args):
    directory, manifest = load_records()['1fb0646dde9f4530bd617327155c3bcd']
    original = packets_for_record(directory, manifest)
    large = []
    for stamp, packet in original:
        with Image.open(BytesIO(packet[8:])) as image:
            output = BytesIO()
            image.resize((1280, 720), Image.Resampling.BILINEAR).save(output, 'JPEG', quality=92)
            large.append((stamp, struct.pack('<d', stamp) + output.getvalue()))
    results = []
    for index, (label, packets) in enumerate([('640x360', original), ('1280x720', large)] * 2):
        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/status', timeout=5) as response:
            status = json.load(response)
        if not status['ready'] or status['busy']:
            raise RuntimeError('The local model must be ready and idle.')
        result = await replay(args.port, label, packets, SimpleNamespace())
        result['round'] = index // 2 + 1
        recognition = [s for s in result['samples'] if s['timings_ms'].get('inference', 0) > 0]
        result['summary'] = {
            'recognition_windows': len(recognition),
            'median_server_ms': statistics.median(s['server_processing_ms'] for s in recognition) if recognition else None,
            'max_queue_ms': max(s['timings_ms'].get('queue', 0) for s in result['samples']),
            'final_caption_present': result['samples'][-1]['caption_present'],
            'final_received': result['samples'][-1]['final'],
            'uploaded_megabytes': round(sum(len(packet) for _, packet in packets) / 1e6, 2),
        }
        results.append(result)
        print(json.dumps({'size': label, 'round': result['round'], **result['summary']}), flush=True)
    write_private(args.output, {'note': 'Cost and integration check only: upscaled legacy images cannot establish native high-resolution word accuracy.', 'results': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/visual_speech/distance-live-cost.json')
    asyncio.run(run(parser.parse_args()))

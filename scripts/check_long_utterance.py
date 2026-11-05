#!/usr/bin/env python3
"""Exercise longer complete inputs using local, explicitly recorded examples.

Joined/repeated clips validate transport, bounds and runtime, not long-speech
word accuracy. Generated words and camera frames are never included in reports.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.limits import MAX_UTTERANCE_SECONDS, UTTERANCE_CACHE_FRAMES
from backend.stream import Clip, Frame, DecodeCache, resample
from backend.preprocessing import PreprocessCache
from scripts.check_speaking_gate import load_records, packets_for_record, replay_live, write_private


def joined_packets(seconds):
    records = load_records()
    sources = [packets_for_record(directory, manifest) for directory, manifest in records.values()
               if manifest['take'] == 2 and manifest['label'] in ('speech_normal', 'speech_subtle')]
    if len(sources) != 2:
        raise ValueError('Exactly two second-take speech recordings are required.')
    packets = []
    offset, source_index = 0., 0
    while offset <= seconds * 1000:
        for timestamp, packet in sources[source_index % 2]:
            current = offset + timestamp
            if current > seconds * 1000:
                break
            packets.append((current, struct.pack('<d', current) + packet[8:]))
        offset = packets[-1][0] + 40.
        source_index += 1
    return packets


def paused_packets():
    records = {(manifest['label'], manifest['take']): (directory, manifest)
               for directory, manifest in load_records().values()}
    packets = []
    # Exercise a session beyond the default limit, with three naturally gated
    # utterances and quiet between them. Only the joins are artificial.
    for label, duration in [('speech_subtle', 8), ('quiet_still', 4), ('speech_normal', 8),
                            ('quiet_still', 4), ('speech_subtle', 8), ('quiet_still', 8)]:
        offset = packets[-1][0] + 40. if packets else 0.
        directory, manifest = records[(label, 2)]
        packets.extend((timestamp, packet) for timestamp, packet in packets_for_record(directory, manifest, offset_ms=offset)
                       if timestamp < offset + duration * 1000)
    return packets


def model_check(packets, output):
    import torch
    from backend.model import FlareModel
    runtime = FlareModel()
    runtime.load()
    decode = DecodeCache(max_frames=UTTERANCE_CACHE_FRAMES)
    cache = PreprocessCache(max_frames=UTTERANCE_CACHE_FRAMES)
    frames = tuple(Frame(timestamp, packet[8:]) for timestamp, packet in packets)
    rows = []
    first_frame = None
    for seconds in (12, 20, 30, 45, 60):
        source = tuple(frame for frame in frames if frame.timestamp <= seconds * 1000)
        rgb, _ = resample(Clip(0, source, True, sample_origin=0.), cache=decode)
        if first_frame is None:
            first_frame = rgb[0]
        assert rgb[0] is first_frame
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        result = runtime.decode_utterance(rgb, 25, cache=cache, verify_visual=True)
        torch.cuda.synchronize()
        row = {'seconds': seconds, 'frames': len(rgb), 'preprocessed_frames': result['quality']['frames'],
               'elapsed_ms': round((time.perf_counter() - begin) * 1000, 1),
               'timings_ms': result['timings_ms'], 'caption_present': bool(result['text']),
               'verification_state': result.get('recognition', {}).get('state'),
               'cache': cache.stats(), 'peak_gpu_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        assert row['preprocessed_frames'] == len(rgb)
        rows.append(row)
        write_private(output, {'created_at': datetime.now(timezone.utc).isoformat(), 'rows': rows,
                              'caption_text_saved': False, 'passed': len(rows) == 5,
                              'limitations': 'Constructed speech input; no reference transcript or natural long-utterance accuracy claim.'})
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Exercise the deployed WebSocket with one 60-second input.')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/visual_speech/long-utterance-validation.json')
    args = parser.parse_args()
    with urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=5) as response:
        status = json.load(response)
    if status.get('busy'):
        raise SystemExit('A live session is active; wait until it finishes.')
    packets = joined_packets(MAX_UTTERANCE_SECONDS)
    if args.live:
        if status.get('max_utterance_seconds') != MAX_UTTERANCE_SECONDS:
            raise SystemExit('The deployed server still has the old utterance limit.')
        # Deliberately bypass only activity segmentation to force a single long
        # utterance. Visual/repetition verification remains enabled throughout.
        row = asyncio.run(replay_live('continuous_long_speech', packets, speech_gate=False,
                                     max_utterance_seconds=MAX_UTTERANCE_SECONDS))
        intervals = row['decoded_intervals']
        final = [interval for interval in intervals if interval['final']]
        assert len(final) == 1 and final[0]['end_ms'] - final[0]['start_ms'] > 59_000.
        assert all(interval['start_ms'] == 0. for interval in intervals)
        assert row['actual_decode_calls'] > 1
        paused = asyncio.run(replay_live('long_session_with_pauses', paused_packets(), mixed=True, expected_utterances=3))
        assert paused['duration_seconds'] > 30
        write_private(args.output, {'passed': True, 'caption_text_saved': False, 'live': row, 'paused_session': paused,
                                   'limitations': 'Gate bypassed only in the full-minute case; constructed input is not an accuracy benchmark.'})
        print(json.dumps({key: row[key] for key in ('passed', 'actual_decode_calls', 'refinement_passes', 'caption_present')}))
    else:
        model_check(packets, args.output)


if __name__ == '__main__':
    main()

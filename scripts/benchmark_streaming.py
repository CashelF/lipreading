#!/usr/bin/env python3
"""Compare uninstrumented real-time replays against the still-running baseline.

Starts an isolated candidate listener, preserves production, and saves numeric
measurements/equality flags only. No captions, token IDs or video are exported.
"""
import argparse
import asyncio
import json
from pathlib import Path
import socket
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_speaking_gate import load_records, packets_for_record, write_private
from scripts.profile_pipeline import replay


def assert_idle(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/status', timeout=3) as response:
        if json.load(response).get('busy'):
            raise RuntimeError('A reading session is active; stopping the benchmark.')


class Context:
    pass


async def run(args, runtime):
    import uvicorn
    from backend import app as application
    application.model = runtime
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(application.app, log_level='warning', access_log=False))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(.1)
        if not server.started:
            raise RuntimeError('Candidate listener did not start.')
        records = {(m['label'], m['take']): (d, m) for d, m in load_records().values()}
        normal = packets_for_record(*records['speech_normal', 2])
        subtle = packets_for_record(*records['speech_subtle', 2])
        quiet = packets_for_record(*records['quiet_still', 2], offset_ms=subtle[-1][0] + 40)
        motion = packets_for_record(*records['quiet_motion', 2])
        report = {'transport': 'local loopback; excludes laptop capture/encoding/network/paint',
                  'instrumented': False, 'caption_text_saved': False, 'scenarios': []}
        scenarios = [('normal', normal), ('subtle_then_quiet', subtle + quiet), ('quiet_motion', motion)]
        for index, (label, packets) in enumerate(scenarios):
            rows = {}
            order = [('before', args.baseline_port), ('after', port)]
            if index % 2:
                order.reverse()
            for name, target in order:
                assert_idle(args.baseline_port)
                rows[name] = await replay(target, label, packets, Context(), compare_text=True)
            same = rows['before'].pop('_comparison_text') == rows['after'].pop('_comparison_text')
            report['scenarios'].append({'label': label, 'final_text_equal': same, **rows})
            write_private(args.output, report)
            print(json.dumps({'scenario': label, 'final_text_equal': same,
                              'before_updates': len(rows['before']['samples']),
                              'after_updates': len(rows['after']['samples'])}), flush=True)
    finally:
        server.should_exit = True
        await task
        listener.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-port', type=int, default=8000)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/visual_speech/streaming-speed-comparison.json')
    args = parser.parse_args()
    assert_idle(args.baseline_port)
    from backend.model import FlareModel
    model = FlareModel()
    model.load()
    asyncio.run(run(args, model))


if __name__ == '__main__':
    main()

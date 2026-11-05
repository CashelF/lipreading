#!/usr/bin/env python3
"""Profile an isolated copy of the live app on consented local camera examples.

No production patch, captions, token IDs or images are saved. GPU synchronizing
probes add overhead, so use production logs for ordinary latency and these
measurements to attribute work. Loopback excludes the user's laptop/network.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import functools
import json
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_speaking_gate import load_records, packets_for_record, write_private


class Profiler:
    def __init__(self):
        self.local = threading.local()
        self.current = None
        self.rows = []

    @contextmanager
    def section(self, name, sync=False):
        import torch
        if self.current is None:
            yield
            return
        stack = getattr(self.local, 'stack', [])
        path = '/'.join([*stack, name])
        self.local.stack = [*stack, name]
        if sync:
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            values = self.current['stages'].setdefault(path, {'ms': 0., 'calls': 0})
            values['ms'] += elapsed
            values['calls'] += 1
            self.local.stack = stack

    def wrap(self, owner, attr, name, sync=False):
        original = getattr(owner, attr)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            label = name(args, kwargs) if callable(name) else name
            with self.section(label, sync):
                return original(*args, **kwargs)
        setattr(owner, attr, measured)

    def install(self, runtime):
        import backend.grounding as grounding
        import backend.preprocessing as preprocessing
        import backend.visual_speech as visual_speech
        import backend.utterance_stream as utterance_stream
        import skimage.transform
        from backend.speech_gate import SpeechGateState

        self.wrap(runtime, 'inspect_activity', 'gate')
        self.wrap(runtime, 'decode_utterance', 'recognition')
        self.wrap(runtime, 'refine_utterance', 'recognition')
        self.wrap(runtime.preprocessor, '_prepare', 'preprocess')
        self.wrap(runtime.preprocessor.detector, 'get_landmarks_from_image', 'landmarks_including_detection', True)
        self.wrap(runtime.preprocessor.detector.face_detector, 'detect_from_image', 'face_detection', True)
        self.wrap(skimage.transform, 'warp', 'face_warp')
        self.wrap(skimage.transform.SimilarityTransform, 'estimate', 'alignment_fit')
        self.wrap(preprocessing, 'normalize_roi', 'resize_normalize')
        self.wrap(visual_speech, 'crop_faces', 'speaking_crops')
        self.wrap(runtime.preprocessor.visual_speech, 'predict_crops', 'speaking_network', True)
        self.wrap(SpeechGateState, 'update', 'speaking_policy')
        self.wrap(runtime.model.encoder, 'forward', 'visual_encoder', True)
        self.wrap(runtime.model.modality_projection, 'forward', 'visual_projection', True)
        self.wrap(grounding, 'verify_candidate', 'verification', True)
        qwen, lm = grounding.qwen_parts(runtime.model)
        self.wrap(qwen, 'forward', lambda args, kwargs: 'qwen_prefill' if kwargs.get('inputs_embeds') is not None else 'qwen_step', True)
        self.wrap(lm.lm_head, 'forward', 'vocabulary_projection', True)
        namespace = runtime._search.__globals__
        original_beam = namespace['beam_search']
        @functools.wraps(original_beam)
        def beam(ids, *args, **kwargs):
            with self.section('beam_search', True):
                result = original_beam(ids, *args, **kwargs)
            self.current['candidates'].append({'winning_tokens': result[0].numel() - ids.shape[1],
                                               'completed_candidates': len(result[2])})
            return result
        namespace['beam_search'] = beam
        original_resample = utterance_stream.resample
        original_decode = utterance_stream.UtteranceReader._decode
        self.decoding = False
        async def decode(reader, *args, **kwargs):
            self.decoding = True
            try:
                return await original_decode(reader, *args, **kwargs)
            finally:
                self.decoding = False
        utterance_stream.UtteranceReader._decode = decode
        def resample(clip, *args, **kwargs):
            # Activity and recognition now share the immutable decode cache.
            kind = 'jpeg_recognition' if self.decoding else 'jpeg_gate'
            with self.section(kind):
                return original_resample(clip, *args, **kwargs)
        utterance_stream.resample = resample
        original_process = utterance_stream.UtteranceReader.process
        async def process(reader, clip, speech_gate, send):
            self.reader = reader
            self.current = {'scenario': self.scenario, 'segment_id': clip.segment_id, 'source_end_ms': clip.end_ms,
                            'final': clip.final, 'stages': {}, 'candidates': []}
            before = {name: cache.stats() for name, cache in [('gate', reader.gate_preprocess), ('recognition', reader.utterance_preprocess)]}
            begin = time.perf_counter()
            try:
                result = await original_process(reader, clip, speech_gate, send)
                self.current.update(processing_ms=(time.perf_counter() - begin) * 1000,
                                    production_timings_ms=result['timings_ms'].copy(),
                                    decoded_intervals=result['decoded_intervals'],
                                    caption_present=bool(result['text']))
                self.current['cache_deltas'] = {name: {key: value - before[name][key] for key, value in cache.stats().items()}
                                                for name, cache in [('gate', reader.gate_preprocess), ('recognition', reader.utterance_preprocess)]}
                self.rows.append(self.current)
                return result
            finally:
                self.current = None
        utterance_stream.UtteranceReader.process = process


async def replay(port, label, packets, profiler, *, compare_text=False):
    from websockets.asyncio.client import connect
    # This is an isolated listener, using an origin already admitted by the app.
    profiler.scenario = label
    samples = []
    private_text = ''
    async with connect(f'ws://127.0.0.1:{port}/api/stream', origin='http://127.0.0.1:8000') as ws:
        await ws.send(json.dumps({'type': 'start', 'speech_gate': True, 'verify_visual': True, 'max_utterance_seconds': 30}))
        first = json.loads(await ws.recv())
        if first.get('state') != 'capturing':
            raise RuntimeError('Profile session was not admitted.')
        began = time.perf_counter()
        async def receive():
            nonlocal private_text
            async for raw in ws:
                message = json.loads(raw)
                received_ms = (time.perf_counter() - began) * 1000
                if message['type'] == 'error':
                    raise RuntimeError('Profile session had an inference error; see local server diagnostics.')
                if message['type'] == 'transcript':
                    private_text = message['text']
                    samples.append({'segment_id': message.get('segment_id'), 'received_ms': received_ms,
                                    'frame_to_response_ms': received_ms - message.get('window_end_ms', received_ms),
                                    'server_processing_ms': message.get('latency_ms'),
                                    'timings_ms': message.get('timings_ms'), 'caption_present': bool(message['text']),
                                    'utterances': len(message.get('utterances', [])),
                                    'final': message.get('final', False)})
                if message['type'] == 'stopped':
                    return
        receiver = asyncio.create_task(receive())
        for timestamp, packet in packets:
            await asyncio.sleep(max(0., began + timestamp / 1000 - time.perf_counter()))
            if receiver.done():
                await receiver
                raise RuntimeError('Profile closed before all frames were sent.')
            await ws.send(packet)
        stopped_at_ms = (time.perf_counter() - began) * 1000
        await ws.send(json.dumps({'type': 'stop'}))
        try:
            await asyncio.wait_for(receiver, 90)
        finally:
            if not receiver.done():
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
    result = {'label': label, 'duration_ms': packets[-1][0], 'stop_sent_ms': stopped_at_ms, 'samples': samples}
    if compare_text:
        # Comparison callers must remove this transient field before reporting.
        result['_comparison_text'] = private_text
    return result


async def run(args, runtime, profiler):
    import uvicorn
    from backend import app as server_module
    server_module.model = runtime
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    config = uvicorn.Config(server_module.app, log_level='warning', access_log=False, lifespan='on')
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(.1)
        if not server.started:
            raise RuntimeError('Isolated profiler listener did not start.')
        records = {(m['label'], m['take']): (d, m) for d, m in load_records().values()}
        report = {'created_at': datetime.now(timezone.utc).isoformat(), 'production_modified': False,
                  'caption_text_saved': False, 'timing_method': 'Wall-clock probes; CUDA synchronization at GPU boundaries.',
                  'limitations': ['Instrumentation adds overhead.', 'Local replay excludes laptop camera/JPEG encoding, tailnet transport, and screen rendering.',
                                  'Source clips are saved personal development examples, not natural long-speech ground truth.'],
                  'scenarios': [], 'profile': profiler.rows}
        normal = packets_for_record(*records[('speech_normal', 2)])
        subtle = packets_for_record(*records[('speech_subtle', 2)])
        quiet = packets_for_record(*records[('quiet_still', 2)], offset_ms=subtle[-1][0] + 40.)
        for label, packets in [('normal', normal), ('subtle_then_quiet', subtle + quiet)]:
            with urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=3) as response:
                if json.load(response).get('busy'):
                    raise RuntimeError('A user session started; stopping the separate profiling workload.')
            result = await replay(port, label, packets, profiler)
            report['scenarios'].append(result)
            write_private(args.output, report)
            print(json.dumps({'scenario': label, 'updates': len(result['samples']), 'profiled_windows': len(profiler.rows)}), flush=True)
    finally:
        server.should_exit = True
        await task
        listener.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/visual_speech/pipeline-profile.json')
    args = parser.parse_args()
    with urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=3) as response:
        if json.load(response).get('busy'):
            raise SystemExit('Wait for the active user session to finish before profiling.')
    from backend.model import FlareModel
    runtime = FlareModel()
    runtime.load()
    profiler = Profiler()
    profiler.install(runtime)
    asyncio.run(run(args, runtime, profiler))


if __name__ == '__main__':
    main()

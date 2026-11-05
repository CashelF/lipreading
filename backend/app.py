"""Single-user camera inference server; only explicit examples are recorded."""
import asyncio
from collections import deque
from contextlib import asynccontextmanager
import json
import logging
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.model import FlareModel, ModelUnavailable
from backend.stream import (Clip, ClipBuffer, DecodeCache, PreparationSampler, resample, sample_times, ensure_overlap,
                            SEGMENT_MS, PARTIAL_MS, INITIAL_MS)
from backend.preprocessing import PreprocessCache
from backend.transcript import ContinuousTranscript
from backend.examples import create_examples_router
from backend.utterance_stream import UtteranceReader, UtteranceLimitError
from backend.speech_gate import START_SCORE_THRESHOLD, START_CONFIRM_FRAMES
from backend.limits import MAX_UTTERANCE_SECONDS, DEFAULT_UTTERANCE_SECONDS, UTTERANCE_OPTIONS

log = logging.getLogger(__name__)
model = FlareModel()
loading = False
session_lock = asyncio.Lock()


async def load_model():
    global loading
    loading = True
    try:
        await asyncio.to_thread(model.load)
    except ModelUnavailable as exc:
        log.warning('Model unavailable: %s', exc)
    except Exception:
        log.exception('Model could not load; camera preview remains available')
    finally:
        loading = False


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(load_model())
    yield
    await task


app = FastAPI(title='Still — Visual speech studio', lifespan=lifespan)


@app.get('/api/status')
async def status():
    result = await asyncio.to_thread(model.status)
    return {**result, 'loading': loading, 'busy': session_lock.locked(),
            'timing_probes': True,
            'fps': 25, 'segment_seconds': SEGMENT_MS / 1000,
            'partial_seconds': PARTIAL_MS / 1000, 'initial_seconds': INITIAL_MS / 1000,
            'window_seconds': SEGMENT_MS / 1000, 'continuous_revisions': True,
            'utterance_refinement': hasattr(model, 'inspect_activity'),
            'speech_start_confirmation': {'threshold': START_SCORE_THRESHOLD,
                                          'consecutive_means': START_CONFIRM_FRAMES},
            'pause_regions_version': 1 if hasattr(model, 'inspect_activity') else None,
            'visual_grounding': {'available': bool(getattr(model, 'supports_visual_grounding', False)),
                                 'default': True, 'experimental': True,
                                 'policy': 'phrase-and-repetition-v2'},
            'max_utterance_seconds': MAX_UTTERANCE_SECONDS,
            'default_utterance_seconds': DEFAULT_UTTERANCE_SECONDS,
            'utterance_options': list(UTTERANCE_OPTIONS)}


@app.post('/api/model/load')
async def reload_model(request: Request):
    global loading
    origin = request.headers.get('origin')
    if origin and not allowed_origin(origin, request.headers.get('host')):
        raise HTTPException(status_code=403, detail='Origin is not allowed.')
    if model.status().get('ready'):
        return {'loading': False, 'ready': True}
    if not loading and not session_lock.locked():
        # Set synchronously before scheduling, so duplicate clicks cannot race.
        loading = True
        asyncio.create_task(load_model())
    return {'loading': loading}

def allowed_origin(origin: str | None, host: str | None) -> bool:
    if not origin:
        return False
    parts = urlsplit(origin)
    if parts.scheme not in ('http', 'https'):
        return False
    allowed = {'http://localhost:5173', 'http://127.0.0.1:5173',
               'http://localhost:8000', 'http://127.0.0.1:8000'}
    allowed.update(x.strip() for x in os.getenv('LIPREAD_ALLOWED_ORIGINS', '').split(',') if x.strip())
    # Explicit allowlist prevents arbitrary sites using the local GPU/camera socket.
    return origin in allowed


app.include_router(create_examples_router(allowed_origin, lambda: session_lock))


@app.websocket('/api/stream')
async def stream(ws: WebSocket):
    if not allowed_origin(ws.headers.get('origin'), ws.headers.get('host')):
        await ws.close(code=1008)
        return
    await ws.accept()
    if session_lock.locked():
        await ws.send_json({'type': 'error', 'detail': 'Another reading session is active.'})
        await ws.close(code=1013)
        return
    await session_lock.acquire()
    send_lock = asyncio.Lock()
    wake = asyncio.Event()
    pending: Clip | None = None
    admitted: Clip | None = None
    stopped = False
    disconnected = False
    started = False
    speech_gate = True
    buffer = ClipBuffer()
    decoded_cache = DecodeCache()
    preprocessing_cache = PreprocessCache()
    reader = UtteranceReader(model) if hasattr(model, 'inspect_activity') else None
    transcript = reader.transcript if reader is not None else ContinuousTranscript()
    # At most a small camera backlog; all unprepared frames remain available to
    # ordinary inference in VideoHistory. Preparation never advances the gate.
    preparing = reader is not None and callable(getattr(model, 'prepare_frame', None))
    preparation_queue = deque(maxlen=64)
    preparation_sampler = PreparationSampler()

    def enqueue(clip: Clip):
        nonlocal pending
        # One in-flight request and one newest pending window. Superseded
        # requests still have their video represented by the overlapping window.
        if reader is None:
            ensure_overlap(admitted, clip)
        pending = clip
        wake.set()

    async def send(payload):
        if not disconnected:
            async with send_lock:
                await ws.send_json(payload)

    async def worker():
        nonlocal pending, admitted, preparing
        last_success: Clip | None = None
        last_result = None
        last_snapshot = None
        while True:
            await wake.wait()
            wake.clear()
            processed = False
            while pending is not None or preparation_queue:
                if disconnected:
                    return
                if pending is None:
                    source = preparation_queue.popleft()
                    try:
                        # One frame at a time: a new scheduled update can wait
                        # behind at most one frame, never a preparation batch.
                        await asyncio.to_thread(reader.prepare_frame, source, speech_gate)
                    except ValueError:
                        # Speculative quality failures are handled at the normal
                        # window boundary, including multi-face/corrupt input.
                        pass
                    except Exception:
                        preparing = False
                        preparation_queue.clear()
                        log.warning('Early frame preparation disabled for this session; using normal preprocessing.')
                    continue
                clip, pending = pending, None
                processed = True
                admitted = clip
                await send({'type': 'status', 'state': 'processing', 'detail': 'Reading visual speech…'})
                begin = time.perf_counter()
                frames = None
                try:
                    # Stop often arrives on an already processed camera frame.
                    # Finalize the displayed tail without repeating the GPU call.
                    same_source = (last_success is not None and clip.final
                                   and clip.start_ms == last_success.start_ms
                                   and clip.end_ms == last_success.end_ms
                                   and len(clip.frames) == len(last_success.frames)
                                   and all(a is b for a, b in zip(clip.frames, last_success.frames)))
                    if reader is not None:
                        result = await reader.process(clip, speech_gate, send)
                        snapshot = result['snapshot']
                        capture_detail = result['detail']
                        timings = result['timings_ms']
                        last_success, last_result = clip, result
                    elif same_source:
                        result = last_result
                        snapshot = transcript.finish()
                        capture_detail = 'Finalized the current transcript.'
                        timings = {'preprocessing': 0., 'upload': 0., 'inference': 0., 'frame_decode': 0.}
                    else:
                        frames, capture_fps = await asyncio.to_thread(resample, clip, cache=decoded_cache)
                        sampled_at = time.perf_counter()
                        preprocessing_cache.set_sample_times(sample_times(clip))
                        result = await asyncio.to_thread(model.transcribe, frames, 25,
                                                         cache=preprocessing_cache, speech_gate=speech_gate)
                        activity = result.get('activity', {})
                        if activity.get('suppress') and activity.get('preserve_transcript'):
                            snapshot = transcript.hold(final=clip.final)
                        else:
                            snapshot = transcript.update(result['text'], clip.start_ms, clip.end_ms,
                                                         final=clip.final,
                                                         suppressed=activity.get('suppress', False))
                        capture_detail = f'{capture_fps:.1f} fps captured · continuous overlapping updates'
                        timings = {**result.get('timings_ms', {}),
                                   'frame_decode': round((sampled_at - begin) * 1000, 1)}
                        last_success, last_result = clip, result
                    timings['queue'] = round((begin - clip.queued_at) * 1000, 1)
                    last_snapshot = snapshot
                    log.info('Window %s final=%s span_ms=%.0f frames=%s timing_ms=%s',
                             clip.segment_id, clip.final, clip.end_ms - clip.start_ms,
                             len(frames) if frames is not None else 'reused', timings)
                    if result.get('activity') is not None:
                        activity = result['activity']
                        # Numeric gate diagnostics only: no camera frames or
                        # recognized words are written to application logs.
                        log.info('Activity window=%s state=%s suppress=%s metrics=%s',
                                 clip.segment_id, activity.get('state'),
                                 activity.get('suppress'), activity.get('metrics', {}))
                        await send({**result['activity'], 'type': 'activity', 'segment_id': clip.segment_id})
                    await send({**snapshot, 'type': 'transcript', 'segment_id': clip.segment_id,
                                'window_start_ms': clip.start_ms, 'window_end_ms': clip.end_ms,
                                **({'decoded_intervals': result.get('decoded_intervals', [])}
                                   if reader is not None else {}),
                                'timings_ms': timings,
                                'latency_ms': round((time.perf_counter() - begin) * 1000)})
                    await send({'type': 'quality', 'detail': result.get('detail', capture_detail)})
                except UtteranceLimitError as exc:
                    last_snapshot = transcript.finish()
                    await send({**last_snapshot, 'type': 'transcript', 'segment_id': clip.segment_id, 'latency_ms': 0})
                    await send({'type': 'error', 'detail': str(exc)})
                    return
                except ValueError as exc:
                    await send({'type': 'quality', 'detail': str(exc)})
                    if clip.final:
                        last_snapshot = transcript.finish()
                        await send({**last_snapshot, 'type': 'transcript', 'segment_id': clip.segment_id, 'latency_ms': 0})
                except Exception:
                    log.exception('Visual speech inference failed')
                    last_snapshot = transcript.finish()
                    await send({**last_snapshot, 'type': 'transcript', 'segment_id': clip.segment_id, 'latency_ms': 0})
                    await send({'type': 'error', 'detail': 'Inference failed. Check the model setup and server log.'})
                finally:
                    frames = None
            if stopped:
                if last_snapshot is not None and not last_snapshot['final']:
                    await send({**transcript.finish(), 'type': 'transcript', 'latency_ms': 0})
                await send({'type': 'stopped'})
                return
            if processed:
                await send({'type': 'status', 'state': 'capturing', 'detail': 'Following your speech · recent words may update'})

    worker_task = asyncio.create_task(worker())
    try:
        while not stopped:
            message = await asyncio.wait_for(ws.receive(), timeout=30)
            if message['type'] == 'websocket.disconnect':
                raise WebSocketDisconnect()
            if message.get('bytes') is not None:
                if not started:
                    raise ValueError('Start a reading session before sending video.')
                clip = buffer.append(message['bytes'])
                if reader is not None and buffer.frames:
                    reader.history.append(buffer.frames[-1])
                    if preparing:
                        preparation_queue.extend(preparation_sampler.append(buffer.frames[-1]))
                        if preparation_queue:
                            wake.set()
                if clip:
                    enqueue(clip)
                continue
            text = message.get('text', '')
            if len(text) > 1024:
                raise ValueError('Control message is too large.')
            command = json.loads(text)
            if not isinstance(command, dict):
                raise ValueError('Invalid control message.')
            if command.get('type') == 'start':
                if started:
                    raise ValueError('Session already started.')
                speech_gate = command.get('speech_gate', True)
                if type(speech_gate) is not bool:
                    raise ValueError('speech_gate must be true or false.')
                verify_visual = command.get('verify_visual', True)
                if type(verify_visual) is not bool:
                    raise ValueError('verify_visual must be true or false.')
                if reader is not None:
                    reader.verify_visual = verify_visual
                    reader.set_utterance_limit(command.get('max_utterance_seconds', DEFAULT_UTTERANCE_SECONDS))
                if loading or not model.status().get('ready'):
                    raise ValueError(model.status().get('detail', 'The model is not ready.'))
                started = True
                log.info('Reading session started speech_gate=%s utterance_seconds=%s',
                         speech_gate, reader.max_utterance_seconds if reader is not None else None)
                await send({'type': 'status', 'state': 'capturing', 'detail': 'Camera connected · no audio',
                            'max_utterance_seconds': reader.max_utterance_seconds if reader is not None else None})
            elif command.get('type') == 'ping':
                sent_ms = command.get('client_time_ms')
                if not started or type(sent_ms) not in (int, float) or not math.isfinite(sent_ms) or sent_ms < 0:
                    raise ValueError('Invalid timing probe.')
                await send({'type': 'pong', 'client_time_ms': sent_ms})
            elif command.get('type') == 'stop':
                stopped = True
                preparation_queue.clear()
                clip = buffer.finish()
                if clip:
                    enqueue(clip)
                wake.set()
            else:
                raise ValueError('Unknown stream command.')
        await asyncio.shield(worker_task)
    except WebSocketDisconnect:
        disconnected = True
    except asyncio.TimeoutError:
        await send({'type': 'error', 'detail': 'Camera stream timed out. Start a new session.'})
    except (ValueError, json.JSONDecodeError) as exc:
        await send({'type': 'error', 'detail': str(exc)})
    finally:
        disconnected = True
        stopped = True
        pending = None
        preparation_queue.clear()
        # Let in-flight CUDA finish before releasing GPU ownership. Session caches
        # persist across moving windows, and release all video at teardown.
        wake.set()
        try:
            await asyncio.shield(worker_task)
        except Exception:
            log.exception('Stream ended while inference was running')
        decoded_cache.clear()
        preprocessing_cache.clear()
        if reader is not None:
            reader.clear()
        session_lock.release()
        try:
            await ws.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


DIST = Path(__file__).resolve().parents[1] / 'dist'
if DIST.exists():
    app.mount('/assets', StaticFiles(directory=DIST / 'assets'), name='assets')

    @app.get('/')
    async def index():
        return FileResponse(DIST / 'index.html')

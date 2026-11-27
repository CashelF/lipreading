"""Explicit, local-only recording of labeled camera examples.

Normal reading never calls this module. Frames remain in bounded memory until
the user finishes a sample; only then are the original JPEGs saved locally.
Two collections share this code: eight-second speaking-filter examples and
held-out read sentences for word error rate (see docs/EVALUATION.md).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Callable
import uuid

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from PIL import Image, UnidentifiedImageError

from backend.stream import Frame, parse_packet

LABELS = frozenset({'quiet_still', 'quiet_motion', 'speech_normal', 'speech_subtle'})
MAX_FRAMES = 400
MAX_BYTES = 128 * 1024 * 1024
MAX_SAMPLES = 100
SESSION_TIMEOUT_SECONDS = 30
SAMPLE_ID = re.compile(r'[0-9a-f]{32}\Z')
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'speaking-examples'
SENTENCE_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'eval-sentences'
SENTENCE_PROMPTS = Path(__file__).resolve().with_name('eval_sentences.json')
MAX_SENTENCE_SAMPLES = 2000
SPEAKER = re.compile(r'[A-Za-z0-9_-]{1,16}\Z')
EXAMPLE_KEYS = ('id', 'label', 'take', 'frames', 'duration_seconds')
SENTENCE_KEYS = ('id', 'prompt_id', 'speaker', 'frames', 'duration_seconds')


def load_prompts(path: Path = SENTENCE_PROMPTS) -> dict[str, str]:
    """Fixed public prompt list: ids stay stable so recordings keep their reference."""
    sentences = json.loads(path.read_text())['sentences']
    prompts = {item['id']: item['text'] for item in sentences}
    if len(prompts) != len(sentences):
        raise ValueError('Duplicate evaluation sentence id.')
    return prompts


PROMPTS = load_prompts()


class ExampleBuffer:
    min_seconds, max_seconds = 6, 10
    length_message = 'Keep each sample between six and ten seconds.'

    def __init__(self, label: str, take: int):
        if not isinstance(label, str) or label not in LABELS:
            raise ValueError('Choose a valid speaking example.')
        if type(take) is not int or take not in (1, 2):
            raise ValueError('Take must be 1 or 2.')
        self.label, self.take = label, take
        self._reset()

    def _reset(self):
        self.frames: list[Frame] = []
        self.total_bytes = 0
        self.dimensions: tuple[int, int] | None = None

    def append(self, packet: bytes):
        if len(self.frames) >= MAX_FRAMES or self.total_bytes + len(packet) > MAX_BYTES:
            raise ValueError('This recording is too large. Record a new eight-second sample.')
        previous = self.frames[-1].timestamp if self.frames else None
        frame = parse_packet(packet, previous)
        if previous is not None and frame.timestamp - previous > 250:
            raise ValueError('Camera frames were interrupted. Record this sample again.')
        if self.frames and frame.timestamp - self.frames[0].timestamp > self.max_seconds * 1000:
            raise ValueError(self.length_message)
        try:
            with Image.open(BytesIO(frame.jpeg)) as image:
                if image.format != 'JPEG' or not (96 <= image.width <= 1280 and 96 <= image.height <= 960):
                    raise ValueError('Sample camera frames must be between 96×96 and 1280×960 pixels.')
                image.load()  # Validate the entire JPEG, not just its header.
                dimensions = image.size
                if image.mode not in ('RGB', 'L'):
                    raise ValueError('Sample camera frames must be RGB or grayscale JPEGs.')
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError('Could not decode a sample camera frame.') from exc
        if self.dimensions is not None and dimensions != self.dimensions:
            raise ValueError('Camera resolution changed. Record this sample again.')
        self.dimensions = dimensions
        self.frames.append(frame)
        self.total_bytes += len(packet)

    def timing(self) -> dict:
        if len(self.frames) < 2:
            raise ValueError('Record a full sample before saving.')
        duration = (self.frames[-1].timestamp - self.frames[0].timestamp) / 1000
        if not self.min_seconds <= duration <= self.max_seconds:
            raise ValueError(self.length_message)
        capture_fps = (len(self.frames) - 1) / duration
        if capture_fps < 15:
            raise ValueError('Camera frames arrived too slowly. Improve lighting and record this sample again.')
        return {'frames': len(self.frames), 'duration_seconds': round(duration, 3),
                'capture_fps': round(capture_fps, 3)}

    def metadata(self) -> dict:
        return {'label': self.label, 'take': self.take, **self.timing()}

    def conflicts(self, saved: dict) -> bool:
        return False


class SentenceBuffer(ExampleBuffer):
    """One read prompt sentence, kept as a held-out word error rate clip."""
    min_seconds, max_seconds = 1.5, 15
    length_message = 'Keep each sentence recording between 1.5 and 15 seconds.'

    def __init__(self, prompt_id: str, speaker: str):
        if not isinstance(prompt_id, str) or prompt_id not in PROMPTS:
            raise ValueError('Choose a sentence from the list.')
        if not isinstance(speaker, str) or not SPEAKER.fullmatch(speaker):
            raise ValueError('Use a speaker code of 1–16 letters, digits, - or _.')
        self.prompt_id, self.speaker = prompt_id, speaker
        self._reset()

    def metadata(self) -> dict:
        return {'label': 'sentence', 'prompt_id': self.prompt_id, 'speaker': self.speaker,
                'reference': PROMPTS[self.prompt_id], **self.timing()}

    def conflicts(self, saved: dict) -> bool:
        return saved['prompt_id'] == self.prompt_id and saved['speaker'] == self.speaker


class ExampleStore:
    def __init__(self, root: Path = DEFAULT_ROOT, keys: tuple[str, ...] = EXAMPLE_KEYS,
                 max_samples: int | None = None):
        self.root, self.keys = root, keys
        self.max_samples = max_samples

    def _sample_path(self, sample_id: str) -> Path:
        if not SAMPLE_ID.fullmatch(sample_id):
            raise ValueError('Invalid sample identifier.')
        path = self.root / sample_id
        if path.is_symlink():
            raise ValueError('Invalid sample path.')
        return path

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        samples = []
        for path in sorted(self.root.iterdir()):
            if not SAMPLE_ID.fullmatch(path.name) or path.is_symlink() or not path.is_dir():
                continue
            try:
                manifest = json.loads((path / 'manifest.json').read_text())
                samples.append({key: manifest[key] for key in self.keys})
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return samples

    def save(self, buffer: ExampleBuffer) -> dict:
        metadata = buffer.metadata()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        saved = self.list()
        if len(saved) >= (self.max_samples or MAX_SAMPLES):
            raise ValueError('The local sample collection is full. Delete old samples before recording more.')
        if any(buffer.conflicts(item) for item in saved):
            raise ValueError('This speaker already recorded this sentence. Delete it first to record it again.')
        sample_id = uuid.uuid4().hex
        destination = self._sample_path(sample_id)
        staging = Path(tempfile.mkdtemp(prefix='.pending-', dir=self.root))
        manifest = {'version': 1, 'id': sample_id, **metadata,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'width': buffer.dimensions[0], 'height': buffer.dimensions[1],
                    'timestamps_ms': [frame.timestamp for frame in buffer.frames],
                    'frame_files': [f'{index:06d}.jpg' for index in range(len(buffer.frames))]}
        try:
            for frame, name in zip(buffer.frames, manifest['frame_files']):
                self._write_private(staging / name, frame.jpeg)
            self._write_private(staging / 'manifest.json', json.dumps(manifest, indent=2).encode())
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return {key: manifest[key] for key in self.keys}

    @staticmethod
    def _write_private(path: Path, contents: bytes):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as target:
            target.write(contents)

    def delete(self, sample_id: str):
        path = self._sample_path(sample_id)
        if not path.is_dir():
            raise FileNotFoundError(sample_id)
        shutil.rmtree(path)


store = ExampleStore()
sentence_store = ExampleStore(SENTENCE_ROOT, SENTENCE_KEYS, MAX_SENTENCE_SAMPLES)


def create_examples_router(allowed_origin: Callable, get_session_lock: Callable) -> APIRouter:
    router = APIRouter()

    def check_origin(request: Request, required: bool):
        origin = request.headers.get('origin')
        if (required or origin) and not allowed_origin(origin, request.headers.get('host')):
            raise HTTPException(status_code=403, detail='Origin is not allowed.')

    async def delete_from(collection: ExampleStore, sample_id: str, request: Request):
        check_origin(request, required=True)
        try:
            await asyncio.to_thread(collection.delete, sample_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail='Sample was not found.') from exc
        return {'deleted': sample_id}

    @router.get('/api/examples')
    async def list_examples(request: Request):
        check_origin(request, required=False)
        return await asyncio.to_thread(store.list)

    @router.delete('/api/examples/{sample_id}')
    async def delete_example(sample_id: str, request: Request):
        return await delete_from(store, sample_id, request)

    @router.websocket('/api/examples')
    async def capture_example(ws: WebSocket):
        await capture(ws, store, lambda command: ExampleBuffer(command.get('label'), command.get('take')))

    @router.get('/api/eval-sentences')
    async def list_sentences(request: Request):
        check_origin(request, required=False)
        prompts = [{'id': key, 'text': text} for key, text in PROMPTS.items()]
        return {'prompts': prompts, 'recordings': await asyncio.to_thread(sentence_store.list)}

    @router.delete('/api/eval-sentences/{sample_id}')
    async def delete_sentence(sample_id: str, request: Request):
        return await delete_from(sentence_store, sample_id, request)

    @router.websocket('/api/eval-sentences')
    async def capture_sentence(ws: WebSocket):
        await capture(ws, sentence_store, lambda command: SentenceBuffer(command.get('prompt_id'), command.get('speaker')))

    async def capture(ws: WebSocket, collection: ExampleStore, create_buffer: Callable[[dict], ExampleBuffer]):
        if not allowed_origin(ws.headers.get('origin'), ws.headers.get('host')):
            await ws.close(code=1008)
            return
        await ws.accept()
        session_lock = get_session_lock()
        if session_lock.locked():
            await ws.send_json({'type': 'error', 'detail': 'Stop the other reading or recording session first.'})
            await ws.close(code=1013)
            return
        await session_lock.acquire()
        buffer = None
        deadline = time.monotonic() + SESSION_TIMEOUT_SECONDS
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                message = await asyncio.wait_for(ws.receive(), timeout=remaining)
                if message['type'] == 'websocket.disconnect':
                    raise WebSocketDisconnect()
                if message.get('bytes') is not None:
                    if buffer is None:
                        raise ValueError('Start a sample before sending camera frames.')
                    await asyncio.to_thread(buffer.append, message['bytes'])
                    continue
                text = message.get('text', '')
                if len(text) > 1024:
                    raise ValueError('Control message is too large.')
                command = json.loads(text)
                if not isinstance(command, dict):
                    raise ValueError('Invalid sample control message.')
                kind = command.get('type')
                if kind == 'start':
                    if buffer is not None:
                        raise ValueError('This sample has already started.')
                    buffer = create_buffer(command)
                    await ws.send_json({'type': 'ready'})
                elif kind == 'stop':
                    if buffer is None:
                        raise ValueError('Start a sample before saving it.')
                    result = await asyncio.to_thread(collection.save, buffer)
                    await ws.send_json({'type': 'saved', **result})
                    return
                elif kind == 'cancel':
                    await ws.send_json({'type': 'canceled'})
                    return
                else:
                    raise ValueError('Unknown sample command.')
        except WebSocketDisconnect:
            pass
        except asyncio.TimeoutError:
            await ws.send_json({'type': 'error', 'detail': 'Recording timed out. Record this sample again.'})
        except (ValueError, json.JSONDecodeError) as exc:
            await ws.send_json({'type': 'error', 'detail': str(exc)})
        except OSError:
            await ws.send_json({'type': 'error', 'detail': 'The server could not save this sample. Check local disk space and try again.'})
        finally:
            buffer = None
            session_lock.release()
            try:
                await ws.close()
            except (RuntimeError, WebSocketDisconnect):
                pass

    return router

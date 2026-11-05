"""Explicit, local-only recording of labeled camera examples.

Normal reading never calls this module. Frames remain in bounded memory until
the user finishes a sample; only then are the original JPEGs saved locally.
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


class ExampleBuffer:
    def __init__(self, label: str, take: int):
        if not isinstance(label, str) or label not in LABELS:
            raise ValueError('Choose a valid speaking example.')
        if type(take) is not int or take not in (1, 2):
            raise ValueError('Take must be 1 or 2.')
        self.label, self.take = label, take
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
        if self.frames and frame.timestamp - self.frames[0].timestamp > 10_000:
            raise ValueError('Keep each sample between six and ten seconds.')
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

    def metadata(self) -> dict:
        if len(self.frames) < 2:
            raise ValueError('Record a full eight-second sample before saving.')
        duration = (self.frames[-1].timestamp - self.frames[0].timestamp) / 1000
        if not 6 <= duration <= 10:
            raise ValueError('Keep each sample between six and ten seconds.')
        capture_fps = (len(self.frames) - 1) / duration
        if capture_fps < 15:
            raise ValueError('Camera frames arrived too slowly. Improve lighting and record this sample again.')
        return {'label': self.label, 'take': self.take, 'frames': len(self.frames),
                'duration_seconds': round(duration, 3), 'capture_fps': round(capture_fps, 3)}


class ExampleStore:
    def __init__(self, root: Path = DEFAULT_ROOT):
        self.root = root

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
                samples.append({key: manifest[key] for key in
                                ('id', 'label', 'take', 'frames', 'duration_seconds')})
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return samples

    def save(self, buffer: ExampleBuffer) -> dict:
        metadata = buffer.metadata()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        if len(self.list()) >= MAX_SAMPLES:
            raise ValueError('The local sample collection is full. Delete old samples before recording more.')
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
        return {key: manifest[key] for key in ('id', 'label', 'take', 'frames', 'duration_seconds')}

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


def create_examples_router(allowed_origin: Callable, get_session_lock: Callable) -> APIRouter:
    router = APIRouter()

    @router.get('/api/examples')
    async def list_examples(request: Request):
        origin = request.headers.get('origin')
        if origin and not allowed_origin(origin, request.headers.get('host')):
            raise HTTPException(status_code=403, detail='Origin is not allowed.')
        return await asyncio.to_thread(store.list)

    @router.delete('/api/examples/{sample_id}')
    async def delete_example(sample_id: str, request: Request):
        if not allowed_origin(request.headers.get('origin'), request.headers.get('host')):
            raise HTTPException(status_code=403, detail='Origin is not allowed.')
        try:
            await asyncio.to_thread(store.delete, sample_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail='Sample was not found.') from exc
        return {'deleted': sample_id}

    @router.websocket('/api/examples')
    async def capture_example(ws: WebSocket):
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
                    buffer = ExampleBuffer(command.get('label'), command.get('take'))
                    await ws.send_json({'type': 'ready'})
                elif kind == 'stop':
                    if buffer is None:
                        raise ValueError('Start a sample before saving it.')
                    result = await asyncio.to_thread(store.save, buffer)
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

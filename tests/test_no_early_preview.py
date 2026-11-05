"""Stale browser preferences cannot reactivate the removed preview path."""
import asyncio
from fastapi.testclient import TestClient
from backend import app as server
from test_rolling_transport import ORIGIN, send_frames, receive_until


def test_old_preview_request_only_runs_regular_speech_checks(monkeypatch):
    class Model:
        supports_provisional_preview = True  # A stale capability must be ignored.
        def __init__(self):
            self.checks = []
        def load(self):
            pass
        def status(self):
            return {'ready': True}
        def inspect_activity(self, frames, fps, *, cache, speech_gate):
            self.checks.append(speech_gate)
            return {'activity': {'state': 'still', 'suppress': True,
                                 'metrics': {'speech_regions_version': 1}},
                    'timings_ms': {'preprocessing': 1.}}
        def preview_transcribe(self, *args, **kwargs):
            raise AssertionError('Removed preview decoder must never run.')
    model = Model()
    monkeypatch.setattr(server, 'model', model)
    monkeypatch.setattr(server, 'loading', False)
    monkeypatch.setattr(server, 'session_lock', asyncio.Lock())
    with TestClient(server.app) as client:
        assert not client.get('/api/status').json().get('provisional_preview', False)
        with client.websocket_connect('/api/stream', headers=ORIGIN) as ws:
            ws.send_json({'type': 'start', 'provisional_preview': True})
            assert ws.receive_json()['state'] == 'capturing'
            send_frames(ws, 0, 61)
            ws.send_json({'type': 'stop'})
            messages = receive_until(ws, stopped=True)
    assert model.checks and all(model.checks)
    assert not any(message['type'] == 'preview' for message in messages)
    assert all(not message['text'] for message in messages if message['type'] == 'transcript')
    assert not server.session_lock.locked()

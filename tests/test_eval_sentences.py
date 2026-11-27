"""Held-out sentence recordings and word error rate scoring."""
import asyncio
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend import examples
from backend.word_errors import bootstrap_interval, corpus_wer, normalize, score
from scripts import evaluate_sentences
from tests.test_examples import ORIGIN, jpeg, packet


@pytest.fixture
def client(tmp_path, monkeypatch):
    sentences = examples.ExampleStore(tmp_path / 'sentences', examples.SENTENCE_KEYS, 5)
    monkeypatch.setattr(examples, 'sentence_store', sentences)
    monkeypatch.setattr(examples, 'store', examples.ExampleStore(tmp_path / 'samples'))
    lock = asyncio.Lock()
    app = FastAPI()
    app.include_router(examples.create_examples_router(lambda origin, host: origin == ORIGIN['origin'], lambda: lock))
    with TestClient(app) as connection:
        yield connection, sentences, lock


def record(ws, prompt_id='s001', speaker='CF', count=76):
    ws.send_json({'type': 'start', 'prompt_id': prompt_id, 'speaker': speaker})
    assert ws.receive_json() == {'type': 'ready'}
    image = jpeg()
    for index in range(count):
        ws.send_bytes(packet(index * 40, image))
    ws.send_json({'type': 'stop'})
    return ws.receive_json()


def test_prompt_list_is_stable_and_spelled_out():
    prompts = examples.load_prompts()
    assert len(prompts) == 100
    assert list(prompts)[:2] == ['s001', 's002']
    assert not any(character.isdigit() for text in prompts.values() for character in text)


def test_sentence_saved_with_reference_and_listed(client):
    connection, sentences, lock = client
    with connection.websocket_connect('/api/eval-sentences', headers=ORIGIN) as ws:
        saved = record(ws)
    assert saved['type'] == 'saved' and saved['prompt_id'] == 's001' and saved['speaker'] == 'CF'
    assert saved['duration_seconds'] == 3
    assert not lock.locked()
    manifest = json.loads((sentences.root / saved['id'] / 'manifest.json').read_text())
    assert manifest['label'] == 'sentence'
    assert manifest['reference'] == examples.PROMPTS['s001']
    listing = connection.get('/api/eval-sentences').json()
    assert len(listing['prompts']) == 100
    assert listing['recordings'] == [{key: saved[key] for key in examples.SENTENCE_KEYS}]
    # Sentence clips never appear among speaking-filter examples.
    assert connection.get('/api/examples').json() == []


def test_same_speaker_cannot_record_a_sentence_twice_until_deleted(client):
    connection, sentences, _ = client
    with connection.websocket_connect('/api/eval-sentences', headers=ORIGIN) as ws:
        first = record(ws)
    with connection.websocket_connect('/api/eval-sentences', headers=ORIGIN) as ws:
        assert 'already recorded' in record(ws)['detail']
    with connection.websocket_connect('/api/eval-sentences', headers=ORIGIN) as ws:
        assert record(ws, speaker='VR')['type'] == 'saved'
    assert connection.delete(f'/api/eval-sentences/{first["id"]}', headers=ORIGIN).status_code == 200
    with connection.websocket_connect('/api/eval-sentences', headers=ORIGIN) as ws:
        assert record(ws)['type'] == 'saved'


@pytest.mark.parametrize('command', [
    {'prompt_id': 'missing', 'speaker': 'CF'},
    {'prompt_id': 's001', 'speaker': ''},
    {'prompt_id': 's001', 'speaker': '../etc'},
    {'prompt_id': 's001', 'speaker': 'x' * 17},
])
def test_invalid_prompt_or_speaker_rejected(client, command):
    connection, sentences, lock = client
    with connection.websocket_connect('/api/eval-sentences', headers=ORIGIN) as ws:
        ws.send_json({'type': 'start', **command})
        assert ws.receive_json()['type'] == 'error'
    assert not lock.locked() and sentences.list() == []


@pytest.mark.parametrize('count', [30, 380])
def test_sentence_length_bounds(client, count):
    connection, sentences, _ = client
    with connection.websocket_connect('/api/eval-sentences', headers=ORIGIN) as ws:
        result = record(ws, count=count)
    assert result['type'] == 'error' and '15 seconds' in result['detail']
    assert sentences.list() == []


def test_delete_requires_origin(client):
    connection, _, _ = client
    with connection.websocket_connect('/api/eval-sentences', headers=ORIGIN) as ws:
        saved = record(ws)
    assert connection.delete(f'/api/eval-sentences/{saved["id"]}').status_code == 403


def test_normalization_ignores_case_punctuation_and_apostrophe_style():
    assert normalize("I'm going — to the CAR, right?") == ["i'm", 'going', 'to', 'the', 'car', 'right']
    assert normalize('I’m well-known') == ["i'm", 'well', 'known']
    assert score('Could you pass me the salt, please?', 'could you pass me the salt please')['wer'] == 0


def test_error_counts():
    assert score('a b c d', 'a x c d e') == {'reference_words': 4, 'substitutions': 1, 'deletions': 0,
                                            'insertions': 1, 'errors': 2, 'wer': 0.5}
    assert score('a b c', '')['deletions'] == 3
    empty = score('a b', 'a b')
    assert empty['errors'] == 0


def test_corpus_wer_weights_by_words_and_interval_brackets_it():
    rows = [score('a b c d e f g h', 'a b c d e f g h'), score('a b', 'x y')]
    assert corpus_wer(rows) == 0.2
    low, high = bootstrap_interval(rows * 10)
    assert low <= 0.2 <= high
    assert bootstrap_interval(rows[:1]) is None


def test_report_rows_hold_no_text(client, tmp_path):
    connection, sentences, _ = client
    with connection.websocket_connect('/api/eval-sentences', headers=ORIGIN) as ws:
        record(ws)
    records = evaluate_sentences.load_recordings(sentences.root)
    assert len(records) == 1
    directory, manifest = records[0]
    frames = evaluate_sentences.frames_of(directory, manifest)
    assert len(frames) == 76 and frames[0].jpeg == jpeg()
    row = {'id': manifest['id'], 'prompt_id': 's001', 'speaker': 'CF',
           **score(manifest['reference'], 'could you pass the salt'), 'failure': None, 'detail': ''}
    summary = evaluate_sentences.summarize([row])
    assert summary['wer'] == row['wer'] and summary['speakers']['CF']['recordings'] == 1
    assert 'salt' not in json.dumps(summary) + json.dumps({k: v for k, v in row.items() if k != 'detail'})

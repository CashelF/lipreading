"""Repeated language needs evidence; actual repeated speech remains possible."""
from types import SimpleNamespace

import pytest

from backend.repetition import repeated_phrases, repeated_token_spans, repetition_evidence
from backend import grounding

torch = pytest.importorskip('torch')


class CharacterTokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(i) for i in ids if i)

    def __call__(self, text, **kwargs):
        return {'input_ids': list(map(ord, text)), 'offset_mapping': [(i, i + 1) for i in range(len(text))]}


def test_reported_multi_phrase_triplication_is_found_without_rewriting():
    phrases = ['what does our mouth do', 'interesting that i decided to finalise those words']
    text = ' '.join(' '.join([phrase] * 3) for phrase in phrases)
    runs = repeated_phrases(text)
    assert [(run['words_per_copy'], run['copies']) for run in runs] == [(5, 3), (8, 3)]
    for phrase, run in zip(phrases, runs):
        assert [text[a:b] for a, b in run['char_spans']] == [phrase] * 3


@pytest.mark.parametrize('text', ['no no no no no no', 'thank you thank you thank you thank you',
                                 'I know that you know that we know', 'very very good, very good'])
def test_short_repeated_words_and_nonmatching_phrases_are_not_loop_candidates(text):
    assert repeated_phrases(text) == []


def test_case_punctuation_and_unicode_words_map_to_exact_generated_tokens():
    text = 'Ça va bien. ÇA va bien! Ça va bien?'
    tokenizer = CharacterTokenizer()
    runs = repeated_token_spans(tokenizer, list(map(ord, text)))
    assert len(runs) == 1 and runs[0]['copies'] == 3
    assert [tokenizer.decode([ord(text[i]) for i in indices]) for indices in runs[0]['token_spans']] == [
        'Ça va bien', 'ÇA va bien', 'Ça va bien']


def test_changed_tokenization_cannot_guess_evidence_boundaries():
    class NoncanonicalTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            result['input_ids'][0] += 1
            return result
    assert repeated_token_spans(NoncanonicalTokenizer(), list(map(ord, 'one two three one two three'))) == []


@pytest.mark.parametrize('lifts,accepted', [
    ([2.237, .132, .146], False),  # Single speech video, copied text.
    ([1.414, .934, .350], True),   # Same video actually repeated three times.
    ([.816, .771, .157], True),    # Subtle repeated speech: absolute .2 would fail.
    ([2., .3, .4], True),
    ([.5, .06, .07], True),        # Small first-copy support: use relative evidence too.
])
def test_each_extra_copy_needs_support_but_repetition_itself_is_allowed(lifts, accepted):
    runs = [{'words_per_copy': 5, 'copies': 3, 'token_spans': [[0], [1], [2]]}]
    result = repetition_evidence(runs, torch.tensor(lifts), {'first': torch.zeros(3), 'last': torch.ones(3)})
    assert result['accepted'] is accepted
    assert set(result['runs'][0]) == {'words_per_copy', 'copies', 'max_lift_per_copy', 'support_floor', 'unsupported_copy_indices'}


def fake_scoring(monkeypatch):
    """Strong support for the first phrase, negligible evidence for later copies."""
    tokenizer = CharacterTokenizer()
    model = SimpleNamespace(llm_tokenizer=tokenizer)
    prefixes = []
    monkeypatch.setattr(grounding, 'control_sample', lambda sample, name: name)
    def visual_prefix(model, sample):
        prefixes.append(sample)
        return 'frozen'
    monkeypatch.setattr(grounding, 'visual_prefix', visual_prefix)
    def score(model, prefix, tokens):
        values = torch.full((len(tokens),), -3.)
        if prefix == 'real':
            content = tokens[:-1].tolist()
            text = tokenizer.decode(content)
            values[:] = -1.
            if text.startswith('unrelated'):
                values[:] = -3.1  # Fails the whole-phrase check.
            for run in repeated_token_spans(tokenizer, content):
                for indices in run['token_spans'][1:]:
                    values[indices] = -2.99
        return values
    monkeypatch.setattr(grounding, 'token_log_probabilities', score)
    def candidate(text, alternatives=()):
        return grounding.Candidate('real', torch.tensor(list(map(ord, text)) + [0]), -.4, True, alternatives)
    return model, candidate, prefixes


def test_unsupported_loop_uses_an_existing_alternative_only_after_all_checks(monkeypatch):
    model, candidate, prefixes = fake_scoring(monkeypatch)
    good = candidate('one two three')
    loop = candidate('one two three one two three one two three',
                     (candidate('unrelated wrong phrase'), good))
    result = grounding.verify_candidate(model, {}, loop)
    assert result['accepted'] and result['alternative_candidates_scored'] == 2
    assert torch.equal(result['_replacement_tokens'], good.tokens)
    assert not result['original_repetition']['accepted']
    assert len(prefixes) == 3  # Encoders are reused for alternatives.


def test_failed_alternatives_abstain_with_a_bounded_workload(monkeypatch):
    model, candidate, _ = fake_scoring(monkeypatch)
    alternatives = tuple(candidate('unrelated wrong phrase ' + str(i)) for i in range(8))
    result = grounding.verify_candidate(model, {}, candidate('one two three one two three', alternatives))
    assert not result['accepted'] and result['state'] == 'repetition_withheld'
    assert result['alternative_candidates_scored'] == grounding.MAX_REPEAT_ALTERNATIVES
    assert '_replacement_tokens' not in result


def test_weak_whole_phrase_does_not_fish_for_an_alternative(monkeypatch):
    model, candidate, _ = fake_scoring(monkeypatch)
    result = grounding.verify_candidate(model, {}, candidate('unrelated wrong phrase', (candidate('one two three'),)))
    assert not result['accepted'] and result['alternative_candidates_scored'] == 0


def test_duplicate_beam_candidates_do_not_consume_retry_budget(monkeypatch):
    model, candidate, _ = fake_scoring(monkeypatch)
    text = 'one two three one two three'
    good = candidate('one two three')
    result = grounding.verify_candidate(model, {}, candidate(text, (candidate(text), candidate(text), good)))
    assert result['accepted'] and result['alternative_candidates_scored'] == 1

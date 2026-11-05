"""Causal scoring and isolated instrumentation; no recordings/model required."""
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')

from backend.grounding import (capture_candidate, control_sample, summarize_grounding,
                               token_log_probabilities, grounding_decision, FROZEN_CONTROLS)


def test_teacher_forcing_scores_exact_next_tokens_including_eos():
    vocab = 6
    table = torch.arange(vocab * vocab, dtype=torch.float32).reshape(vocab, vocab) / 11
    class Qwen:
        def embed_tokens(self, ids):
            return torch.nn.functional.one_hot(ids, vocab).float()
        def __call__(self, inputs_embeds, **kwargs):
            return SimpleNamespace(last_hidden_state=inputs_embeds @ table)
    qwen = Qwen()
    model = SimpleNamespace(use_lora=False, llm=SimpleNamespace(model=qwen, lm_head=lambda x: x))
    prefix = qwen.embed_tokens(torch.tensor([0, 1])).unsqueeze(0)
    tokens = torch.tensor([3, 4, 2])
    actual = token_log_probabilities(model, prefix, tokens)
    expected = torch.log_softmax(table[torch.tensor([1, 3, 4])] / 1.2, dim=-1)
    assert torch.equal(actual, expected.gather(-1, tokens[:, None]).squeeze(-1))


@pytest.mark.parametrize('kind', ['freeze_first', 'freeze_middle', 'freeze_last', 'shuffle'])
def test_controls_keep_face_lip_pairing_length_and_original_data(kind):
    face = torch.arange(9.).reshape(1, 1, 9, 1, 1)
    original = face.clone()
    sample = {'net_input': {'source': {'video': (face, face + 100), 'audio': None}, 'padding_mask': 'same'}}
    result = control_sample(sample, kind)
    altered_face, altered_lips = result['net_input']['source']['video']
    assert torch.equal(face, original)
    assert torch.equal(altered_lips - altered_face, torch.full_like(face, 100))
    assert altered_face.shape == face.shape
    assert result['net_input']['padding_mask'] == 'same'
    if kind == 'shuffle':
        assert not torch.equal(altered_face, face)
        assert torch.equal(altered_face.flatten().sort().values, face.flatten())
    else:
        assert len(altered_face.unique()) == 1


@pytest.mark.parametrize('raise_error', [False, True])
def test_capture_restores_search_and_preserves_exact_candidate(raise_error):
    prefix = torch.ones(1, 2, 4)
    selected = torch.tensor([0, 0, 3, 2])
    result = (selected, torch.tensor(-.25), [selected], [torch.tensor(-.25)])
    def beam(*args, **kwargs):
        if raise_error:
            raise RuntimeError('controlled failure')
        return result
    namespace = {'beam_search': beam}
    exec('def search(ids, prefix):\n    return beam_search(ids, prefix, None, EOS_TOKEN=2)', namespace)
    search = namespace['search']
    with capture_candidate(search) as observed:
        if raise_error:
            with pytest.raises(RuntimeError):
                search(torch.zeros(1, 2), prefix)
        else:
            assert search(torch.zeros(1, 2), prefix) is result
            assert torch.equal(observed[0].tokens, torch.tensor([3, 2]))
            assert torch.equal(observed[0].prefix, prefix)
            assert observed[0].prefix.data_ptr() == prefix.data_ptr()
            assert observed[0].ended
    assert namespace['beam_search'] is beam


def test_summary_excludes_eos_and_does_not_report_tokens():
    result = summarize_grounding(torch.tensor([-1., -2., -100.]),
                                 {'frozen': torch.tensor([-2., -4., 0.])}, ended=True)
    assert result['real_mean_logp'] == -1.5
    assert result['controls']['frozen']['mean_lift'] == 1.5
    assert result['content_tokens'] == 2
    assert result['finite'] and result['usable']
    assert not summarize_grounding(torch.tensor([0.]), {'frozen': torch.tensor([0.])}, ended=True)['usable']


@pytest.mark.parametrize('lifts, accepted', [([-1., -.1, .09], False),
                                           ([0., 0., .1], True), ([2., 3., 4.], True)])
def test_decision_requires_all_frozen_controls_to_agree_before_withholding(lifts, accepted):
    summary = {'usable': True, 'finite': True,
               'controls': {name: {'mean_lift': lift} for name, lift in zip(FROZEN_CONTROLS, lifts)}}
    assert grounding_decision(summary)['accepted'] is accepted
    summary['finite'] = False
    assert not grounding_decision(summary)['accepted']

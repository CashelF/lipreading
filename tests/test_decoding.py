"""Exact decoder equivalence; no model weights or customer video are needed."""
from pathlib import Path
from types import SimpleNamespace
import logging

import pytest

torch = pytest.importorskip('torch')

from backend.decoding import _decode_results, build_search

SOURCE = Path(__file__).resolve().parents[1] / 'vendor/FLARE/flare_llm/decode_scripts/infer_s2s_beamsearch_s3_chime_llm.py'
pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason='Pinned FLARE source is needed for decoder comparison')
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


class TinyQwen:
    """Deterministic autoregressive logits whose values depend on reordered KV state."""
    def __init__(self, policy, device, vocab=8):
        self.vocab = vocab
        self.device = device
        self.policy = policy
        self.step = 0
        self.cache_snapshots = []
        generator = torch.Generator().manual_seed(90210)
        self.table = (torch.rand((64, vocab), generator=generator) * 5).to(device)
        if policy == 'no_eos':
            self.table[:, 2] = -1000
        elif policy == 'eos':
            self.table[:, 2] = 12
        elif policy == 'mixed':
            self.table[:, 2] = torch.arange(64, device=device) % 3 * 4
        elif policy == 'ties':
            self.table[:] = 0

    def embed_tokens(self, tokens):
        return self.table[0].expand(len(tokens), -1).clone()

    def __call__(self, inputs_embeds=None, input_ids=None, past_key_values=None, **kwargs):
        self.step += 1
        if inputs_embeds is not None:
            batch, length, _ = inputs_embeds.shape
            values = inputs_embeds
            key = torch.arange(length, dtype=torch.float32, device=self.device)[None, None, :, None]
            key = key.expand(batch, 1, length, 1).clone()
            cache = ((key, key + 10), (key + 20, key + 30))
        else:
            self.cache_snapshots.append(tuple((p[0].clone(), p[1].clone()) for p in past_key_values))
            history = past_key_values[0][0].sum(dim=(1, 2, 3)).long()
            index = (input_ids[:, -1] + history + self.step) % len(self.table)
            values = self.table[index][:, None, :]
            extension = input_ids[:, None, -1:, None].float()
            cache = tuple((torch.cat((k, extension), dim=2), torch.cat((v, extension + 10), dim=2))
                          for k, v in past_key_values)
        return SimpleNamespace(last_hidden_state=values, past_key_values=cache)


class TinyLM:
    def __init__(self, policy, device):
        self.model = TinyQwen(policy, device)
        self.head_shapes = []

    def lm_head(self, hidden):
        self.head_shapes.append(tuple(hidden.shape))
        return hidden

    def prepare_inputs_for_generation(self, input_ids, past_key_values, use_cache):
        return {'input_ids': input_ids[:, -1:], 'past_key_values': past_key_values, 'use_cache': use_cache}


def run_beam(optimized, policy, beam, device, use_lora):
    search = build_search(SOURCE, logging.getLogger(__name__), optimized)
    lm = TinyLM(policy, device)
    model = SimpleNamespace(use_lora=use_lora, llm=SimpleNamespace(model=lm) if use_lora else lm)
    hidden = lm.model.table[0][None, None, :].repeat(1, 4, 1)
    initial = torch.tensor([[0, 0, 4, 5]], device=device)
    result = search.__globals__['beam_search'](initial, hidden, model, max_length=5,
            real_beam_size=beam, vocab_size=8, EOS_TOKEN=2, temp=1.2)
    return result, lm


def assert_result_equal(reference, optimized):
    assert torch.equal(reference[0], optimized[0])
    assert torch.equal(reference[1], optimized[1])
    assert len(reference[2]) == len(optimized[2])
    assert len(reference[3]) == len(optimized[3])
    for left, right in zip(reference[2], optimized[2]):
        assert torch.equal(left, right)
    for left, right in zip(reference[3], optimized[3]):
        assert torch.equal(left, right)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('policy', ['no_eos', 'eos', 'mixed', 'ties'])
@pytest.mark.parametrize('beam', [1, 3, 6])
@pytest.mark.parametrize('use_lora', [False, True])
def test_beam_tokens_scores_and_reordered_caches_are_exact(device, policy, beam, use_lora):
    reference, reference_lm = run_beam(False, policy, beam, device, use_lora)
    optimized, optimized_lm = run_beam(True, policy, beam, device, use_lora)
    assert_result_equal(reference, optimized)
    # Including prefill positions ensures no GEMM shape/rounding shortcut was added.
    assert reference_lm.head_shapes == optimized_lm.head_shapes
    assert reference_lm.head_shapes[0] == (1, 4, 8)
    assert len(reference_lm.model.cache_snapshots) == len(optimized_lm.model.cache_snapshots)
    for left, right in zip(reference_lm.model.cache_snapshots, optimized_lm.model.cache_snapshots):
        for lp, rp in zip(left, right):
            assert torch.equal(lp[0], rp[0])
            assert torch.equal(lp[1], rp[1])


class Tokenizer:
    eos_token_id = 2

    def encode(self, text):
        return [4] if text == 'A' else [5, 6]

    def decode(self, tokens, skip_special_tokens=True):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        return '/'.join(map(str, tokens))


@pytest.mark.parametrize('device', DEVICES)
def test_bulk_hypothesis_transfer_preserves_variable_lengths_and_scores(device):
    sequences = [torch.tensor([0, 0, 7, 2], device=device), torch.tensor([0, 0, 3, 4, 2], device=device), torch.tensor([0, 0], device=device)]
    scores = [torch.tensor(value, dtype=torch.float32, device=device) for value in [-1.1, -0.2, -3.]]
    tokenizer = Tokenizer()
    best = sequences[1]
    expected = (tokenizer.decode(best[2:]), [(tokenizer.decode(seq[2:]), score.item()) for seq, score in zip(sequences, scores)])
    assert _decode_results(torch, tokenizer, best, sequences, scores, 2) == expected


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('policy', ['no_eos', 'eos', 'mixed', 'ties'])
def test_complete_search_hypotheses_and_scores_are_exact(device, policy):
    def run(optimized):
        search = build_search(SOURCE, logging.getLogger(__name__), optimized)
        beam = search.__globals__['beam_search']
        # The real search hardcodes the release's 152064-token vocabulary. Use
        # the fixture's eight-token vocabulary while exercising its entire flow.
        def tiny_vocabulary_beam(*args, **kwargs):
            kwargs['vocab_size'], kwargs['EOS_TOKEN'] = 8, 2
            return beam(*args, **kwargs)
        search.__globals__['beam_search'] = tiny_vocabulary_beam
        lm = TinyLM(policy, device)
        model = SimpleNamespace(
            use_lora=True, llm=SimpleNamespace(model=lm), llm_tokenizer=Tokenizer(),
            cfg=SimpleNamespace(downsample=False), prompt_str='transcribe the video:',
            parameters=lambda: iter([torch.empty(0, device=device)]),
            modality_projection=lambda value: value,
            encoder=SimpleNamespace(forward=lambda **kwargs: {
                'encoder_out': lm.model.table[0][None, None, :].repeat(4, 1, 1),
                'encoder_padding_mask': torch.zeros((1, 4), device=device, dtype=torch.bool),
            }),
        )
        return search([model], {'net_input': {}}, beam_size=3, audio_only=False, video_only=True, temp=1.2)
    assert run(False) == run(True)


def test_pinned_source_change_fails_closed(tmp_path):
    changed = tmp_path / 'changed.py'
    changed.write_text(SOURCE.read_text().replace('next_word != EOS_TOKEN', 'next_word > EOS_TOKEN'))
    with pytest.raises(RuntimeError, match='unchecked optimization'):
        build_search(changed, logging.getLogger(__name__), True)

"""Experimental visual-evidence diagnostics; never a calibrated confidence score.

Beam search remains unchanged. A separately selectable trial rule withholds
whole unsupported hypotheses; its score is not a correctness probability.
Candidate IDs and video-derived tensors are transient and must not be logged.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

FROZEN_CONTROLS = ('freeze_first', 'freeze_middle', 'freeze_last')
MINIMUM_VISUAL_LIFT = .1  # Trial nats/content-token; not a probability threshold.
MAX_REPEAT_ALTERNATIVES = 3


@dataclass
class Candidate:
    prefix: object
    tokens: object
    beam_score: float
    ended: bool
    alternatives: tuple[Candidate, ...] = ()


@contextmanager
def capture_candidate(search):
    """Observe an isolated search instance; caller must serialize its use."""
    original = search.__globals__['beam_search']
    captured = []

    def observe(fake_ids, prefix, model, **kwargs):
        result = original(fake_ids, prefix, model, **kwargs)
        tokens = result[0][fake_ids.shape[1]:].detach().clone()
        import torch
        scores = torch.stack([score.reshape(()) for score in result[3]]).tolist()
        final_tokens = torch.stack([sequence[-1] for sequence in result[2]]).tolist()
        alternatives = []
        for sequence, score, last_token in zip(result[2], scores, final_tokens):
            suffix = sequence[fake_ids.shape[1]:].detach()
            alternatives.append(Candidate(prefix.detach(), suffix, score,
                                          bool(suffix.numel() and last_token == kwargs['EOS_TOKEN'])))
        captured.append(Candidate(prefix.detach(), tokens, float(result[1].reshape(())),
                                  bool(tokens.numel() and tokens[-1].item() == kwargs['EOS_TOKEN']),
                                  tuple(sorted(alternatives, key=lambda candidate: -candidate.beam_score))))
        return result

    search.__globals__['beam_search'] = observe
    try:
        yield captured
    finally:
        search.__globals__['beam_search'] = original


def qwen_parts(model):
    lm = model.llm.model if model.use_lora else model.llm
    return lm.model, lm


def visual_prefix(model, sample):
    """Match the released single-video inference prompt/visual projection."""
    import torch
    import torch.nn.functional as F
    qwen, _ = qwen_parts(model)
    encoded = model.encoder.forward(**sample['net_input'])
    hidden, mask = encoded['encoder_out'], encoded['encoder_padding_mask']
    if hidden.shape[1] != 1:
        raise ValueError('Grounding diagnostics require one video at a time.')
    lengths = (~mask).long().sum(dim=1)
    if model.cfg.downsample:
        hidden = F.avg_pool1d(hidden.permute(1, 2, 0), 2, 2).permute(2, 0, 1)
        lengths = torch.ceil(lengths.float() / 2.).long()
    projected = model.modality_projection(hidden)[:lengths[0], 0, :]
    device = projected.device
    leading = qwen.embed_tokens(torch.tensor(model.llm_tokenizer.encode('A'), device=device))
    prompt_ids = model.llm_tokenizer.encode(model.prompt_str) + [model.llm_tokenizer.eos_token_id]
    prompt = qwen.embed_tokens(torch.tensor(prompt_ids, device=device))
    return torch.cat((leading, projected, prompt)).to(torch.bfloat16).unsqueeze(0)


def token_log_probabilities(model, prefix, tokens, temperature=1.2):
    """Teacher-force exact generated IDs, including EOS, with causal alignment.

This is a diagnostic rescore, not bitwise equivalent to autoregressive BF16
search. All controls use identical token sequences, shapes and temperature.
"""
    import torch
    if tokens.ndim != 1 or not tokens.numel():
        raise ValueError('A nonempty one-dimensional candidate is required.')
    if temperature <= 0:
        raise ValueError('Temperature must be positive.')
    qwen, lm = qwen_parts(model)
    suffix = qwen.embed_tokens(tokens[:-1]).unsqueeze(0)
    inputs = torch.cat((prefix, suffix), dim=1)
    output = qwen(inputs_embeds=inputs, use_cache=False, return_dict=True)
    # The final prefix position predicts token0, then token i predicts token i+1.
    predicting = output.last_hidden_state[:, prefix.shape[1] - 1:, :]
    logits = lm.lm_head(predicting).float() / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[0].gather(-1, tokens[:, None]).squeeze(-1)


def control_sample(sample, kind):
    """Remove temporal information in both aligned visual streams together."""
    import torch
    source = sample['net_input']['source']
    videos = source['video']
    length = videos[0].shape[2]
    if length < 2 or any(video.shape[2] != length for video in videos):
        raise ValueError('Matching multi-frame face/lip streams are required.')
    if kind == 'shuffle':
        indices = torch.randperm(length, generator=torch.Generator().manual_seed(731))
        changed = tuple(video.index_select(2, indices.to(video.device)) for video in videos)
    elif kind in ('freeze_first', 'freeze_middle', 'freeze_last'):
        index = {'freeze_first': 0, 'freeze_middle': length // 2, 'freeze_last': length - 1}[kind]
        changed = tuple(video[:, :, index:index + 1].expand_as(video).clone() for video in videos)
    else:
        raise ValueError('Unknown visual control.')
    return {**sample, 'net_input': {**sample['net_input'],
            'source': {**source, 'video': changed}}}


def summarize_grounding(real, controls, *, ended):
    """Return numeric diagnostics only; never serialize tokens or caption text."""
    import torch
    count = len(real) - int(ended)
    if count <= 0:
        return {'content_tokens': 0, 'eos': ended, 'controls': {}, 'usable': False}
    values = real[:count].float()
    result = {'content_tokens': count, 'eos': ended, 'usable': True,
              'real_mean_logp': float(values.mean()), 'controls': {}}
    for name, scores in controls.items():
        if scores.shape != real.shape:
            raise ValueError('Every control must score the same candidate tokens.')
        lift = values - scores[:count].float()
        result['controls'][name] = {
            'mean_lift': float(lift.mean()),
            'last_three_lift': float(lift[-3:].mean()),
            'min_token_lift': float(lift.min()),
            'positive_token_fraction': float((lift > 0).float().mean()),
        }
    result['finite'] = bool(torch.isfinite(real).all() and all(torch.isfinite(x).all() for x in controls.values()))
    return result


def grounding_decision(summary):
    """Conservative phrase-level abstention; never deletes individual words."""
    if not summary.get('usable') or not summary.get('finite'):
        return {'accepted': False, 'state': 'unavailable',
                'detail': 'Could not verify these words against the video.'}
    if any(name not in summary['controls'] for name in FROZEN_CONTROLS):
        raise ValueError('All three frozen controls are required.')
    # Reject only when ALL controls show little benefit from actual lip motion.
    lift = max(summary['controls'][name]['mean_lift'] for name in FROZEN_CONTROLS)
    accepted = lift >= MINIMUM_VISUAL_LIFT
    return {'accepted': accepted, 'state': 'supported' if accepted else 'withheld',
            'maximum_visual_lift': lift, 'minimum_visual_lift': MINIMUM_VISUAL_LIFT,
            'detail': 'Visual evidence check passed.' if accepted else
                      'Waiting for words that are better supported by your lip movements.'}


def verify_candidate(model, sample, candidate):
    """Check each repeated portion; try bounded existing alternatives if needed.

The original beam search and its candidate order are unchanged. A failing
whole-phrase check still abstains immediately. Only an unsupported repetition
permits testing up to three different existing candidates with the same checks.
"""
    from .repetition import repeated_token_spans, repetition_evidence
    prefixes = {name: visual_prefix(model, control_sample(sample, name)) for name in FROZEN_CONTROLS}

    def assess(proposal):
        real = token_log_probabilities(model, proposal.prefix, proposal.tokens)
        controls = {name: token_log_probabilities(model, prefix, proposal.tokens) for name, prefix in prefixes.items()}
        summary = summarize_grounding(real, controls, ended=proposal.ended)
        decision = grounding_decision(summary)
        content = proposal.tokens[:-1] if proposal.ended else proposal.tokens
        spans = repeated_token_spans(model.llm_tokenizer, content.tolist())
        repetition = repetition_evidence(spans, real, controls) if summary.get('finite') else {'accepted': False, 'runs': []}
        if decision['accepted'] and not repetition['accepted']:
            decision.update(accepted=False, state='repetition_withheld',
                            detail='Waiting for clearer evidence for the repeated words.')
        return {**decision, 'diagnostics': summary, 'repetition': repetition}

    result = assess(candidate)
    result['alternative_candidates_scored'] = 0
    if result['state'] != 'repetition_withheld':
        return result
    seen = {tuple(candidate.tokens.tolist())}
    for alternative in candidate.alternatives:
        key = tuple(alternative.tokens.tolist())
        if key in seen:
            continue
        seen.add(key)
        alternate = assess(alternative)
        result['alternative_candidates_scored'] += 1
        if alternate['accepted']:
            alternate.update(alternative_candidates_scored=result['alternative_candidates_scored'],
                             original_repetition=result['repetition'],
                             _replacement_tokens=alternative.tokens)
            return alternate
        if result['alternative_candidates_scored'] >= MAX_REPEAT_ALTERNATIVES:
            break
    return result

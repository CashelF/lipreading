"""Experimental greedy/speculative decoder, never the trusted beam-search path."""
from __future__ import annotations
import time


def merged_preview_model(model):
    """Share the visual encoder, independently copy and merge the LLM."""
    import copy
    candidate = copy.copy(model)
    # nn.Module's registry must be copied too: assigning a module through a
    # shallow-copy registry would silently replace the trusted model's LLM.
    candidate._modules = model._modules.copy()
    candidate.llm = copy.deepcopy(model.llm)
    candidate.llm = candidate.llm.merge_and_unload()
    candidate.use_lora = False
    return candidate


class PreviewTimeout(RuntimeError):
    pass


def greedy_preview(model, prefix, *, draft=(), block_size=4, max_tokens=64, deadline=None):
    """Verify draft tokens against this video's target logits, then crop the KV.

    In real arithmetic this implements greedy target decoding, not beam search.
    BF16 multi-token kernels can round differently; benchmark agreement explicitly.
    A previous preview supplies proposals only, never unverified output or KV.
    """
    import torch
    from backend.grounding import qwen_parts, Candidate
    qwen, lm = qwen_parts(model)
    eos = model.llm_tokenizer.eos_token_id
    generated, logps = [], []
    calls = accepted = proposed = 0
    def check():
        if deadline is not None and time.perf_counter() >= deadline:
            raise PreviewTimeout('Preview budget exhausted')
    def forward(**kwargs):
        nonlocal calls
        check()
        result = qwen(**kwargs, use_cache=True, return_dict=True)
        calls += 1
        check()
        return result
    output = forward(inputs_embeds=prefix)
    logits = lm.lm_head(output.last_hidden_state[:, -1:, :]).float() / 1.2
    first = int(logits[0, 0].argmax())
    generated.append(first)
    logps.append(torch.log_softmax(logits[0, 0], -1)[first])
    cache = output.past_key_values
    while generated[-1] != eos and len(generated) < max_tokens:
        remaining = max_tokens - len(generated)
        proposals = list(draft[len(generated):len(generated)+min(block_size, remaining-1)])
        proposed += len(proposals)
        ids = torch.tensor([[generated[-1], *proposals]], device=prefix.device)
        old_length = cache[0][0].shape[2]
        output = forward(input_ids=ids, past_key_values=cache)
        logits = lm.lm_head(output.last_hidden_state).float() / 1.2
        predicted = logits.argmax(-1)[0].tolist()
        scores = torch.log_softmax(logits[0], -1)
        matched = 0
        for candidate, expected in zip(proposals, predicted):
            if candidate != expected:
                break
            generated.append(candidate)
            logps.append(scores[matched, candidate])
            matched += 1
            accepted += 1
            if candidate == eos:
                break
        if generated[-1] != eos:
            token = predicted[matched]
            generated.append(token)
            logps.append(scores[matched, token])
        keep = old_length + 1 + matched
        cache = tuple((key[:, :, :keep, :].contiguous(), value[:, :, :keep, :].contiguous())
                      for key, value in output.past_key_values)
    check()
    tokens = torch.tensor(generated, device=prefix.device)
    candidate = Candidate(prefix.detach(), tokens, float(torch.stack(logps).mean()), generated[-1] == eos)
    return candidate, {'forward_calls': calls, 'draft_proposed': proposed, 'draft_accepted': accepted,
                       'tokens': len(generated), 'ended': generated[-1] == eos}

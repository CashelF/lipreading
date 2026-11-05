"""Load the released decoder with exact-value reductions in host/copy overhead.

The optimized path keeps the same floating-point operations, their tensor shapes,
beam ordering, search parameters, and stopping rule. In particular, it deliberately
keeps the full-position lm_head: slicing before GEMM can change BF16 rounding.
"""
from __future__ import annotations

import ast
from pathlib import Path


def _replace_once(source: str, original: str, replacement: str) -> str:
    if source.count(original) != 1:
        raise RuntimeError('Pinned FLARE decoder changed; refusing an unchecked optimization.')
    return source.replace(original, replacement, 1)


def _decode_results(torch, tokenizer, best, sequences, scores, start):
    """Move generated IDs/scores to the CPU in two transfers, preserving order."""
    suffixes = [best[start:], *(sequence[start:] for sequence in sequences)]
    lengths = [suffix.numel() for suffix in suffixes]
    token_ids = torch.cat(suffixes).tolist()
    score_values = torch.stack([score.reshape(()) for score in scores]).tolist()
    decoded = []
    offset = 0
    for length in lengths:
        decoded.append(tokenizer.decode(token_ids[offset:offset + length], skip_special_tokens=True))
        offset += length
    return decoded[0], list(zip(decoded[1:], score_values))


def build_search(path: str | Path, logger, optimized: bool = True):
    """Compile only beam_search/search; False is the pre-optimization oracle."""
    import torch

    path = Path(path)
    source = path.read_text()
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('beam_search', 'search')]
    if len(functions) != 2:
        raise RuntimeError('Expected the two released FLARE decoding functions.')
    source = '\n\n'.join(ast.get_source_segment(source, node) for node in functions)
    # Existing compatibility fixes, applied identically to both paths.
    source = _replace_once(source,
        'return fake_input_ids[0], top_k_scores[0], fake_input_ids',
        'return fake_input_ids[0], top_k_scores[0], list(fake_input_ids), list(top_k_scores.reshape(-1))')
    source = _replace_once(source, 'device="cuda"', 'device=next(model.parameters()).device')
    if optimized:
        # A CPU integer list has the exact same token values and iteration order;
        # this replaces one GPU synchronization per candidate with one per step.
        source = _replace_once(source,
            'incomplete_inds = [ind for ind, next_word in enumerate(top_k_words) if\n                                next_word != EOS_TOKEN]',
            'incomplete_inds = [ind for ind, next_word in enumerate(top_k_words.tolist()) if\n                                next_word != EOS_TOKEN]')
        # Keep the original set-complement expression, including its tie ordering.
        # Materialize the remaining-beam index once, rather than once per KV layer.
        source = _replace_once(source,
            '        if step == 1:\n\n            presents = outputs.past_key_values',
            '        incomplete_index = torch.tensor(incomplete_inds, dtype=torch.long, device=beam_id.device)\n'
            '        if step == 1:\n\n            presents = outputs.past_key_values')
        source = _replace_once(source,
            'new_p = (p[0].expand(real_beam_size, -1, -1, -1)[incomplete_inds], \n                        p[1].expand(real_beam_size, -1, -1, -1)[incomplete_inds])',
            'new_p = (p[0].expand(real_beam_size, -1, -1, -1)[incomplete_index], \n                        p[1].expand(real_beam_size, -1, -1, -1)[incomplete_index])')
        source = _replace_once(source,
            '        else:\n            presents = outputs.past_key_values # .to_legacy_cache()\n            new_presents = ()',
            '        else:\n            presents = outputs.past_key_values # .to_legacy_cache()\n'
            '            selected_beams = beam_id[incomplete_index]\n            new_presents = ()')
        source = _replace_once(source,
            'new_p = (p[0][beam_id, :, :, :][incomplete_inds], \n                        p[1][beam_id, :, :, :][incomplete_inds])',
            'new_p = (p[0].index_select(0, selected_beams), \n                        p[1].index_select(0, selected_beams))')
        # Single-axis gathers have specialized kernels; they copy identical
        # values/layout without changing any floating-point model operation.
        source = _replace_once(source,
            'fake_input_ids = torch.cat([fake_input_ids[beam_id], top_k_words.unsqueeze(1)], dim=1)',
            'fake_input_ids = torch.cat([fake_input_ids.index_select(0, beam_id), top_k_words.unsqueeze(1)], dim=1)')
        source = _replace_once(source,
            '        if len(complete_inds) > 0:\n            complete_seqs.extend(fake_input_ids[complete_inds])',
            '        if len(complete_inds) > 0:\n'
            '            complete_index = torch.tensor(complete_inds, dtype=torch.long, device=beam_id.device)\n'
            '            complete_seqs.extend(fake_input_ids.index_select(0, complete_index))')
        source = _replace_once(source,
            'complete_seqs_scores.extend(top_k_scores[complete_inds] / step)',
            'complete_seqs_scores.extend(top_k_scores.index_select(0, complete_index) / step)')
        # prepare_inputs_for_generation in the pinned Qwen implementation only
        # returns views; it does not mutate these IDs. One gather can be reused.
        source = _replace_once(source,
            '        model_inputs = QWENLM.prepare_inputs_for_generation(fake_input_ids[incomplete_inds],',
            '        active_ids = fake_input_ids[incomplete_index]\n'
            '        model_inputs = QWENLM.prepare_inputs_for_generation(active_ids,')
        source = _replace_once(source,
            '        final_top_k_scores = top_k_scores[incomplete_inds]\n'
            '        fake_input_ids = fake_input_ids[incomplete_inds]\n'
            '        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)',
            '        final_top_k_scores = top_k_scores[incomplete_index]\n'
            '        fake_input_ids = active_ids\n'
            '        top_k_scores = final_top_k_scores.unsqueeze(1)')
        # Preserve Python max/index behavior, including first-equal-score ties.
        # Scores are already float32; transferring to Python floats is lossless.
        source = _replace_once(source,
            '    i = complete_seqs_scores.index(max(complete_seqs_scores))',
            '    host_scores = torch.stack(complete_seqs_scores).tolist()\n'
            '    i = host_scores.index(max(host_scores))')
        source = _replace_once(source,
            '            fake_input_ids_filter = res[len(speech_embedding)+len(gmask_sop):]\n'
            '            decode_string = model.llm_tokenizer.decode(fake_input_ids_filter, skip_special_tokens=True)\n'
            '            hypos.append(decode_string)\n\n'
            '            n_best = []\n'
            '            for ii, b_res in enumerate(all_beam_res):\n'
            '                b_hypo = model.llm_tokenizer.decode(b_res[len(speech_embedding)+len(gmask_sop):], skip_special_tokens=True)\n'
            '                n_best.append((b_hypo, all_beam_res_scores[ii].item()))',
            '            decode_string, n_best = _decode_results(torch, model.llm_tokenizer, res,\n'
            '                all_beam_res, all_beam_res_scores, len(speech_embedding)+len(gmask_sop))\n'
            '            hypos.append(decode_string)')

    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('beam_search', 'search')]
    if len(functions) != 2:
        raise RuntimeError('Expected the two released FLARE decoding functions.')
    namespace = {'torch': torch, 'F': torch.nn.functional, 'logger': logger,
                 '_decode_results': _decode_results}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['search']

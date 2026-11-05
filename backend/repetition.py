"""Locate adjacent phrase copies for experimental visual-evidence checks.

Repetition is not itself an error. Return spans, never rewrite or blacklist text.
Word spans are mapped only when re-encoding preserves the exact generated IDs.
"""
from __future__ import annotations

import re
import unicodedata

MIN_REPEAT_WORDS = 3
MAX_REPEAT_WORDS = 32
RELATIVE_SUPPORT_FLOOR = .1
ABSOLUTE_SUPPORT_CEILING = .2


def repeated_phrases(text):
    """Find nonoverlapping runs of at least two identical multiword phrases."""
    words = list(re.finditer(r"[^\W_]+(?:['’][^\W_]+)*", text, re.UNICODE))
    keys = [unicodedata.normalize('NFKC', word.group()).casefold().replace('’', "'") for word in words]
    runs = []
    start = 0
    while start < len(words):
        choices = []
        for width in range(MIN_REPEAT_WORDS, min(MAX_REPEAT_WORDS, (len(words) - start) // 2) + 1):
            unit = keys[start:start + width]
            # A run of single words / two-word phrases is not a phrase loop.
            if any(width % period == 0 and unit == unit[:period] * (width // period) for period in (1, 2)):
                continue
            copies = 1
            while keys[start + copies * width:start + (copies + 1) * width] == unit:
                copies += 1
            if copies >= 2:
                choices.append((width * copies, -width, copies))
        if not choices:
            start += 1
            continue
        total, negative_width, copies = max(choices)
        width = -negative_width
        runs.append({'words_per_copy': width, 'copies': copies,
                     'char_spans': [(words[start + copy * width].start(),
                                     words[start + (copy + 1) * width - 1].end())
                                    for copy in range(copies)]})
        start += total
    return runs


def repeated_token_spans(tokenizer, token_ids):
    """Do not guess token alignments when normalization changes tokenization."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    runs = repeated_phrases(text)
    if not runs:
        return []
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) != list(token_ids):
        return []
    offsets = encoded['offset_mapping']
    mapped = []
    for run in runs:
        spans = [[i for i, (left, right) in enumerate(offsets) if left < end and right > start]
                 for start, end in run['char_spans']]
        # A token shared across two copies would make their evidence inseparable.
        if any(not span for span in spans) or any(set(a) & set(b) for a, b in zip(spans, spans[1:])):
            continue
        mapped.append({'words_per_copy': run['words_per_copy'], 'copies': run['copies'],
                       'token_spans': spans})
    return mapped


def repetition_evidence(runs, real, controls):
    """Reject only copies whose support collapses relative to their first copy.

Use the most favorable frozen control for each copy, as in phrase verification.
Both relative and absolute conditions must fail. This is an experimental rule
from constructed local examples, not a calibrated speech-alignment method.
"""
    findings = []
    for run in runs:
        lifts = [max(float((real[indices] - scores[indices]).mean()) for scores in controls.values())
                 for indices in run['token_spans']]
        threshold = min(ABSOLUTE_SUPPORT_CEILING, max(0., lifts[0]) * RELATIVE_SUPPORT_FLOOR)
        unsupported = [i + 1 for i, lift in enumerate(lifts[1:]) if lift < threshold]
        findings.append({'words_per_copy': run['words_per_copy'], 'copies': run['copies'],
                         'max_lift_per_copy': lifts, 'support_floor': threshold,
                         'unsupported_copy_indices': unsupported})
    return {'accepted': not any(row['unsupported_copy_indices'] for row in findings),
            'runs': findings}

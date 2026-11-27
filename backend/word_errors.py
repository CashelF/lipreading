"""Word error rate for held-out sentence recordings.

Normalization is deliberately small and symmetric: case, punctuation and
curly apostrophes never count as errors; every word does. Numbers are spelled
out in the prompt list, so no number normalization is applied.
"""
from __future__ import annotations

import random
import re

_KEEP = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> list[str]:
    text = text.lower().replace('’', "'").replace('‘', "'").replace('-', ' ')
    return [word for word in (token.strip("'") for token in _KEEP.sub(' ', text).split()) if word]


def align(reference: list[str], hypothesis: list[str]) -> dict:
    """Minimum edits; ties prefer substitutions, the usual WER convention."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[(0, 0, 0)] * cols for _ in range(rows)]  # (substitutions, deletions, insertions)
    for i in range(1, rows):
        cost[i][0] = (0, i, 0)
    for j in range(1, cols):
        cost[0][j] = (0, 0, j)
    for i in range(1, rows):
        for j in range(1, cols):
            s, d, n = cost[i - 1][j - 1]
            options = [(s + (reference[i - 1] != hypothesis[j - 1]), d, n)]
            s, d, n = cost[i - 1][j]
            options.append((s, d + 1, n))
            s, d, n = cost[i][j - 1]
            options.append((s, d, n + 1))
            cost[i][j] = min(options, key=lambda edit: (sum(edit), -edit[0]))
    substitutions, deletions, insertions = cost[-1][-1]
    return {'reference_words': len(reference), 'substitutions': substitutions,
            'deletions': deletions, 'insertions': insertions,
            'errors': substitutions + deletions + insertions}


def score(reference: str, hypothesis: str) -> dict:
    result = align(normalize(reference), normalize(hypothesis))
    result['wer'] = round(result['errors'] / result['reference_words'], 4) if result['reference_words'] else None
    return result


def corpus_wer(rows: list[dict]) -> float | None:
    words = sum(row['reference_words'] for row in rows)
    return round(sum(row['errors'] for row in rows) / words, 4) if words else None


def bootstrap_interval(rows: list[dict], *, samples: int = 2000, seed: int = 0) -> list[float] | None:
    """95% interval over recordings, since words within a clip are not independent."""
    if len(rows) < 2:
        return None
    generator = random.Random(seed)
    values = sorted(corpus_wer([generator.choice(rows) for _ in rows]) for _ in range(samples))
    return [values[int(.025 * samples)], values[int(.975 * samples) - 1]]

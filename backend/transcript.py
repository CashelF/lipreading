"""Reconcile overlapping, untimed clip hypotheses into a revisable transcript.

FLARE returns words without alignments. Positions below are private uniform
estimates used to disambiguate repeated overlap, never word timestamps or speech
confidence. Ambiguous, unanchored sliding windows retain text rather than silently
discarding it; this can leave a duplicate until a later hypothesis supplies an
anchor. No text-only reconciliation can guarantee lossless recognition.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata


def _key(word: str) -> str:
    # Apostrophes are part of a word; case and surrounding punctuation are not.
    word = unicodedata.normalize('NFKC', word).casefold().replace('’', "'")
    return ''.join(character for character in word if character.isalnum() or character == "'").strip("'")


def _tokens(text: str) -> list[str]:
    return re.findall(r'\S+', text.strip())


@dataclass(frozen=True)
class _Word:
    text: str
    start: float
    end: float
    source_start: float

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2


class ContinuousTranscript:
    """Session-local history plus a bounded mutable matching tail.

    Older words become stable after their estimated interval leaves the video
    window with a small guard. Same-start growing windows replace their own tail
    completely. Sliding windows splice at a text anchor selected using overlap
    and approximate position; repeated words are not globally deduplicated.
    """

    def __init__(self, *, max_draft_words: int = 128, guard_ms: float = 750.):
        if max_draft_words < 8 or not math.isfinite(guard_ms) or guard_ms < 0:
            raise ValueError('Invalid transcript history limits.')
        self.max_draft_words = max_draft_words
        self.guard_ms = guard_ms
        self._stable: list[str] = []
        self._draft: list[_Word] = []
        self._last_start: float | None = None
        self._last_end: float | None = None
        self._origin_committed = 0
        self._revision = 0
        self._final = False

    def snapshot(self) -> dict:
        stable = ' '.join(self._stable)
        draft = ' '.join(word.text for word in self._draft)
        return {'stable_text': stable, 'draft_text': draft,
                'text': ' '.join(part for part in (stable, draft) if part),
                'revision': self._revision, 'final': self._final}

    def _commit(self, count: int) -> None:
        self._origin_committed += sum(word.source_start == self._last_start for word in self._draft[:count])
        self._stable.extend(word.text for word in self._draft[:count])
        del self._draft[:count]

    def _commit_before(self, timestamp: float) -> None:
        count = 0
        # Commit only a prefix: uncertain approximate positions must never reorder
        # words or make later words stable ahead of earlier words.
        for word in self._draft:
            if word.end > timestamp:
                break
            count += 1
        self._commit(count)

    def _words(self, text: str, start: float, end: float) -> list[_Word]:
        words = _tokens(text)
        duration = (end - start) / max(1, len(words))
        return [_Word(word, start + index * duration, start + (index + 1) * duration, start)
                for index, word in enumerate(words)]

    def _anchor(self, incoming: list[_Word], start: float, end: float) -> tuple[int, int] | None:
        """Return matching old/new splice points, or None without a safe anchor."""
        if not self._draft or not incoming:
            return None
        old = [_key(word.text) for word in self._draft]
        new = [_key(word.text) for word in incoming]
        # Long model outputs are displayed in full, but matching work stays bounded.
        new = new[:self.max_draft_words]
        rows = [[0] * (len(new) + 1) for _ in range(len(old) + 1)]
        candidates = []
        assert self._last_end is not None and self._last_start is not None
        overlap = max(0., min(self._last_end, end) - max(self._last_start, start))
        old_duration = max(1., self._last_end - self._last_start)
        # Cap the benefit of long repeated strings at the plausible shared span.
        # Otherwise "go go go" would always match its oldest occurrence.
        support_cap = max(1., len(old) * overlap / old_duration)
        word_span = max(200., old_duration / max(1, len(old)))
        for i in range(len(old) - 1, -1, -1):
            for j in range(len(new) - 1, -1, -1):
                if not old[i] or old[i] != new[j]:
                    continue
                length = 1 + rows[i + 1][j + 1]
                rows[i][j] = length
                if i and j and old[i - 1] == new[j - 1]:
                    continue  # Score each contiguous matching run once.
                # One common word inside either phrase is not evidence that their
                # surrounding words describe the same video. A lone word is used
                # only at the exact old-suffix/new-prefix boundary.
                if length < 2 and not (i == len(old) - 1 and j == 0):
                    continue
                if self._draft[i].end < start - self.guard_ms:
                    continue
                distance = abs(self._draft[i].center - incoming[j].center) / word_span
                if distance > max(3., support_cap):
                    continue
                score = 3 * min(length, support_cap) - 2 * distance - .15 * j
                # Keep the latest equally plausible occurrence: it preserves
                # deliberate repetition instead of collapsing earlier words.
                candidates.append((score, length, i, j))
        if not candidates:
            return None
        _, _, old_index, new_index = max(candidates)
        return old_index, new_index

    def update(self, text: str, start_ms: float, end_ms: float, *,
               final: bool = False, suppressed: bool = False) -> dict:
        if not isinstance(text, str):
            raise ValueError('Transcript must be text.')
        if (not math.isfinite(start_ms) or not math.isfinite(end_ms)
                or start_ms < 0 or end_ms <= start_ms):
            raise ValueError('Invalid transcript window.')
        # Inference may finish after capture has moved on. A duplicate final can
        # promote the latest hypothesis, but stale results cannot undo newer text.
        if self._final:
            return self.snapshot()
        if self._last_end is not None:
            if end_ms < self._last_end or (end_ms == self._last_end and not final):
                return self.snapshot()
            if end_ms == self._last_end and start_ms != self._last_start:
                return self.snapshot()

        same_origin = self._last_start == start_ms
        disjoint = self._last_end is not None and start_ms >= self._last_end
        if disjoint:
            self._commit(len(self._draft))
        elif not same_origin:
            self._commit_before(start_ms - self.guard_ms)

        incoming = self._words(text, start_ms, end_ms) if not suppressed else []
        if same_origin:
            # The safety cap can stabilize part of an unusually long hypothesis.
            # Its next growing update must not repeat that already displayed part.
            incoming = incoming[self._origin_committed:]
            # Prefixes retained from an older window must survive replacement or
            # suppression of this growing source window's own provisional words.
            cut = next((index for index, word in enumerate(self._draft)
                        if word.source_start == start_ms), len(self._draft))
            if cut < len(self._draft) or not incoming:
                self._draft = self._draft[:cut] + incoming
            else:
                # A prior silent result may have retained older-source words.
                # Reconcile those when speech resumes instead of duplicating them.
                anchor = self._anchor(incoming, start_ms, end_ms)
                self._draft = (self._draft[:anchor[0]] + incoming[anchor[1]:]
                               if anchor is not None else self._draft + incoming)
        elif incoming:
            anchor = None if disjoint else self._anchor(incoming, start_ms, end_ms)
            # Preserve the complete earlier edge when a new window begins with a
            # clipped word. Do not infer a splice by subtracting token counts:
            # insertions/deletions make that lose valid boundary words.
            self._draft = (self._draft[:anchor[0]] + incoming[anchor[1]:]
                           if anchor is not None else self._draft + incoming)
        # Empty sliding/suppressed results cannot locate prior speech within the
        # overlap. Preserve it; wholly later silence has already committed it.

        if not same_origin:
            self._origin_committed = 0
        self._last_start, self._last_end = start_ms, end_ms
        self._revision += 1
        if len(self._draft) > self.max_draft_words:
            self._commit(len(self._draft) - self.max_draft_words)
        if final:
            self._commit(len(self._draft))
            self._final = True
        return self.snapshot()

    def hold(self, *, final: bool = False) -> dict:
        """No new speech evidence: retain all words from the last decoded video.

        A quiet recent interval says nothing about earlier speech in the same
        rolling window. Do not treat it as an empty replacement hypothesis or
        advance the last decoded window used for overlap reconciliation.
        """
        if final:
            return self.finish()
        if not self._final:
            self._revision += 1
        return self.snapshot()

    def finish(self) -> dict:
        """Finalize retained words without inventing another model hypothesis."""
        if not self._final:
            self._commit(len(self._draft))
            self._revision += 1
            self._final = True
        return self.snapshot()

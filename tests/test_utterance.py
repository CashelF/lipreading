"""Whole-utterance replacement and bounded original-frame retention."""
import numpy as np
import pytest

from backend.stream import Frame, sample_times
from backend.utterance import UtteranceTranscript, VideoHistory


def test_later_prefix_correction_replaces_all_provisional_words():
    transcript = UtteranceTranscript()
    transcript.revise('we should send it tomorrow')
    final = transcript.commit('we could send it tomorrow morning')
    assert final['text'] == 'we could send it tomorrow morning'
    assert final['stable_text'] == final['text']
    assert final['draft_text'] == ''
    assert final['utterances'] == ['we could send it tomorrow morning']


def test_disagreeing_previews_replace_instead_of_duplicate():
    transcript = UtteranceTranscript()
    first = transcript.revise('I need a coffee')
    second = transcript.revise('We need some coffee')
    assert second['text'] == second['draft_text'] == 'We need some coffee'
    assert second['stable_text'] == ''
    assert second['utterances'] == []
    assert second['revision'] > first['revision']


def test_silence_holds_draft_and_final_pass_commits_only_once():
    transcript = UtteranceTranscript()
    transcript.revise('provisional phrase')
    assert transcript.hold()['draft_text'] == 'provisional phrase'
    final = transcript.commit('corrected phrase')
    assert transcript.commit('corrected phrase') == final
    assert transcript.commit() == final
    stopped = transcript.finish()
    assert stopped['text'] == 'corrected phrase'
    assert stopped['final'] and stopped['draft_text'] == ''
    assert transcript.finish() == stopped
    assert transcript.revise('late result') == stopped
    assert transcript.hold() == stopped


def test_new_utterance_revisions_do_not_change_completed_words():
    transcript = UtteranceTranscript()
    transcript.revise('first phrase')
    transcript.commit()
    transcript.revise('second draft')
    result = transcript.revise('second corrected draft')
    assert result['stable_text'] == 'first phrase'
    assert result['draft_text'] == 'second corrected draft'
    assert result['utterances'] == ['first phrase']
    committed = transcript.commit('second final phrase')
    assert committed['text'] == 'first phrase second final phrase'
    assert committed['utterances'] == ['first phrase', 'second final phrase']
    assert result['utterances'] == ['first phrase']  # Earlier snapshots do not mutate.


def test_repeated_actual_utterances_are_not_deduplicated():
    transcript = UtteranceTranscript()
    for _ in range(2):
        transcript.revise('thank you')
        transcript.commit('thank you')
    assert transcript.finish()['text'] == 'thank you thank you'
    assert transcript.snapshot()['utterances'] == ['thank you', 'thank you']


def test_empty_refinement_removes_draft_but_failure_preserves_it():
    transcript = UtteranceTranscript()
    transcript.revise('false draft')
    assert transcript.commit('')['text'] == ''
    assert transcript.snapshot()['utterances'] == []
    transcript.revise('retained preview')
    assert transcript.commit(None)['text'] == 'retained preview'


def test_final_pass_without_preview_can_use_explicit_empty_draft():
    transcript = UtteranceTranscript()
    transcript.revise('')
    assert transcript.active
    assert transcript.commit('the completed utterance')['text'] == 'the completed utterance'
    assert not transcript.active


def test_finish_preserves_unfinalized_draft_on_interruption():
    transcript = UtteranceTranscript()
    transcript.revise('words captured so far')
    assert transcript.finish()['stable_text'] == 'words captured so far'


def test_transcript_normalizes_whitespace_and_rejects_nontext():
    transcript = UtteranceTranscript()
    assert transcript.revise('  one\n two  ')['text'] == 'one two'
    before = transcript.snapshot()
    with pytest.raises(ValueError, match='text'):
        transcript.commit(3)
    assert transcript.snapshot() == before


def history(count=501, step=40., **kwargs):
    # Small custom limits keep eviction tests quick; defaults are tested below.
    bounds = {'retention_ms': 16_000., 'max_clip_ms': 12_000.}
    bounds.update(kwargs)
    result = VideoHistory(**bounds)
    frames = [Frame(1000. + index * step, b'original-jpeg') for index in range(count)]
    for frame in frames:
        result.append(frame)
    return result, frames


def test_history_keeps_source_identity_and_session_sampling_phase():
    retained, original = history(count=251)
    clip = retained.slice(3041.5, 9011., segment_id=7, final=True)
    assert clip.frames[0] is original[51]  # 3040: nearest predecessor.
    assert clip.frames[-1] is original[200]
    assert clip.start_ms == 3041.5
    assert clip.end_ms == 9000.
    assert clip.sample_origin == 1000.
    assert clip.segment_id == 7 and clip.final
    times = sample_times(clip)
    assert times[0] == 3080.
    assert np.all(np.diff(times) == 40.)


def test_history_rejects_lost_prefix_instead_of_truncating():
    retained, original = history(count=501)
    assert retained.oldest_ms == 5000.
    assert retained.latest_ms == 21000.
    assert retained.origin == 1000.
    with pytest.raises(ValueError, match='beginning'):
        retained.slice(4000., 12000., segment_id=0)
    assert retained.slice(5000., 17000., segment_id=1).frames[0] is original[100]


def test_history_keeps_one_bracket_before_time_cutoff():
    retained, _ = history(count=500, step=41.)
    cutoff = retained.latest_ms - 16000.
    assert retained.frames[0].timestamp <= cutoff
    assert retained.frames[1].timestamp > cutoff


@pytest.mark.parametrize('kwargs', [{'max_frames': 20}, {'max_bytes': 20 * len(b'original-jpeg')}])
def test_history_remains_bounded_under_capacity_pressure(kwargs):
    retained, _ = history(count=100, **kwargs)
    assert len(retained.frames) <= 20
    assert retained.total_bytes <= 20 * len(b'original-jpeg')
    with pytest.raises(ValueError, match='beginning'):
        retained.slice(1000., 4000., segment_id=0)


def test_duplicate_last_frame_from_rate_limiter_is_ignored():
    retained, original = history(count=2)
    retained.append(original[-1])
    retained.append(Frame(original[-1].timestamp, b'different-bytes-same-timestamp'))
    assert list(retained.frames) == original
    with pytest.raises(ValueError, match='increase'):
        retained.append(Frame(original[0].timestamp, b'old'))


@pytest.mark.parametrize('start,end,match', [
    (1000., 13001., 'bounded'),
    (1000., 11001., 'not arrived'),
    (1000., 1000., 'interval'),
    (float('nan'), 5000., 'interval'),
    (1000., 1010., 'more video'),
])
def test_invalid_or_incomplete_slices_are_explicit(start, end, match):
    retained, _ = history(count=251)
    with pytest.raises(ValueError, match=match):
        retained.slice(start, end, segment_id=0)


def test_history_clear_releases_frames_and_resets_sampling_origin():
    retained, _ = history(count=30)
    retained.clear()
    assert not retained.frames
    assert retained.origin is None and retained.total_bytes == 0
    retained.append(Frame(50., b'new-session'))
    assert retained.origin == 50.


def test_oversized_frame_does_not_partially_mutate_history():
    retained = VideoHistory(max_bytes=10)
    with pytest.raises(ValueError, match='capacity'):
        retained.append(Frame(0., b'x' * 11))
    assert not retained.frames and retained.origin is None


def test_default_history_keeps_full_minute_and_inflight_capture_at_max_rate():
    retained = VideoHistory()
    for index in range(3201):
        retained.append(Frame(index * 25., b'jpeg'))
    clip = retained.slice(0., 60_000., 0, final=True)
    assert clip.frames[0].timestamp == 0.
    assert clip.frames[-1].timestamp == 60_000.
    assert len(sample_times(clip)) == 1501
    assert len(retained.frames) <= retained.max_frames
    retained.append(Frame(80_025., b'jpeg'))
    with pytest.raises(ValueError, match='beginning'):
        retained.slice(0., 60_000., 0)

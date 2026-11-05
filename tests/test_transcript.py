"""Behavior of untimed overlap reconciliation; no recognizer is simulated here."""
import pytest

from backend.transcript import ContinuousTranscript


def test_growing_window_replaces_complete_hypothesis_without_anchor():
    stream = ContinuousTranscript()
    assert stream.update('the cat', 0, 2000)['text'] == 'the cat'
    result = stream.update('a dog ran away', 0, 4000)
    assert result['text'] == result['draft_text'] == 'a dog ran away'
    assert not result['stable_text']


def test_sliding_window_retains_boundary_context_without_duplicate_overlap():
    stream = ContinuousTranscript()
    stream.update('I want to go to the store', 0, 6000)
    result = stream.update('to go to the store and buy milk', 2000, 8000)
    assert result['text'] == 'I want to go to the store and buy milk'


def test_sliding_tail_correction_replaces_wrong_word():
    stream = ContinuousTranscript()
    stream.update('I want to go to the shore', 0, 6000)
    result = stream.update('to go to the store and buy milk', 2000, 8000)
    assert result['text'] == 'I want to go to the store and buy milk'


def test_next_growing_window_does_not_replace_prior_window_prefix():
    stream = ContinuousTranscript()
    stream.update('I want to go to the shore', 0, 6000)
    stream.update('to go to the store and buy milk', 2000, 8000)
    result = stream.update('to go to the store and buy bread', 2000, 9000)
    assert result['text'] == 'I want to go to the store and buy bread'


def test_single_word_boundary_anchor():
    stream = ContinuousTranscript()
    stream.update('please turn left', 0, 6000)
    result = stream.update('left at the next corner', 4000, 10000)
    assert result['text'] == 'please turn left at the next corner'


def test_punctuation_case_and_curly_apostrophe_do_not_duplicate_overlap():
    stream = ContinuousTranscript()
    stream.update("I think we're ready.", 0, 6000)
    result = stream.update('We’re ready, so let’s go.', 2500, 8500)
    assert result['text'] == 'I think We’re ready, so let’s go.'


def test_consecutive_repeated_words_are_preserved():
    stream = ContinuousTranscript()
    stream.update('that was very very good', 0, 6000)
    result = stream.update('very good indeed', 3500, 9500)
    assert result['text'] == 'that was very very good indeed'


def test_temporal_overlap_selects_later_repeated_phrase():
    stream = ContinuousTranscript()
    stream.update('we go now we go now', 0, 6000)
    result = stream.update('we go now and then stop', 3000, 9000)
    assert result['text'] == 'we go now we go now and then stop'


def test_repeated_phrase_spoken_again_is_not_globally_deduplicated():
    stream = ContinuousTranscript()
    stream.update('we go now we go now', 0, 6000)
    result = stream.update('we go now we go now', 3000, 9000)
    assert result['text'] == 'we go now we go now we go now'


def test_identical_nonoverlapping_utterance_is_kept_twice():
    stream = ContinuousTranscript()
    stream.update('thank you', 0, 2000)
    result = stream.update('thank you', 2000, 4000)
    assert result['text'] == 'thank you thank you'
    assert result['stable_text'] == 'thank you'


def test_no_anchor_preserves_old_words_instead_of_silently_deleting_them():
    stream = ContinuousTranscript()
    stream.update('I remember the address', 0, 6000)
    result = stream.update('forty seven maple street', 3000, 9000)
    assert result['text'] == 'I remember the address forty seven maple street'


def test_internal_common_word_does_not_erase_plausible_old_words():
    stream = ContinuousTranscript()
    stream.update('this is the address where we should meet tomorrow', 0, 6000)
    result = stream.update('the mountain looks beautiful', 1000, 7000)
    assert result['text'] == 'this is the address where we should meet tomorrow the mountain looks beautiful'


def test_clipped_start_word_does_not_replace_complete_old_boundary_word():
    stream = ContinuousTranscript()
    stream.update('this is an interesting question today', 0, 6000)
    result = stream.update('esting question today please', 3000, 9000)
    assert result['text'] == 'this is an interesting question today please'


def test_many_successive_windows_have_no_boundary_loss_or_duplication():
    stream = ContinuousTranscript()
    words = [f'word{index}' for index in range(90)]
    for second in range(25):
        result = stream.update(' '.join(words[second * 3:(second + 6) * 3]), second * 1000, (second + 6) * 1000)
        assert result['text'].split() == words[:(second + 6) * 3]
    assert len(stream._draft) <= stream.max_draft_words


def test_speech_after_silent_overlap_can_reconcile_retained_old_source():
    stream = ContinuousTranscript()
    stream.update('keep this phrase here', 0, 6000)
    stream.update('', 1000, 7000, suppressed=True)
    result = stream.update('this phrase here and more', 1000, 8000)
    assert result['text'] == 'keep this phrase here and more'


def test_suppression_of_growing_window_retracts_only_its_draft():
    stream = ContinuousTranscript()
    stream.update('previous speech', 0, 2000)
    stream.update('invented words', 3000, 5000)
    result = stream.update('', 3000, 6000, suppressed=True)
    assert result['text'] == result['stable_text'] == 'previous speech'
    assert result['draft_text'] == ''


def test_suppression_does_not_erase_older_prefix_inside_mutable_tail():
    stream = ContinuousTranscript()
    stream.update('please turn left', 0, 6000)
    stream.update('left at the corner', 4000, 10000)
    result = stream.update('', 4000, 11000, suppressed=True)
    assert result['text'] == 'please turn'


def test_sliding_silence_keeps_prior_speech_without_inferred_true_timestamps():
    stream = ContinuousTranscript()
    stream.update('words before the pause', 0, 6000)
    result = stream.update('', 1000, 7000, suppressed=True)
    assert result['text'] == 'words before the pause'
    assert set(result) == {'stable_text', 'draft_text', 'text', 'revision', 'final'}


def test_later_silence_finalizes_earlier_text_without_erasing_it():
    stream = ContinuousTranscript()
    stream.update('words before the pause', 0, 6000)
    result = stream.update('', 6500, 12500, suppressed=True)
    assert result['stable_text'] == 'words before the pause'
    assert result['draft_text'] == ''


def test_empty_non_suppressed_growing_result_retracts_its_own_draft():
    stream = ContinuousTranscript()
    stream.update('provisional text', 0, 2000)
    assert stream.update('', 0, 4000)['text'] == ''


def test_stale_and_duplicate_results_do_not_rewind_or_duplicate_caption():
    stream = ContinuousTranscript()
    stream.update('first', 0, 2000)
    newest = stream.update('first second', 0, 4000)
    assert stream.update('outdated', 0, 3000) == newest
    assert stream.update('duplicate', 0, 4000) == newest
    assert stream.update('shorter context', 1000, 4000, final=True) == newest


def test_final_can_promote_same_window_and_is_idempotent():
    stream = ContinuousTranscript()
    before = stream.update('these are the words', 0, 6000)
    result = stream.update('these are the words', 0, 6000, final=True)
    assert result['revision'] == before['revision'] + 1
    assert result['final'] is True
    assert result['draft_text'] == ''
    assert result['stable_text'] == before['text']
    assert stream.finish() == result
    assert stream.update('late words', 1000, 7000) == result


def test_finish_preserves_pending_words_and_advances_revision_without_decoding():
    stream = ContinuousTranscript()
    before = stream.update('short final speech', 0, 800)
    result = stream.finish()
    assert result['final'] is True
    assert result['revision'] == before['revision'] + 1
    assert result['text'] == before['text']
    assert result['draft_text'] == ''


def test_empty_session_finishes_cleanly():
    stream = ContinuousTranscript()
    assert stream.finish() == {'stable_text': '', 'draft_text': '', 'text': '', 'revision': 1, 'final': True}


def test_no_new_speech_preserves_growing_draft_and_decode_overlap():
    stream = ContinuousTranscript()
    first = stream.update('words before the pause', 0, 2000)
    quiet = stream.hold()
    assert quiet['text'] == first['text']
    assert quiet['draft_text'] == first['draft_text']
    assert quiet['revision'] == first['revision'] + 1
    # A later decode still reconciles against the actual earlier hypothesis.
    resumed = stream.update('words before the pause and after', 0, 5000)
    assert resumed['text'] == 'words before the pause and after'
    stopped = stream.hold(final=True)
    assert stopped['text'] == resumed['text']
    assert stopped['stable_text'] == resumed['text']
    assert stream.hold() == stopped


def test_matching_tail_stays_bounded_and_history_remains_available():
    stream = ContinuousTranscript(max_draft_words=8)
    words = [f'word{i}' for i in range(30)]
    result = stream.update(' '.join(words), 0, 6000)
    assert len(stream._draft) <= 8
    assert result['text'].split() == words
    for index in range(1, 50):
        result = stream.update(f'phrase{index} another{index}', index * 6000, (index + 1) * 6000)
        assert len(stream._draft) <= 8
    assert result['text'].startswith(' '.join(words))
    assert result['text'].endswith('phrase49 another49')


def test_growing_long_window_does_not_duplicate_words_committed_by_tail_cap():
    stream = ContinuousTranscript(max_draft_words=8)
    words = [f'word{i}' for i in range(35)]
    stream.update(' '.join(words[:30]), 0, 6000)
    result = stream.update(' '.join(words), 0, 7000)
    assert result['text'].split() == words
    assert len(stream._draft) == 8


@pytest.mark.parametrize('start,end', [(0, 0), (10, 9), (-1, 3), (float('nan'), 4), (0, float('inf'))])
def test_invalid_windows_do_not_mutate_history(start, end):
    stream = ContinuousTranscript()
    before = stream.snapshot()
    with pytest.raises(ValueError, match='window'):
        stream.update('words', start, end)
    assert stream.snapshot() == before

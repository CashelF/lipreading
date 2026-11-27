"""Pilot lipreader training: shards, clip coding, batching and the model's losses.

Runs in the training environment (.venv-train); skipped where its packages are absent.
"""
import math

import pytest

np = pytest.importorskip('numpy')
# Torch must load before PyAV: PyAV's bundled OpenMP runtime otherwise crashes Torch's CPU kernels.
torch = pytest.importorskip('torch')
pytest.importorskip('torchaudio')
pytest.importorskip('av')
pytest.importorskip('sentencepiece')

from training.data import ClipBatches, prepare
from training.model import LipReader
from training.shards import ShardWriter, read_samples
from training.text import Vocabulary, clean, train_vocabulary
from training.train import learning_rate
from training.video import decode_gray, encode_gray, frame_count

# Torch 2.9.1's CPU CTC loss crashes with several OpenMP threads on this machine.
# Training and evaluation compute it on the GPU, where it is unaffected.
torch.set_num_threads(1)

SENTENCES = ['the quick brown fox jumps over the lazy dog', 'please pass the salt and the pepper',
             'we are leaving early in the morning', 'i left my keys on the kitchen table'] * 30


def clip(frames=50, seed=0):
    generator = np.random.default_rng(seed)
    base = np.linspace(20, 230, 96, dtype=np.float32)[None, None, :].repeat(96, 1)
    drift = generator.normal(0, 4, (frames, 1, 1)).cumsum(0)
    return np.clip(base + drift, 0, 255).astype(np.uint8)


@pytest.fixture(scope='module')
def vocabulary(tmp_path_factory):
    prefix = tmp_path_factory.mktemp('vocabulary') / 'unigram'
    return train_vocabulary(SENTENCES, prefix, size=40)


def write_shard(path, clips):
    writer = ShardWriter(path)
    for index, (frames, text) in enumerate(clips):
        writer.add(f'clip{index:04d}', encode_gray(clip(frames, index)), text)
    writer.close()
    return path


def test_recoding_keeps_brightness_and_frame_count():
    original = clip(40)
    encoded = encode_gray(original)
    decoded = decode_gray(encoded)
    assert decoded.shape == original.shape
    assert frame_count(encoded) == 40
    # A range mix-up shifts every pixel by several levels; compression alone does not.
    assert np.abs(decoded.astype(int) - original.astype(int)).mean() < 3
    assert abs(float(decoded.mean()) - float(original.mean())) < 1.5


def test_shard_round_trip_is_atomic_and_reads_both_layouts(tmp_path):
    writer = ShardWriter(tmp_path / 'a.tar')
    writer.add('one', b'video-bytes', 'HELLO THERE', audio=b'sound')
    assert not (tmp_path / 'a.tar').exists()
    writer.close()
    assert list(read_samples(tmp_path / 'a.tar')) == [{'key': 'one', 'video': b'video-bytes', 'text': 'HELLO THERE'}]
    import io, tarfile
    with tarfile.open(tmp_path / 'upstream.tar', 'w') as tar:
        for name, data in (('x.label', b'UPSTREAM TEXT'), ('x.length', b'3'), ('x.video', b'vv'), ('y.video', b'no label')):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    assert list(read_samples(tmp_path / 'upstream.tar')) == [{'key': 'x', 'video': b'vv', 'text': 'UPSTREAM TEXT'}]
    aborted = ShardWriter(tmp_path / 'b.tar')
    aborted.abort()
    assert sorted(path.name for path in tmp_path.iterdir()) == ['a.tar', 'upstream.tar']


def test_vocabulary_reserves_blank_and_sentence_marker(vocabulary):
    words = Vocabulary(vocabulary)
    assert clean("It’s 5 O'CLOCK,  now!") == "it's o'clock now"
    ids = words.encode('Please pass the SALT')
    assert ids and 0 not in ids and max(ids) < words.sos_eos == words.size - 1
    assert words.decode(ids + [words.sos_eos, 0]) == 'please pass the salt'


def test_batches_respect_frame_budget_and_drop_unusable_clips(tmp_path, vocabulary):
    shard = write_shard(tmp_path / 's.tar', [(90, SENTENCES[0]), (96, SENTENCES[1]), (100, SENTENCES[2]),
                                             (5, SENTENCES[3]),                # shorter than min_frames
                                             (150, SENTENCES[0]),              # longer than max_frames
                                             (14, ' '.join(SENTENCES[:3])),    # more tokens than frames allow
                                             (40, '123 456')])                 # nothing left after cleaning
    batches = list(ClipBatches([shard], vocabulary, frames_per_batch=200, max_frames=100, shuffle=False, with_text=True))
    assert sorted(int(n) for batch in batches for n in batch[1]) == [90, 96, 100]
    for video, frames, tokens, token_counts, texts in batches:
        assert video.dtype == torch.uint8 and video.shape[1:] == (int(frames.max()), 96, 96)
        assert video.shape[0] * video.shape[1] <= 200
        assert tokens.shape == (len(texts), int(token_counts.max()))


def test_prepare_crops_normalizes_and_zeroes_padding():
    video = torch.full((2, 6, 96, 96), 128, dtype=torch.uint8)
    frames = torch.tensor([6, 3])
    evaluated = prepare(video, frames, train=False)
    assert evaluated.shape == (2, 6, 88, 88)
    assert torch.allclose(evaluated[0], torch.full((6, 88, 88), (128 / 255 - 0.421) / 0.165))
    assert evaluated[1, 3:].abs().sum() == 0
    trained = prepare(video, frames, train=True)
    assert trained.shape == evaluated.shape and trained[1, 3:].abs().sum() == 0


def test_model_losses_and_decoders_on_a_tiny_network(vocabulary):
    torch.manual_seed(0)
    words = Vocabulary(vocabulary)
    model = LipReader(words.size, width=8, dim=32, layers=1, heads=2, decoder_layers=1)
    video = torch.randn(2, 30, 88, 88)
    frames = torch.tensor([30, 20])
    ids = [words.encode(SENTENCES[0])[:8], words.encode(SENTENCES[1])[:5]]
    tokens = torch.zeros(2, 8, dtype=torch.long)
    for row, values in enumerate(ids):
        tokens[row, :len(values)] = torch.tensor(values)
    loss, ctc, attention = model(video, frames, tokens, torch.tensor([len(v) for v in ids]))
    assert torch.isfinite(loss) and ctc > 0 and attention > 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    greedy = model.greedy(video, frames, max_tokens=6)
    assert len(greedy) == 2 and all(len(row) <= 6 and words.sos_eos not in row for row in greedy)
    best = model.beam_search(video[0], frames[0], beam=3, max_tokens=6)
    assert len(best) <= 6 and all(0 <= token < words.sos_eos for token in best)


def test_padding_does_not_change_a_clips_loss(vocabulary):
    torch.manual_seed(1)
    words = Vocabulary(vocabulary)
    model = LipReader(words.size, width=8, dim=32, layers=1, heads=2, decoder_layers=1).eval()
    video = torch.randn(1, 24, 88, 88)
    tokens = torch.tensor([words.encode(SENTENCES[2])[:6]])
    count = torch.tensor([tokens.shape[1]])
    alone = model(video, torch.tensor([24]), tokens, count)[0]
    padded_video = torch.cat([video, torch.zeros(1, 12, 88, 88)], 1)
    padded_tokens = torch.cat([tokens, torch.zeros(1, 3, dtype=torch.long)], 1)
    padded = model(padded_video, torch.tensor([24]), padded_tokens, count)[0]
    # The 5-frame stem sees zeros past the clip end either way; only tiny numeric drift is allowed.
    assert abs(float(alone) - float(padded)) < 0.05 * float(alone)


def test_learning_rate_warms_up_then_decays_to_zero():
    assert learning_rate(0, 1000, 40, 1e-3) == 0
    assert learning_rate(20, 1000, 40, 1e-3) == pytest.approx(5e-4)
    assert learning_rate(40, 1000, 40, 1e-3) == pytest.approx(1e-3)
    assert learning_rate(520, 1000, 40, 1e-3) == pytest.approx(5e-4)
    assert learning_rate(5000, 1000, 40, 1e-3) == pytest.approx(0, abs=1e-12)
    assert not math.isnan(learning_rate(0, 0, 0, 1e-3))

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
from training.teacher import CODES, WhisperCodes
from training.text import Vocabulary, clean, train_vocabulary
from training.train import learning_rate
from training.video import decode_audio, decode_gray, encode_gray, frame_count, transcode_audio

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
    assert list(read_samples(tmp_path / 'a.tar')) == [{'key': 'one', 'video': b'video-bytes', 'text': 'HELLO THERE', 'audio': b'sound'}]
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


def tone(seconds=2.0, start=0.5):
    """A WAV file: silence, then a 440 Hz tone from ``start`` seconds on."""
    import io, wave
    time = np.arange(int(16000 * seconds)) / 16000
    samples = (np.sin(2 * np.pi * 440 * time) * 12000 * (time >= start)).astype(np.int16)
    output = io.BytesIO()
    with wave.open(output, 'wb') as file:
        file.setnchannels(1), file.setsampwidth(2), file.setframerate(16000)
        file.writeframes(samples.tobytes())
    return output.getvalue(), samples


def test_stored_audio_stays_in_step_with_the_video(tmp_path, vocabulary):
    wav, samples = tone()
    stored = transcode_audio(wav)
    decoded = decode_audio(stored)
    assert abs(len(decoded) - len(samples)) <= 640
    # The tone must start within one video frame (640 samples) of where it started before storage.
    onset = lambda signal: int(np.argmax(np.abs(signal.astype(int)) > 3000))
    assert abs(onset(decoded) - onset(samples)) < 640
    writer = ShardWriter(tmp_path / 'sound.tar')
    writer.add('with', encode_gray(clip(100)), SENTENCES[0], audio=stored)  # Twice as long as its sound.
    writer.add('without', encode_gray(clip(100, 1)), SENTENCES[1])
    writer.close()
    batches = list(ClipBatches([tmp_path / 'sound.tar'], vocabulary, frames_per_batch=200, max_frames=100, shuffle=False,
                               with_audio=True))
    assert len(batches) == 1  # The clip with no soundtrack has nothing to predict.
    video, frames, _, _, audio = batches[0]
    assert audio.dtype == torch.int16 and audio.shape == (1, 100 * 640) and int(frames[0]) == 100
    # The tone starts in video frame 12 (0.48 to 0.52 s) and ends with the sound at frame 50.
    loudness = audio[0].float().abs().view(100, 640).mean(1)
    assert loudness[:12].max() < 100 < loudness[13:49].min() and loudness[51:].max() < 100


@pytest.fixture(scope='module')
def teacher():
    transformers = pytest.importorskip('transformers')
    torch.manual_seed(4)
    config = transformers.WhisperConfig(d_model=32, encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=64,
                                        decoder_layers=1, decoder_attention_heads=2, decoder_ffn_dim=64)
    return WhisperCodes(transformers.WhisperModel(config).encoder, transformers.WhisperFeatureExtractor().mel_filters)


def test_teacher_gives_two_codes_per_video_frame(teacher):
    generator = torch.Generator().manual_seed(5)
    audio = (torch.randn(3, 40 * 640, generator=generator) * 3000).to(torch.int16)
    frames = torch.tensor([40, 40, 25])
    teacher.fit([(audio, frames)])
    codes = teacher(audio)
    assert codes.shape == (3, 40, 2) and codes.dtype == torch.long and 0 <= codes.min() and codes.max() < CODES
    # The codes must spread out; a collapsed quantizer gives the same one or two everywhere.
    assert codes.unique().numel() > 20
    assert torch.equal(codes, teacher(audio))
    # A run that is continued must keep its codes: they come from the saved state, not from a new fit.
    transformers = pytest.importorskip('transformers')
    other = WhisperCodes(teacher.encoder, transformers.WhisperFeatureExtractor().mel_filters, seed=1)
    assert not torch.equal(other(audio), codes)
    other.load_state_dict(teacher.code_state(), strict=False)
    assert torch.equal(other(audio), codes)
    assert not any(parameter.requires_grad for parameter in teacher.parameters())


def test_audio_loss_counts_only_real_frames(vocabulary):
    torch.manual_seed(3)
    words = Vocabulary(vocabulary)
    model = LipReader(words.size, width=8, dim=32, layers=1, heads=2, decoder_layers=1).eval()
    video, frames = torch.randn(2, 30, 88, 88), torch.tensor([30, 20])
    tokens, counts = torch.tensor([words.encode(SENTENCES[0])[:4]] * 2), torch.tensor([4, 4])
    sound = torch.randint(0, CODES, (2, 30, 2))
    without = model(video, frames, tokens, counts)
    loss, _, _, audio = model(video, frames, tokens, counts, audio=sound, audio_weight=2.0)
    assert audio > 0 and float(loss) == pytest.approx(float(without[0]) + 2 * float(audio), rel=1e-5)
    changed = sound.clone()
    changed[1, 20:] = (changed[1, 20:] + 1) % CODES  # Past the end of the second clip.
    assert float(model(video, frames, tokens, counts, audio=changed)[3]) == pytest.approx(float(audio), rel=1e-5)


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
    loss, ctc, attention, sound = model(video, frames, tokens, torch.tensor([len(v) for v in ids]))
    assert torch.isfinite(loss) and ctc > 0 and attention > 0 and sound == 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    greedy = model.greedy(video, frames, max_tokens=6)
    assert len(greedy) == 2 and all(len(row) <= 6 and words.sos_eos not in row for row in greedy)
    best = model.beam_search(video[0], frames[0], beam=3, max_tokens=6)
    assert len(best) <= 6 and all(0 <= token < words.sos_eos for token in best)


def test_losses_are_per_token_not_per_clip(vocabulary):
    torch.manual_seed(2)
    words = Vocabulary(vocabulary)
    model = LipReader(words.size, width=8, dim=32, layers=1, heads=2, decoder_layers=1).eval()
    video = torch.randn(2, 40, 88, 88)
    frames = torch.tensor([40, 40])
    long, short = words.encode(SENTENCES[0])[:9], words.encode(SENTENCES[1])[:3]
    tokens = torch.zeros(2, 9, dtype=torch.long)
    tokens[0], tokens[1, :3] = torch.tensor(long), torch.tensor(short)
    _, ctc, attention, _ = model(video, frames, tokens, torch.tensor([9, 3]))
    alone = [model(video[row:row + 1], frames[row:row + 1], tokens[row:row + 1, :count], torch.tensor([count]))
             for row, count in enumerate((9, 3))]
    # A 9-token clip weighs three times a 3-token clip; the attention loss also counts each end marker.
    assert float(ctc) == pytest.approx((9 * float(alone[0][1]) + 3 * float(alone[1][1])) / 12, rel=1e-4)
    assert float(attention) == pytest.approx((10 * float(alone[0][2]) + 4 * float(alone[1][2])) / 14, rel=1e-4)


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

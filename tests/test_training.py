"""Lipreader training: shards, clip coding, stored features, batching and the model's losses.

Runs in the model environment (.venv-model); skipped where its packages are absent.
"""
import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip('numpy')
# Torch must load before PyAV: PyAV's bundled OpenMP runtime otherwise crashes Torch's CPU kernels.
torch = pytest.importorskip('torch')
pytest.importorskip('torchaudio')
pytest.importorskip('av')
pytest.importorskip('sentencepiece')

from training.data import SCALE, ClipBatches, FeatureBatches, prepare, quantize, restore, time_mask
import training.model
from training.model import FlareLayer, FlareStack, Frontend, LipReader
from training.shards import ShardWriter, read_samples
from training.text import Vocabulary, clean, train_vocabulary
from training.train import learning_rate
from training.video import decode_gray, encode_gray, frame_count

# Torch 2.9.1's CPU CTC loss crashes with several OpenMP threads on this machine.
# Training and evaluation compute it on the GPU, where it is unaffected.
torch.set_num_threads(1)

SENTENCES = ['the quick brown fox jumps over the lazy dog', 'please pass the salt and the pepper',
             'we are leaving early in the morning', 'i left my keys on the kitchen table'] * 30
TINY = dict(dim=32, layers=1, heads=2, decoder_layers=1)


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


def write_features(folder, clips, seed=0):
    """A folder as extract_features.py writes it. Each clip's features are its number, so batches can be checked."""
    folder.mkdir(parents=True)
    generator = torch.Generator().manual_seed(seed)
    front, teacher = [], []
    for number, (frames, _) in enumerate(clips):
        front.append(np.full((frames, 768), number, np.int8))
        teacher.append(quantize(torch.randn(frames, 768, generator=generator)))
    np.save(folder / 'front.npy', np.concatenate(front))
    np.save(folder / 'teacher.npy', np.concatenate(teacher))
    (folder / 'index.json').write_text(json.dumps({'scale': SCALE, 'clips': [[f'clip{n}', text, frames] for n, (frames, text) in enumerate(clips)]}))
    return folder


def test_recoding_keeps_brightness_and_frame_count():
    original = clip(40)
    encoded = encode_gray(original)
    decoded = decode_gray(encoded)
    assert decoded.shape == original.shape
    assert frame_count(encoded) == 40
    # A range mix-up shifts every pixel by several levels; compression alone does not.
    assert np.abs(decoded.astype(int) - original.astype(int)).mean() < 3
    assert abs(float(decoded.mean()) - float(original.mean())) < 1.5


def test_colour_clips_decode_to_the_same_gray_without_conversion():
    import io
    import av
    gray = clip(12)
    output = io.BytesIO()
    with av.open(output, 'w', format='mp4') as container:  # As the upstream VoxCeleb2 clips: 4:4:4 colour.
        stream = container.add_stream('libx264', rate=25)
        stream.width = stream.height = 96
        stream.pix_fmt = 'yuv444p'
        stream.options = {'crf': '10'}
        for image in gray:
            frame = av.VideoFrame.from_ndarray(np.stack([image] * 3, -1), format='rgb24').reformat(format='yuv444p')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    decoded = decode_gray(output.getvalue())
    with av.open(io.BytesIO(output.getvalue())) as container:
        assert next(container.decode(video=0)).format.name == 'yuv444p'
        container.seek(0)
        converted = np.stack([frame.to_ndarray(format='gray') for frame in container.decode(video=0)])
    assert decoded.shape == gray.shape and np.abs(decoded.astype(int) - gray.astype(int)).mean() < 3
    assert np.abs(decoded.astype(int) - converted.astype(int)).max() <= 1  # The slow, converting route agrees.


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


UNUSABLE = [(5, SENTENCES[3]),                # shorter than min_frames
            (150, SENTENCES[0]),              # longer than max_frames
            (14, ' '.join(SENTENCES[:3])),    # more tokens than frames allow
            (40, '123 456')]                  # nothing left after cleaning


def test_clip_batches_respect_frame_budget_and_drop_unusable_clips(tmp_path, vocabulary):
    shard = write_shard(tmp_path / 's.tar', [(90, SENTENCES[0]), (96, SENTENCES[1]), (100, SENTENCES[2])] + UNUSABLE)
    batches = list(ClipBatches([shard], vocabulary, frames_per_batch=200, max_frames=100, shuffle=False, with_text=True))
    assert sorted(int(n) for batch in batches for n in batch[1]) == [90, 96, 100]
    for video, frames, tokens, token_counts, texts in batches:
        assert video.dtype == torch.uint8 and video.shape[1:] == (int(frames.max()), 96, 96)
        assert video.shape[0] * video.shape[1] <= 200
        assert tokens.shape == (len(texts), int(token_counts.max()))


def test_feature_batches_keep_each_clip_with_its_own_text_and_target(tmp_path, vocabulary):
    clips = [(90, SENTENCES[0]), (96, SENTENCES[1]), (100, SENTENCES[2]), (93, SENTENCES[3])] + UNUSABLE
    folder = write_features(tmp_path / 'shard', clips)
    stored = np.load(folder / 'teacher.npy')
    starts = np.cumsum([0] + [frames for frames, _ in clips])
    words = Vocabulary(vocabulary)
    seen = set()
    for shuffle in (False, True):
        loader = FeatureBatches([folder], vocabulary, frames_per_batch=200, max_frames=100, shuffle=shuffle, with_text=True)
        for front, frames, tokens, token_counts, teacher, texts in loader:
            assert front.dtype == teacher.dtype == torch.int8 and front.shape == teacher.shape == (len(texts), int(frames.max()), 768)
            assert front.shape[0] * front.shape[1] <= 200
            for row, text in enumerate(texts):
                number, count = int(front[row, 0, 0]), int(frames[row])  # The clip's number was stored as its features.
                seen.add(number)
                assert clips[number] == (count, text)
                assert tokens[row, :token_counts[row]].tolist() == words.encode(text)
                assert np.array_equal(teacher[row, :count].numpy(), stored[starts[number]:starts[number] + count])
                assert not front[row, count:].any() and not teacher[row, count:].any()
    assert seen == {0, 1, 2, 3}


def test_stored_features_survive_the_round_trip_to_int8():
    features = torch.randn(7, 768) * 3 + 5
    stored = quantize(features)
    standardized = torch.nn.functional.layer_norm(features, (768,))
    assert stored.dtype == np.int8 and float((restore(torch.from_numpy(stored)) - standardized).abs().max()) <= 0.5 / SCALE + 1e-6


def test_prepare_takes_the_centre_and_zeroes_padding():
    video = torch.full((2, 6, 96, 96), 128, dtype=torch.uint8)
    video[:, :, :4] = 0  # Outside the 88x88 centre.
    prepared = prepare(video, torch.tensor([6, 3]))
    assert prepared.shape == (2, 6, 88, 88)
    assert torch.allclose(prepared[0], torch.full((6, 88, 88), (128 / 255 - 0.421) / 0.165))
    assert prepared[1, 3:].abs().sum() == 0


def test_time_mask_hides_whole_frames_inside_each_clip():
    features = torch.ones(2, 100, 8)
    frames = torch.tensor([100, 40])
    masked = time_mask(features, frames)
    hidden = masked.sum(-1) == 0
    assert ((masked.sum(-1) == 8) | hidden).all()  # A frame is hidden whole or not at all.
    assert not hidden[1, 40:].any()  # Never past a clip's end.
    assert hidden[0].sum() <= 40 and hidden[1].sum() <= 10  # At most 10 frames per second of video.


def test_clip_targets_pair_each_clip_with_its_own_stored_features(tmp_path, vocabulary):
    from training.train_frontend import ClipTargets, both_distances, distance
    clips = [(60, SENTENCES[0]), (72, SENTENCES[1]), (66, SENTENCES[2])]
    shard = write_shard(tmp_path / 'part.tar', clips)
    # Features are stored in another order than the shard's, as extraction sorts clips by length.
    folder = tmp_path / 'features' / 'part'
    folder.mkdir(parents=True)
    order = [2, 0, 1]
    np.save(folder / 'front.npy', np.concatenate([np.full((clips[n][0], 768), n, np.int8) for n in order]))
    np.save(folder / 'teacher.npy', np.concatenate([np.full((clips[n][0], 768), n + 10, np.int8) for n in order]))
    (folder / 'index.json').write_text(json.dumps({'clips': [[f'clip{n:04d}', clips[n][1], clips[n][0]] for n in order]}))
    seen = {}
    for video, frames, _, _, wanted in ClipTargets([shard], tmp_path / 'features', vocabulary, frames_per_batch=80, shuffle=False):
        assert video.shape[1:] == (int(frames.max()), 96, 96) and wanted.shape == (len(frames), int(frames.max()), 1536)
        for row, count in enumerate(frames.tolist()):
            seen[count] = int(wanted[row, 0, 0])
            assert (wanted[row, :count, :768] == seen[count]).all() and (wanted[row, :count, 768:] == seen[count] + 10).all()
            assert not wanted[row, count:].any()
    assert seen == {60: 0, 72: 1, 66: 2}  # Each clip length came with the features stored under its own key.
    same = torch.randn(2, 9, 768)
    loss, cosine = distance(same * 3 + 1, torch.nn.functional.layer_norm(same, (768,)), torch.tensor([9, 4]))
    assert float(loss) == pytest.approx(0, abs=1e-4) and float(cosine) == pytest.approx(1, abs=1e-4)
    # With FLARE's encoder behind it, the front end is also judged on what the encoder makes of its features,
    # and the gradient for that reaches the front end through the frozen encoder.
    torch.manual_seed(11)
    frontend, flare = Frontend(), FlareStack(1)
    video, frames = torch.randn(2, 5, 64, 64), torch.tensor([5, 3])
    with torch.no_grad():
        features = torch.nn.functional.layer_norm(frontend(video), (768,))
        deep = torch.nn.functional.layer_norm(flare(features, torch.arange(5)[None] >= frames[:, None]), (768,))
    loss, cosine, deep_cosine = both_distances(frontend, flare, video, torch.cat([features, deep], -1), frames)
    assert float(loss) == pytest.approx(0, abs=1e-3) and float(deep_cosine) == pytest.approx(1, abs=1e-3)
    loss, _, deep_cosine = both_distances(frontend, flare, video, torch.cat([features, deep.roll(1, -1)], -1), frames)
    loss.backward()
    assert float(deep_cosine) < 0.9 and frontend.frontend3D[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in flare.parameters())


def test_front_end_runs_on_smaller_frames():
    video = torch.randn(1, 6, 88, 88)
    assert training.model.shrink(video, 88) is video and training.model.shrink(video, 64).shape == (1, 6, 64, 64)
    model = LipReader(50, resolution=64, **TINY).eval()
    assert model.front(video).shape == (1, 6, 768)


def test_front_end_matches_flare(tmp_path):
    """Our front end must compute what FLARE's does, or its trained weights and stored features mean nothing."""
    source = Path(__file__).resolve().parents[1] / 'vendor/FLARE/flare/convnext.py'
    if not source.exists():
        pytest.skip('FLARE source not installed (scripts/setup_model.py).')
    import importlib.util
    spec = importlib.util.spec_from_file_location('flare_convnext', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(5)
    theirs, ours = module.ConvEncoderT().eval(), Frontend().eval()
    ours.load_state_dict(theirs.state_dict())
    video = torch.randn(2, 12, 88, 88)
    with torch.no_grad():
        assert torch.allclose(ours(video), theirs(video.unsqueeze(1)).transpose(1, 2), atol=1e-5)


def tiny_batch(words):
    ids = [words.encode(SENTENCES[0])[:8], words.encode(SENTENCES[1])[:5]]
    tokens = torch.zeros(2, 8, dtype=torch.long)
    for row, values in enumerate(ids):
        tokens[row, :len(values)] = torch.tensor(values)
    return torch.randn(2, 30, 768), torch.tensor([30, 20]), tokens, torch.tensor([len(v) for v in ids])


def test_model_losses_and_decoders_on_a_tiny_network(vocabulary):
    torch.manual_seed(0)
    words = Vocabulary(vocabulary)
    model = LipReader(words.size, **TINY)
    features, frames, tokens, counts = tiny_batch(words)
    loss, ctc, attention, teacher = model(features, frames, tokens, counts)
    assert torch.isfinite(loss) and ctc > 0 and attention > 0 and teacher == 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in model.frontend.parameters())  # FLARE's front end is never trained.
    model.eval()
    greedy = model.greedy(features, frames, max_tokens=6)
    assert len(greedy) == 2 and all(len(row) <= 6 and words.sos_eos not in row for row in greedy)
    best = model.beam_search(features[0], frames[0], beam=3, max_tokens=6)
    assert len(best) <= 6 and all(0 <= token < words.sos_eos for token in best)
    # On video, the front end gives the same kind of features training reads from disk.
    seen = model.front(torch.randn(1, 9, 88, 88))
    assert seen.shape == (1, 9, 768) and torch.allclose(seen.mean(-1), torch.zeros(1, 9), atol=1e-4)


def test_losses_are_per_token_not_per_clip(vocabulary):
    torch.manual_seed(2)
    words = Vocabulary(vocabulary)
    model = LipReader(words.size, **TINY).eval()
    features = torch.randn(2, 40, 768)
    frames = torch.tensor([40, 40])
    long, short = words.encode(SENTENCES[0])[:9], words.encode(SENTENCES[1])[:3]
    tokens = torch.zeros(2, 9, dtype=torch.long)
    tokens[0], tokens[1, :3] = torch.tensor(long), torch.tensor(short)
    with torch.no_grad():
        _, ctc, attention, _ = model(features, frames, tokens, torch.tensor([9, 3]))
        alone = [model(features[row:row + 1], frames[row:row + 1], tokens[row:row + 1, :count], torch.tensor([count]))
                 for row, count in enumerate((9, 3))]
    # A 9-token clip weighs three times a 3-token clip; the attention loss also counts each end marker.
    assert float(ctc) == pytest.approx((9 * float(alone[0][1]) + 3 * float(alone[1][1])) / 12, rel=1e-4)
    assert float(attention) == pytest.approx((10 * float(alone[0][2]) + 4 * float(alone[1][2])) / 14, rel=1e-4)


def test_teacher_loss_counts_only_real_frames_and_is_zero_for_a_perfect_copy(vocabulary):
    torch.manual_seed(3)
    words = Vocabulary(vocabulary)
    model = LipReader(words.size, **TINY).eval()
    features, frames, tokens, counts = tiny_batch(words)
    wanted = torch.randn(2, 30, 768)
    with torch.no_grad():
        without = model(features, frames, tokens, counts)
        loss, _, _, teacher = model(features, frames, tokens, counts, teacher=wanted, teacher_weight=2.0)
        assert teacher > 0 and float(loss) == pytest.approx(float(without[0]) + 2 * float(teacher), rel=1e-5)
        changed = wanted.clone()
        changed[1, 20:] += 100  # Past the end of the second clip.
        assert float(model(features, frames, tokens, counts, teacher=changed)[3]) == pytest.approx(float(teacher), rel=1e-5)
        states, _, _ = model.encode(features, frames)
        assert float(model(features, frames, tokens, counts, teacher=model.teacher(states))[3]) == pytest.approx(0, abs=1e-5)


def test_a_clip_is_encoded_the_same_alone_or_beside_a_longer_clip(vocabulary):
    torch.manual_seed(6)
    model = LipReader(Vocabulary(vocabulary).size, dim=32, layers=3, heads=2, decoder_layers=1).eval()
    features, frames = torch.randn(2, 80, 768), torch.tensor([80, 30])
    with torch.no_grad():
        together, _, padding = model.encode(features, frames)
        alone, _, _ = model.encode(features[1:, :30], frames[1:])
    assert padding[1, 30:].all() and not padding[1, :30].any()
    assert torch.allclose(together[1, :30], alone[0], atol=1e-5)


def test_flare_layer_matches_flare():
    """The ported layer must compute what FLARE's does, or FLARE's weights mean nothing in it."""
    source = Path(__file__).resolve().parents[1] / 'vendor/flare_runtime/models/qwen_conformer.py'
    if not source.exists():
        pytest.skip('FLARE runtime not installed (scripts/setup_model.py).')
    pytest.importorskip('einops')
    pytest.importorskip('transformers')
    import importlib.util
    import types
    spec = importlib.util.spec_from_file_location('flare_qwen_conformer', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(7)
    theirs = module.Qwen2DecoderLayer(types.SimpleNamespace(encoder_embed_dim=64, encoder_attention_heads=4, num_key_value_heads=2), 0).eval()
    ours = FlareLayer(dim=64, heads=4, shared_heads=2).eval()
    ours.load_state_dict(theirs.state_dict())
    x, padding = torch.randn(3, 50, 64), torch.zeros(3, 50, dtype=torch.bool)
    with torch.no_grad():
        expected = theirs(x, attention_mask=padding[:, None, None], position_ids=torch.arange(50)[None])[0]
        assert torch.allclose(ours(x, padding), expected, atol=1e-5)


def test_flare_stack_keeps_the_first_layers_frozen_and_ignores_padding():
    torch.manual_seed(8)
    stack = FlareStack(2)
    whole = {f'layers.{index}.{name}': torch.full_like(value, float(index))
             for index in range(18) for name, value in stack.layers[0].state_dict().items()}
    whole.update({name: value for name, value in stack.state_dict().items() if not name.startswith('layers.')})
    stack.load_flare(whole)
    assert [float(layer.post_layernorm.weight[0]) for layer in stack.layers] == [0, 1]  # The first two of the 18.
    stack = FlareStack(2)
    assert not stack.training and not any(module.training for module in stack.modules())  # From the start: no dropout.
    stack.train()
    assert not stack.training and not any(p.requires_grad for p in stack.parameters())
    features, frames = torch.randn(2, 60, 768), torch.tensor([60, 25])
    padding = torch.arange(60)[None] >= frames[:, None]
    together = stack(features, padding)
    alone = stack(features[1:, :25], padding[1:, :25])
    assert together.shape == (2, 60, 768) and not together.requires_grad
    assert torch.allclose(together[1, :25], alone[0], atol=1e-4)


def test_model_reads_the_output_of_its_flare_layers(vocabulary, monkeypatch):
    torch.manual_seed(9)
    words = Vocabulary(vocabulary)
    features, frames, tokens, counts = tiny_batch(words)
    wanted = torch.randn(2, 30, 768)
    model = LipReader(words.size, flare_layers=1, **TINY)
    loss, ctc, attention, teacher = model(features, frames, tokens, counts, teacher=wanted, teacher_weight=1.0)
    loss.backward()
    assert teacher > 0 and all(p.grad is None for p in model.flare.parameters())  # FLARE's layers are never trained.
    assert model.teacher.weight.grad is not None
    model.eval()
    with torch.no_grad():
        states, _, _ = model.encode(features, frames)
        changed, _, _ = model.encode(features + 1, frames)
        assert not torch.allclose(states, changed)
    # With every FLARE layer the encoder is FLARE's own: nothing to copy, and training hands in
    # FLARE's stored output instead of running the layers.
    monkeypatch.setattr(training.model, 'FLARE_DEPTH', 1)
    model = LipReader(words.size, flare_layers=1, **TINY).eval()
    assert model.whole_flare and model.teacher is None
    with torch.no_grad():
        loss, _, _, teacher = model(features, frames, tokens, counts, teacher=wanted, teacher_weight=1.0)
        assert teacher == 0
        states, _, _ = model.encode(features, frames, wanted)
        again, _, _ = model.encode(torch.randn_like(features), frames, wanted)
        assert torch.allclose(states, again)  # The stored FLARE output is all it reads.


def test_padding_does_not_change_a_clips_loss(vocabulary):
    torch.manual_seed(1)
    words = Vocabulary(vocabulary)
    model = LipReader(words.size, **TINY).eval()
    features = torch.randn(1, 24, 768)
    tokens = torch.tensor([words.encode(SENTENCES[2])[:6]])
    count = torch.tensor([tokens.shape[1]])
    with torch.no_grad():
        alone = model(features, torch.tensor([24]), tokens, count)[0]
        padded_features = torch.cat([features, torch.zeros(1, 12, 768)], 1)
        padded_tokens = torch.cat([tokens, torch.zeros(1, 3, dtype=torch.long)], 1)
        padded = model(padded_features, torch.tensor([24]), padded_tokens, count)[0]
    assert float(alone) == pytest.approx(float(padded), rel=1e-4)


def test_learning_rate_warms_up_then_decays_to_zero():
    assert learning_rate(0, 1000, 40, 1e-3) == 0
    assert learning_rate(20, 1000, 40, 1e-3) == pytest.approx(5e-4)
    assert learning_rate(40, 1000, 40, 1e-3) == pytest.approx(1e-3)
    assert learning_rate(520, 1000, 40, 1e-3) == pytest.approx(5e-4)
    assert learning_rate(5000, 1000, 40, 1e-3) == pytest.approx(0, abs=1e-12)
    assert not math.isnan(learning_rate(0, 0, 0, 1e-3))

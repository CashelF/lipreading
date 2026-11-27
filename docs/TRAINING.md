# Training the pilot lipreader

Research code for the phone-sized model in [SMALL_MODEL_PLAN.md](SMALL_MODEL_PLAN.md). The live app never imports `training/`, and training never uses video from app sessions.

## Environment

Training runs in the model environment from [MODEL.md](../MODEL.md#install). Its only extra package is PyAV (`av`, listed in `requirements-model.txt`), which has no dependencies and leaves the pinned NumPy and OpenCV alone:

```bash
.venv-model/bin/python -m pip install --no-deps av==17.1.0
.venv-model/bin/python -m pytest -q tests/test_training.py
```

Import Torch before PyAV in new scripts, and compute the CTC loss on the GPU: Torch 2.9.1's CPU CTC loss crashes with several OpenMP threads on the development machine, so the tests pin one thread.

## Data

The pilot uses two parts of the public [AVYT release](https://huggingface.co/datasets/nguyenvulebinh/AVYT): `avyt/talking-train` (about 1,450 hours) and `vox2/` (VoxCeleb2 English). Both are 96×96 mouth clips at 25 fps with a transcript per clip. They carry no license and come from YouTube, so models trained on them are research prototypes only.

```bash
.venv-model/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('nguyenvulebinh/AVYT', repo_type='dataset', local_dir='data/train/avyt', allow_patterns=['avyt/talking-train-*.tar', 'vox2/vox2-*.tar'])"
.venv-model/bin/python training/prepare_avyt.py --source data/train/avyt/avyt --output data/train/lips
```

**Keep storage small.** `prepare_avyt.py` rewrites each 1.1 GB `talking-train` shard as grayscale H.264 (CRF 17) plus a 16 kbit/s Opus soundtrack, about 95 MB, and then deletes the source shard. Pass `--watch <systemd unit>` to convert while a download is still running. The `vox2/` shards are already compact and are read as they are. For any larger set, crop and convert while streaming and never keep the source video.

A shard is a tar of `<clip>.mp4`, `<clip>.txt` and optionally `<clip>.ogg` (upstream shards use `.video` and `.label`; both layouts are read). Everything under `data/` is ignored by Git.

## Train

```bash
systemd-run --user --unit=lipreading-pilot-train --property=KillMode=mixed --property=TimeoutStopSec=180 \
  --property=WorkingDirectory="$PWD" training/run_pilot.sh
tail -f data/train/runs/pilot/train.jsonl
systemctl --user stop lipreading-pilot-train    # saves a checkpoint; start again to resume
```

`training/run_pilot.sh` waits for the download and conversion units, builds a 1,000-piece SentencePiece vocabulary once, then runs `training/train.py`. Check `nvidia-smi` first: the run holds about 26 GiB of GPU memory, and the app needs 28 GiB free to load.

- **Model:** 3D-conv stem and ResNet18 per frame, a 12-layer Conformer (width 256), a CTC head and a 6-layer attention decoder. Loss is 0.1 CTC + 0.9 attention with label smoothing.
- **Recipe:** AdamW, peak learning rate 7e-4, warm-up then cosine decay measured in hours of video seen, 9,600 frames per batch, bf16 autocast, 20 passes. Augmentation is random 88×88 crop, horizontal flip and time masking. The first pass uses only clips of 4 seconds or less, because lipreading from scratch needs short clips to learn alignment.
- **Files in the run folder:** `train.jsonl` (numbers only), `last.pt` (written every 2,000 steps and on stop; the run resumes from it) and `best.pt` (lowest dev WER).
- The first shard of each source is the dev set and is excluded from training. Their speakers can also appear in training video, so dev WER is for tracking progress, not a held-out result.

## Evaluate

```bash
.venv-model/bin/python training/evaluate.py --checkpoint data/train/runs/pilot/best.pt \
  --vocabulary data/train/vocabulary/unigram1000.model --shards data/train/avyt/vox2/vox2-test-*.tar
```

`vox2-test` speakers are not in `vox2-dev`. Its transcripts are machine-made, so read small differences with care. Reports hold error counts only.

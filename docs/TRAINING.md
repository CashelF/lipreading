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

## How it learns

The small model is not trained from scratch. It is distilled from FLARE, the large model the app runs:

- **Front end:** FLARE's own lip front end (a 3D patch stem and a small ConvNeXt, 14M parameters), with FLARE's trained weights, kept as they are.
- **Encoder:** a 12-layer Conformer (width 256) that is trained to reproduce FLARE's encoder output for every frame, from the same video.
- **Text:** a CTC head and a 6-layer attention decoder trained on the transcripts at the same time. 40M parameters in all, 26M of them trained; no language model.

FLARE reads these clips well, so it is a good teacher. Lip stream only, on clips decoded exactly as training decodes them, it scored 10% WER on 60 clips from its own training data and 31% (95% interval 26–38%) on 60 clips of held-out VoxCeleb2 speakers (beam 5, measured 2026-11-27).

**Why not from scratch.** Three attempts on 2026-11-27 trained on text alone, on text with a long short-clip stage, and on text plus audio targets (the raw spectrum, then codes from a Whisper teacher). After up to 2,800 video-hours none had a dev WER under 100%. FLARE's checkpoint shows its encoder took 800,000 pretraining steps to learn the same thing; a from-scratch run here would need weeks. With FLARE as the teacher the losses pass those runs' best within 300 video-hours. Logs are in `data/train/archive/`.

## Extract features

FLARE runs once over every clip and saves what training reads:

```bash
.venv-model/bin/python training/extract_features.py --output data/train/features \
  --shards "data/train/lips/avyt-talking-*.tar" "data/train/avyt/vox2/vox2-dev-*.tar"
```

- **Output:** one folder per clip shard. For every video frame it holds FLARE's lip front-end output (`front.npy`, the small model's input) and FLARE's encoder output (`teacher.npy`, its target), each 768 values standardized per frame and stored as int8. `index.json` lists each clip's key, transcript and frame count. `frontend.pt` holds the front-end weights, which every trained checkpoint keeps.
- **Cost:** about 1,000 times real time on this GPU (2.4 hours for the pilot's 2,400 hours of video) and 131 MB per video-hour (about 320 GB for the pilot). Finished shards are skipped on a rerun.
- **Needs** the app's model setup (`scripts/setup_model.py`): `vendor/flare_runtime` and `models/FLARE-checkpoints`. Training itself needs neither, so it can run on a machine that has only the features.
- **No padding.** A batch holds only clips of the same length. FLARE's encoder lets padded frames leak into real ones: a clip padded beside a longer one got features only 0.55–0.97 similar to those it gets alone, which is how the app runs it.
- FLARE's face stream is switched off (the clips are mouth crops) and skipped, which halves the cost and does not change the output.

## Train

```bash
systemd-run --user --unit=lipreading-pilot-train --property=KillMode=mixed --property=TimeoutStopSec=180 \
  --property=WorkingDirectory="$PWD" training/run_pilot.sh
tail -f data/train/runs/pilot/train.jsonl
systemctl --user stop lipreading-pilot-train    # saves a checkpoint; start again to resume
```

`training/run_pilot.sh` waits for the download and conversion units, builds a 1,000-piece SentencePiece vocabulary once, extracts any missing features, then runs `training/train.py` for 20 passes (about 48,000 video-hours).

- **Speed:** about 4,000 video-hours per hour in 4 GiB of GPU memory, because no video is decoded and neither FLARE nor the front end runs. The same model trained from video managed about 500 with FLARE in the loop.
- **Losses:** 0.1 CTC + 0.9 attention (with label smoothing), both per token, plus 5 × the teacher loss (`--teacher-weight`): mean absolute error plus one minus cosine similarity between the model's projected states and FLARE's features.
- **Recipe:** AdamW, peak learning rate 5e-4, warm-up then cosine decay measured in hours of video seen, 9,600 frames per batch, bf16 autocast. The only augmentation is time masking of the input features: the model must fill the gap from context, while its target came from the whole clip.
- **Padding:** the encoder zeroes padded frames before each convolution, so a clip is encoded the same alone or batched with longer clips.
- **Files in a run folder:** `train.jsonl` (numbers only), `last.pt` (written every 5,000 steps and on stop; the run resumes from it) and `best.pt` (lowest dev WER). A checkpoint is the whole model, front end included.
- The first shard of each source is the dev set and is excluded from training. Their speakers can also appear in training video, so dev WER is for tracking progress, not a held-out result.

## Watch

```bash
systemd-run --user --unit=lipreading-dashboard --property=WorkingDirectory="$PWD" \
  python3 training/dashboard.py --bind "$(tailscale ip -4)" --port 8470
```

Open `http://<machine's Tailscale name>:8470` from any device on the tailnet. The page charts dev WER, the text and teacher losses, speed and learning rate against hours of video seen, for every run under `data/train/runs/`, and refreshes every 30 seconds. It serves only the numbers in each `train.jsonl`. Move a finished experiment's folder out of `runs/` to drop it from the page.

## Evaluate

```bash
.venv-model/bin/python training/evaluate.py --checkpoint data/train/runs/pilot/best.pt \
  --vocabulary data/train/vocabulary/unigram1000.model --shards data/train/avyt/vox2/vox2-test-*.tar
```

`vox2-test` speakers are not in `vox2-dev`. Its transcripts are machine-made, so read small differences with care. Reports hold error counts only.

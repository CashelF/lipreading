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

The model is FLARE, the large model the app runs, with its 7B language model taken out and the rest cut down in steps. Every part starts from FLARE's own weights; nothing but the small text side is trained from scratch.

- **Front end:** FLARE's lip front end (a 3D patch stem and a small ConvNeXt, 14M parameters). It is most of the compute, and [a cheaper version](#a-cheaper-front-end) of it works from smaller frames.
- **Encoder:** FLARE's 18 encoder layers (314M parameters), or some of them.
- **Text side:** a 12-layer Conformer (width 256), a CTC head and a 6-layer attention decoder: 26M parameters, in place of FLARE's language model.

Training has two stages after the features are extracted:

1. **The text side learns to read FLARE's whole encoder**, which stays frozen. Training reads the encoder's stored output, so no FLARE layer runs.
2. **Encoder layers are dropped** (`--keep-layers` lists those that stay) and what is left, text side included, starts from stage 1 and is trained to give what the whole encoder gave while still reading text.

**What did not work** (2026-11-27 and 28; logs in `data/train/archive/`): training on text alone, with or without audio targets (dev WER never under 100% in 2,800 video-hours); a small encoder trained from scratch to copy FLARE's output (40M model, best dev WER 74%); FLARE's first six layers frozen with a Conformer above them (146M, best 67%); six layers picked evenly from the 18 and trained from FLARE's weights; a per-layer target for each kept layer in stage 2 (no better than the final output alone). The two from-scratch encoder runs also collapsed after about 88,000 steps at a learning rate of 4e-4 or more: their losses jumped back to the untrained level.

## Results so far

Measured 2026-11-28 unless noted. Dev WER is greedy decoding from saved features on the dev shards of both sources (FLARE trained on those clips). Held-out WER is on VoxCeleb2 speakers that nothing here trained on (`vox2-test-000000`), from raw video with beam 5.

| Model | Parameters | Training | Dev WER | Held-out WER |
| --- | --- | --- | --- | --- |
| FLARE with its 7B LLM, lip stream only | 342M + 7B | — | not measured on this dev set | 31% (26–38%, 60 clips) |
| Stage 1: FLARE's whole encoder + text side | 355M (354 MB at 8 bits) | 10,800 video-hours, 3 hours | 28.8% | 34.5% (31–39%, 159 clips) |
| Stage 1 with the [64×64 front end](#a-cheaper-front-end) | 355M | front end: 400 more video-hours | not measured | 38.3% (34–43%, 159 clips) |
| Stage 2, 12 layers kept (0 1 5 6 7 9 11 12 13 14 15 17) | 250M | 945 video-hours so far | 40.6% | not measured |
| Stage 2, 9 layers kept (0 6 7 9 11 12 13 15 17) | 198M | 945 video-hours so far | 51.2% | not measured |

- **The language model is not needed.** A 26M-parameter text side on FLARE's encoder is within a few points of FLARE with its LLM on held-out speakers, in a model about 20 times smaller.
- **Dropping layers costs accuracy and saves little.** Skipping any one of layers 1–9 costs 1–3 dev WER points before retraining, but the losses add up: a third of the layers gone is about 12 points worse after 945 video-hours of stage 2, and still improving only slowly. Each layer is 17M parameters but little compute.
- **Compute is in the front end.** Per second of video, through ONNX Runtime Web on this machine's CPU (WebAssembly, 4 threads): front end 201 ms at 88×88, 140 ms at 64×64, 92 ms at 48×48; one FLARE layer 6 ms; one text-side layer 1 ms. So the whole 355M model is about 320 ms per second of video there, two thirds of it front end. On the GPU the front end is 63% of the encoder side.
- **Stored size is cheap to cut:** 8-bit weights change nothing (see the front-end section).
- **Not measured yet: a phone.** `data/train/phone-bench/` holds a page that times these parts in a phone browser (WebGPU) and records the result.

## Extract features

FLARE runs once over every clip and saves what training reads:

```bash
.venv-model/bin/python training/extract_features.py --output data/train/features \
  --shards "data/train/lips/avyt-talking-*.tar" "data/train/avyt/vox2/vox2-dev-*.tar"
```

- **Output:** one folder per clip shard, plus `encoder.pt` (FLARE's encoder weights, which every model starts from). For every video frame it holds FLARE's lip front-end output (`front.npy`, the small model's input) and FLARE's encoder output (`teacher.npy`, its target), each 768 values standardized per frame and stored as int8. `index.json` lists each clip's key, transcript and frame count. `frontend.pt` holds the front-end weights, which every trained checkpoint keeps.
- **Cost:** 131 MB per video-hour (about 320 GB for the pilot). The AVYT shards ran at about 1,000 times real time on this GPU; the VoxCeleb2 shards were far slower while the machine was short of memory. Finished shards are skipped on a rerun.
- **Memory:** about 40 GB with the default 8 workers. Start it with a cap (`systemd-run --property=MemoryMax=60G ...`). The first full run used 16 workers holding 4,000 clips each, peaked at 125 GB beside other jobs, and the machine's out-of-memory killer took other people's processes with it.
- **Needs** the app's model setup (`scripts/setup_model.py`): `vendor/flare_runtime` and `models/FLARE-checkpoints`. Training itself needs neither, so it can run on a machine that has only the features.
- **No padding.** A batch holds only clips of the same length. FLARE's encoder lets padded frames leak into real ones: a clip padded beside a longer one got features only 0.55–0.97 similar to those it gets alone, which is how the app runs it.
- FLARE's face stream is switched off (the clips are mouth crops) and skipped, which halves the cost and does not change the output.

## Train

```bash
systemd-run --user --unit=lipreading-pilot-train --property=MemoryMax=60G --property=MemorySwapMax=0 \
  --property=KillMode=mixed --property=TimeoutStopSec=180 --property=WorkingDirectory="$PWD" training/run_pilot.sh
tail -f data/train/runs/*/train.jsonl
systemctl --user stop lipreading-pilot-train    # saves a checkpoint; start again to resume
```

`training/run_pilot.sh` waits for the download and conversion units, builds a 1,000-piece SentencePiece vocabulary once, extracts any missing features, then runs the two stages of `training/train.py`:

```bash
# Stage 1: text side on FLARE's whole encoder (5 passes, about 3 hours).
.venv-model/bin/python training/train.py --features data/train/features --train "avyt-talking-*" "vox2-dev-*" \
  --dev avyt-talking-000000 vox2-dev-000000 --vocabulary data/train/vocabulary/unigram1000.model \
  --output data/train/runs/flare-18 --hours 2400 --epochs 5 --frames-per-batch 9600 --lr 5e-4
# Stage 2: keep 12 of the 18 layers and train what is left.
.venv-model/bin/python training/train.py ... --output data/train/runs/keep-12 --init data/train/runs/flare-18/best.pt \
  --keep-layers 0 1 5 6 7 9 11 12 13 14 15 17 --hours 2400 --epochs 1 --lr 1e-4
```

- **Always cap memory** (`--property=MemoryMax=24G --property=MemorySwapMax=0` on the unit) and check `free -g` first. The machine is shared, and a job without a cap takes other people's processes down with it when memory runs out.
- **Speed:** stage 1 about 4,000 video-hours per hour in 5 GiB of GPU memory, because no video is decoded and no FLARE layer runs. Stage 2 runs the kept layers with gradients: about 1,100 video-hours per hour and 21 GiB with 12 layers.
- **Losses:** 0.1 CTC + 0.9 attention (with label smoothing), both per token. In stage 2, plus 5 × the teacher loss (`--teacher-weight`): mean absolute error plus one minus cosine similarity between the kept layers' output and FLARE's stored encoder output, both standardized per frame.
- **Recipe:** AdamW, warm-up then cosine decay measured in hours of video seen, 9,600 frames per batch, bf16 autocast. Peak learning rate 5e-4 for stage 1 and 1e-4 for stage 2; long runs at 4e-4 or more have collapsed. The only augmentation is time masking of the input features.
- **Padding:** padded frames are zeroed before each convolution in both the FLARE layers and the Conformer, so a clip is encoded the same alone or batched with longer clips.
- **Files in a run folder:** `train.jsonl` (numbers only), `last.pt` and `best.pt` (lowest dev WER). Dev WER is measured and `last.pt` written every 5,000 steps (about 470 video-hours) and at the end; the run resumes from `last.pt`. A checkpoint is the whole model, front end and FLARE layers included.
- The first shard of each source is the dev set and is excluded from training. Their speakers can also appear in training video, so dev WER is for tracking progress, not a held-out result.

## A cheaper front end

The front end is most of the model's compute: on this GPU 1.7 ms per second of video, against 0.8 ms for all 18 FLARE layers and 0.2 ms for the Conformer. Its cost falls with the square of the frame size, so `training/train_frontend.py` teaches it to work from smaller frames:

```bash
.venv-model/bin/python training/train_frontend.py --resolution 64 --hours 400 \
  --clips "data/train/lips/avyt-talking-*.tar" --features data/train/features \
  --dev data/train/lips/avyt-talking-000000.tar --vocabulary data/train/vocabulary/unigram1000.model \
  --output data/train/frontends/res64
```

It starts from FLARE's front-end weights and learns, from video resized to the given size, to reproduce what `extract_features.py` saved at full resolution: the front end's own features and, through FLARE's frozen encoder, the encoder's output. `evaluate.py --frontend data/train/frontends/res64/frontend.pt` scores any trained checkpoint with it swapped in.

Measured 2026-11-27 with the whole-FLARE-encoder checkpoint trained on AVYT only, and nothing but the front end changed. The 64×64 front end was trained for 400 video-hours on AVYT clips against the front-end features alone, before the encoder-output target was added (dev cosine to FLARE's features 0.965), and costs 2.25 times less.

| Front end | Dev WER (323 clips, greedy) | Held-out WER (159 clips, beam 5) |
| --- | --- | --- |
| FLARE's, 88×88 | 24.5% | 35.5% (32–39%) |
| Same weights at 64×64, untrained | 58.1% | not measured |
| Distilled, 64×64 | 28.7% | 39.4% (36–43%) |

The four points lost are before any adaptation: the front end never saw a VoxCeleb2 clip, and the layers behind it were not retrained on its features.

**Stored size.** With that 64×64 front end, rounding every weight matrix to 8 bits per output channel leaves dev WER unchanged (28.7%) at 354 MB, against 1.4 GB in 32-bit floats. At 6 bits it is 30.9% at 266 MB; at 5 and 4 bits plain rounding breaks it (43% and 55%).

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

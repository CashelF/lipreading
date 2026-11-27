# Plan: a phone-sized lipreader

Status: **proposal; costs measured and data surveyed 2026-11-26.** Nothing here is in the live app.

Goal: run lipreading in a phone browser in real time, with word accuracy equal to FLARE-LLM Large or slightly worse, measured on held-out video (see Phases).

## Approach

Train a small streaming model from scratch. Don't scale FLARE up or post-train it.

- About 95% of FLARE's parameters are its 7B Qwen decoder, which mostly supplies language knowledge. In the FLARE paper the same 342M visual model scores 15.9% LRS3 WER with a small 58M decoder and 13.2% with Qwen, so the LLM is worth under 3 points. See [Is 10k hours enough?](#is-10k-hours-enough) for the published data-scaling results.
- **Labels come from audio.** Whisper transcribes each training video's soundtrack, and the lipreader learns from the silent frames only. At run time the app still never uses audio.
- **FLARE can't label at scale.** It took 2.9 s of GPU time to decode a 4 s clip (beam 30), so labelling 10k hours would take about 7,200 GPU-hours. FLARE's visual-encoder features are a cheaper optional teacher, because they need no language-model decoding.

Pipeline: video → MediaPipe face landmarks (the same tracker the phone would run) → 88×88 grayscale lip and face crops → Whisper labels → training → int8 export → WebGPU in the browser.

## Measured costs on this machine

All throughput numbers below come from this machine: an RTX PRO 6000 Blackwell (96 GB) and a 32-core Threadripper 9970X. Training used synthetic tensors, so the numbers show compute only, not accuracy.

| Step | Measured | Per 10k hours of video |
| --- | --- | --- |
| Whisper large-v3 labels (batched, word timestamps) | 237× real time on 1 h of LibriSpeech audio | ~42 GPU-hours (~2 days locally) |
| MediaPipe face landmarks (CPU) | 150 fps per process; 1,620 fps with 32 processes | ~155 hours locally (~6.5 days) or ~1,700 vCPU-hours in the cloud (~$50–100) |
| Training, 30M model (3D-conv + ResNet18 front end, 12-layer conformer, width 256, CTC loss) | ~670 video-hours per GPU-hour (bf16) | ~15 GPU-hours per epoch |
| Training, 10M "phone" model (half-width ResNet18, 8-layer conformer, width 192) | ~720 video-hours per GPU-hour | ~14 GPU-hours per epoch |
| Storage of crops (estimate) | — | ~0.2–0.5 TB (1.7 TB free) |

See [Is this machine enough?](#is-this-machine-enough) for run lengths and rental cost.

**Phone compute:** the 30M model needs 16.7 GFLOPs per second of video, and the 10M model needs 4.7. Both look affordable for a recent phone GPU over WebGPU, but that hasn't been measured on a phone. A WebGPU benchmark is the first phone step.

## Is this machine enough?

For the pilot and all data preparation, yes. For the full-size model, no.

- **Real training speed is not yet known.** Encoder-only synthetic batches ran at 670 video-hours per GPU-hour. The first real batches ran at about 150 while another job held the same GPU at 95%. Expect 300–600 alone. Read the true figure from `video_hours_per_hour` in `data/train/runs/pilot/train.jsonl`.
- **Pilot** (2,500 hours × 20 passes = 50,000 video-hours): 3.5–7 days on this GPU alone, about two weeks if the GPU stays shared.
- **One full-size run** (10,000 hours × about 30 passes, plus a Whisper-feature pretraining stage of similar size, about 600,000 video-hours): 1,000–2,000 GPU-hours. That is 1.5–3 months on this machine, or about a week on 8 rented GPUs of the same class at roughly $2.5–5k.
- **Whole programme** (three full-size runs and smaller ablations): 4,000–6,000 GPU-hours, about $9–15k rented at $2.50 per GPU-hour. The earlier $3–6k figure left out the pretraining stage and assumed the synthetic speed.

Rent GPUs only if the pilot shows a small model is worth scaling.

**Storage.** The pilot's clips take about 110 GB once converted (AVYT 47 GB, VoxCeleb2 64 GB). At the same encoding, 10,000 hours take 350–450 GB. Never keep source video: convert each file as it downloads and delete it. The machine has a 4 TB disk; move finished shards to object storage if free space drops below about 500 GB.

## Training data

Surveyed 2026-11-26. Hours are English speech unless noted. "Checked" means the files or links were opened from this machine that day.

### Ready to download (real video files)

| Source | Hours | What it is | Terms | Checked |
| --- | --- | --- | --- | --- |
| [AVYT](https://huggingface.co/datasets/nguyenvulebinh/AVYT) `avyt/talking-train` | ~1,450 | 96×96 colour mouth clips at 25 fps with audio and transcripts, cut from in-the-wild YouTube. FLARE trained on this set. | No license stated; source videos belong to their uploaders | Ungated; 544 GB; a shard opened and played |
| Same repo, `vox2/` | ~1,100–1,300 | VoxCeleb2 (celebrity interviews) English clips in the same format | As above | Ungated; 64 GB; a shard opened |
| [Seamless Interaction](https://huggingface.co/datasets/facebook/seamless-interaction) (Meta) | ~7,000 on camera, roughly half of it speech | 4,000+ recruited participants in two-person conversations. One 1080p–4K full-body video per person, with word-timed transcripts. Faces are 160–600 px wide. | CC BY-NC 4.0 | Ungated; video and transcript fetch per file; ~17 TB of video, so crop while streaming |
| [AVSpeech mirror](https://huggingface.co/datasets/ProgramComputer/avspeech-visual-audio) | ~1,550 (4,700 all languages) | YouTube clips, 3–10 s each; no transcripts | Gated (auto-approve); no license stated | Listing only; 2.9 TB |
| [LRS3 mirror](https://huggingface.co/datasets/TheNHz/ellipsis-lrs3-raw) | ~430 | TED talks with human transcripts. The official Oxford download is gone (page 404; the KAIST mirror says downloads ended). | Gated (auto-approve); unofficial and about 13% of clips missing | Listing only; 150 GB |

### YouTube lists (ids, timestamps and labels; the video must be fetched from YouTube)

| Source | Hours | Notes |
| --- | --- | --- |
| [MultiVSR](https://huggingface.co/datasets/sindhuhegde/multivsr) (Oxford, 2025) | 5,859 (12k all languages) | Full videos behind the AVSpeech ids, filtered to a visible, in-sync speaker, with Whisper large-v3 word timings. Annotations are MIT. 284 of 300 sampled English videos still exist (95%). Overlaps AVSpeech. |
| [VoxBlink2](https://voxblink2.github.io) | Part of 16,672 | 111k speakers in 18 languages, English the largest. No transcripts. CC BY-NC-SA. |
| [SpeakerVid-5M](https://huggingface.co/datasets/dorni/SpeakerVid-5M-Dataset) | Part of 8,743 | Interviews, news and talk shows, 83k speakers, Whisper transcripts. Many half-body and side views. Research only. |
| [TalkVerse](https://huggingface.co/datasets/zhenzhiwang/TalkVerse) | Part of 6,300 | Single-person clips with ASR text. Gated; license "other". |
| [YouTube-Commons](https://huggingface.co/datasets/PleIAs/YouTube-Commons) | Not measured | Links and transcripts for 3.2M CC BY videos, 71% English. The only source here whose videos carry a reuse license. The share showing a usable talking face is unknown; measure it on a sample first. |

Not usable: LRS2 (BBC agreement, academic research only) and the official VoxCeleb2 site (it no longer serves the links).

**Rights.** Nothing at this scale is cleared for commercial use. Every set is research-only, non-commercial or unlicensed, and bulk downloading from YouTube conflicts with its terms of service. A model trained on them is a research prototype. A public release needs either YouTube-Commons-style CC BY video or licensed recordings.

**Reaching 10k hours.** The ready-to-download rows give about 8,000 hours. MultiVSR adds about 4,300 more after its AVSpeech overlap.

### Is 10k hours enough?

It's enough to learn whether a phone-sized model can get close to FLARE. It probably isn't enough to match it.

Published lip-only LRS3 results, all as cited in the [FLARE paper](https://arxiv.org/abs/2610.19559), Table 2:

| Model | Parameters | Training hours | LRS3 WER |
| --- | --- | --- | --- |
| Auto-AVSR | 186M | 1,902 → 3,448 | 23.5% → 19.1% |
| FLARE Base, small decoder | 117M | 1,759 → 4,966 | 23.4% → 18.5% |
| FLARE Large, small decoder | 342M | 4,966 | 15.9% |
| FLARE Large + 7B Qwen (the live app) | 342M + 7B | 4,966 | 13.2% |
| ViT3D (Google) | 135M | 90,000 | 17.0% |
| LP-Conformer (Google) | 410M | 100,000 | 12.8% |

- Doubling data from about 2k to 4–5k hours cut 4–5 points. The curve flattens after that: 90k hours with a 135M model reached only 17.0%.
- FLARE's training method was worth more than data. It pretrains the lipreader to predict Whisper's audio features and feeds both face and lip crops. Its 117M model on 5k hours (18.5%) is close to Google's 135M model on 90k hours (17.0%). Copy that method.
- No published result exists for a 10–30M streaming model. Our estimate, not a measurement: 17–22% on LRS3 at 10k hours with FLARE-style pretraining, against FLARE-LLM's 13.2%.
- Out-of-domain video is harder. FLARE-LLM scores 34.3% on WildVSR (YouTube clips unlike its training data). Variety of speakers, cameras and lighting matters more for webcam accuracy than raw hours.

Train at 2.5k, 5k and 10k hours and plot the curve. Mine more data from YouTube-Commons only if the curve is still falling at 10k.

## Phases

1. **Baseline.** Score FLARE on public held-out video: WildVSR (5 hours of YouTube clips), the LRS3 test set, and speakers held out from Seamless Interaction. Webcam sentences recorded in the app ([EVALUATION.md](EVALUATION.md)) are an optional extra.
2. **Pilot, about 2,500 hours, local (started 2026-11-27).** Train a 37M model on the AVYT and VoxCeleb2 mouth clips, which need no cropping or labelling ([TRAINING.md](TRAINING.md)). Score it against FLARE.
3. **Scale to 5k, then 10k hours.** Add Seamless Interaction, then MultiVSR. Build the crop pipeline for them. Add FLARE-style Whisper-feature pretraining.
4. **Phone.** Use chunked streaming attention, int8 ONNX, WebGPU through ONNX Runtime Web, and MediaPipe in the browser. Measure real-time factor and battery on two phones.
5. **Optional.** Distill FLARE's visual-encoder features, and use a small language model for rescoring.

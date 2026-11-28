# Plan: a phone-sized lipreader

Status: **pipeline working; approach changed 2026-11-27 from training from scratch to distilling FLARE.** FLARE's encoder with a small text side and no LLM reads held-out speakers at 35.5% WER against FLARE's 31% ([results](TRAINING.md#results-so-far)); the smaller versions are still training. Nothing here is in the live app.

Goal: run lipreading in a phone browser in real time, with word accuracy equal to FLARE-LLM Large or slightly worse, measured on held-out video (see Phases).

## Approach

Distill FLARE's visual encoder into a small model ([TRAINING.md](TRAINING.md#how-it-learns)). Keep FLARE's 14M-parameter lip front end as it is, train a small Conformer to reproduce FLARE's encoder output frame by frame, and train a small text decoder on the transcripts at the same time. Don't scale FLARE up, and don't train from scratch.

- **From scratch was tried first and did not start.** Three runs on 2026-11-27 (text only; a long short-clip stage; audio targets) never got dev WER under 100% in up to 2,800 video-hours. FLARE's own encoder took 800,000 pretraining steps to learn to tell mouth shapes apart.
- **FLARE is a strong teacher on this data.** Lip stream only, it scored 31% WER on held-out VoxCeleb2 speakers and 10% on clips from its own training set (60 clips each, beam 5). The student sees the same video, so everything in the teacher's features is visible to it.
- **The LLM is left out.** About 95% of FLARE's parameters are its 7B Qwen decoder, which mostly supplies language knowledge. In the FLARE paper the same 342M visual model scores 15.9% LRS3 WER with a small 58M decoder and 13.2% with Qwen, so the LLM is worth under 3 points.
- **Labels come from audio**, already made for the pilot's data. For new video, Whisper transcribes the soundtrack. At run time the app still never uses audio.
- **FLARE can't label at scale.** It took 2.9 s of GPU time to decode a 4 s clip (beam 30), so labelling 10k hours with it would take about 7,200 GPU-hours. Its encoder features need no language-model decoding and cost about 1 GPU-hour per 1,000 hours of video.

Pipeline: video → mouth crops as FLARE's lip stream expects them (96×96, aligned as in `backend/preprocessing.py`) → Whisper labels → FLARE features, once → training from features → int8 export → WebGPU in the browser.

## Measured costs on this machine

All throughput numbers below come from this machine: an RTX PRO 6000 Blackwell (96 GB) and a 32-core Threadripper 9970X.

| Step | Measured | Per 10k hours of video |
| --- | --- | --- |
| Whisper large-v3 labels (batched, word timestamps) | 237× real time on 1 h of LibriSpeech audio | ~42 GPU-hours (~2 days locally) |
| MediaPipe face landmarks (CPU) | 150 fps per process; 1,620 fps with 32 processes | ~155 hours locally (~6.5 days) or ~1,700 vCPU-hours in the cloud (~$50–100) |
| FLARE features, once per clip (`extract_features.py`) | ~1,000× real time | ~10 GPU-hours |
| Training the 40M model from saved features | ~4,000 video-hours per GPU-hour, 4 GiB of GPU memory | ~2.5 GPU-hours per pass |
| Training the same model from video with FLARE in the loop (not used) | ~500 video-hours per GPU-hour | ~20 GPU-hours per pass |
| Storage of mouth clips | 46 MB per video-hour | ~0.4 TB |
| Storage of features | 131 MB per video-hour | ~1.3 TB |

See [Is this machine enough?](#is-this-machine-enough) for run lengths and rental cost.

**Phone compute (estimated from parameter counts):** FLARE's lip front end is about 15 GFLOPs per second of video, each of FLARE's encoder layers about 0.9, and the small Conformer about 1. The 355M model that keeps all 18 layers is about 32, the 40M one about 17. Whether a recent phone GPU sustains either over WebGPU hasn't been measured; a WebGPU benchmark is the first phone step, and a lighter front end is the largest saving available.

## Is this machine enough?

Yes, for everything up to and including a 10,000-hour run. Saving FLARE's features once is what changed this: a training pass no longer decodes video or runs a large model.

- **Pilot** (2,400 hours × 20 passes, about 48,000 video-hours): about 2.5 hours to extract features and about 12 hours to train, on this GPU.
- **One full-size run** (10,000 hours × 30 passes, about 300,000 video-hours): about 10 GPU-hours of extraction and about 75 of training. That is 3 to 4 days on this machine alone. Several runs in parallel fit on the one card, at 4 GiB each.
- **What a 10,000-hour run still needs:** the data. Downloading and cropping 7,600 more hours is CPU and network work (about 5 days of face tracking on this machine's 32 cores, by the MediaPipe measurement above). Their clips and features take about 1.35 TB, against about 1.3 TB free once the pilot's features are written, so finished clip shards have to move to object storage.
- **Rented GPUs** would only buy speed: 8 GPUs would cut a full-size run from days to hours, for a few hundred dollars. The earlier estimate of about 2,700 GPU-hours ($7k) was for training from scratch and no longer applies.

These figures are measured speeds. What accuracy the small model reaches is not known until the pilot finishes.

**Storage.** Never keep source video: convert each file as it downloads and delete it. The machine has a 4 TB disk; move finished clip shards to object storage once their features are extracted if free space drops below about 500 GB.

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
2. **Pilot, about 2,400 hours, local (started 2026-11-27).** Distill FLARE on the AVYT and VoxCeleb2 mouth clips, which need no cropping or labelling ([TRAINING.md](TRAINING.md)), at three sizes: all 18 of FLARE's encoder layers (355M parameters), the first six (146M) and none (40M). Score each against FLARE on held-out VoxCeleb2 speakers, where FLARE's lip-only WER is 31%.
3. **Shrink it.** The 355M model is already close to FLARE; the phone needs it smaller and cheaper per frame. Three levers, all started 2026-11-27 ([results](TRAINING.md#a-cheaper-front-end)): a front end that works from smaller frames (64×64 costs 2.25 times less for about 4 points before any adaptation), 8-bit weights (354 MB, no loss), and fewer FLARE layers (runs in progress). Next: adapt the layers behind a cheaper front end to its features, and try 48×48.
4. **Scale to 5k, then 10k hours** if the curves are still improving with data. Add Seamless Interaction, then MultiVSR. Build the crop pipeline for them; the crops must match what FLARE's lip stream expects.
5. **Phone.** Use chunked streaming attention, int8 ONNX, WebGPU through ONNX Runtime Web, and MediaPipe in the browser. Measure real-time factor and battery on two phones.
6. **Optional.** Use a small language model for rescoring.

# Plan: a phone-sized lipreader

Status: **proposal, measured 2026-11-26.** Nothing here is in the live app.

Goal: run lipreading in a phone browser in real time, with word accuracy equal to FLARE-LLM Large or slightly worse, measured on the held-out webcam set ([EVALUATION.md](EVALUATION.md)).

## Approach

Train a small streaming model from scratch. Don't scale FLARE up or post-train it.

- About 85% of FLARE's parameters are its 7B Qwen decoder, which mostly supplies language knowledge. Published visual-only conformers (a compact speech-model architecture) of a few hundred million parameters or less reached LRS3 WER in the low teens when trained on roughly 100k hours of video. A 250M model trained on 3.4k hours reached about 19%. These are published numbers, not reproduced here. Scaling data mattered more than model size.
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

A full run of about 50 epochs over 10k hours takes ~750 GPU-hours. That's about a month on the local GPU, or about 4 days on 8 rented H100s at roughly $2,250 (assuming $3/GPU-hour and at least local speed). At 3.5k hours it's ~260 GPU-hours (~$800). Real data loading will be slower than synthetic tensors.

**Compute budget for the 10k-hour route: about $3–6k**, covering several full runs plus smaller ablations. Labels and crops are a small part of that. The first pilot at 1k hours fits on the local GPU at no cost.

**Phone compute:** the 30M model needs 16.7 GFLOPs per second of video, and the 10M model needs 4.7. Both look affordable for a recent phone GPU over WebGPU, but that hasn't been measured on a phone. A WebGPU benchmark is the first phone step.

## The decision: training data

Data is the expensive, one-way choice.

| Source | Hours (English) | Terms |
| --- | --- | --- |
| LRS3 (TED) | ~430 | Research, non-commercial; access by request |
| VoxCeleb2 | ~2,400 | Annotations CC BY-SA; the videos belong to their YouTube owners |
| AVSpeech | ~4,700 | Annotations CC BY; the videos are YouTube links, and bulk download conflicts with YouTube's terms |
| Paid, consented recordings | Any | Clean rights; rough estimate $30–60 per hour of contributor time |

Recommendation: run a research prototype on LRS3 + VoxCeleb2 + AVSpeech, which is about 3.5k hours, the same mix Auto-AVSR used. It will show whether this approach can reach FLARE's webcam WER. Settle data rights before any public or commercial release.

## Phases

1. **Baseline.** Record the held-out sentences with several speakers. Score FLARE in live and model modes.
2. **Pilot, about 1k hours, local, about 2 weeks.** Build the crop and label pipeline. Train the 30M model and score it in model mode against FLARE.
3. **Scale to 3.5k hours, then 10k.** Add self-supervised pretraining if the pilot WER curve flattens.
4. **Phone.** Use chunked streaming attention, int8 ONNX, WebGPU through ONNX Runtime Web, and MediaPipe in the browser. Measure real-time factor and battery on two phones.
5. **Optional.** Distill FLARE's visual-encoder features, and use a small language model for rescoring.

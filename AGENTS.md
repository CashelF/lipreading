# Working on Still

Read [README.md](README.md) for the user flow, [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for checks, and [MODEL.md](MODEL.md) before changing inference. Run commands from the repository root.

## Code map

| Area | Entry points |
| --- | --- |
| React UI, camera, transcript | `src/App.tsx`, `src/capture.ts`, `src/styles.css` |
| Explicit recordings; timing UI | `src/SpeakingExamples.tsx`, `src/latency.ts`, `src/LatencyPanel.tsx` |
| HTTP/WebSocket lifecycle; preparation scheduling | `backend/app.py` |
| JPEG validation, timestamps, resampling | `backend/stream.py` |
| Utterance history, pause boundaries, revisions | `backend/utterance_stream.py`, `backend/utterance.py` |
| Detection, landmarks, alignment, shared caches | `backend/preprocessing.py` |
| Speaking scores and temporal admission | `backend/visual_speech.py`, `backend/speech_gate.py` |
| FLARE loading/search; hallucination checks | `backend/model.py`, `backend/decoding.py`, `backend/grounding.py`, `backend/repetition.py` |
| Upstream installation | `scripts/setup_model.py`, `scripts/setup_visual_speech.py` |

## Preserve these behaviors

- Audio is never captured or used. The browser sends timestamped JPEGs; no external video or transcript service is involved.
- Capture preserves native detail up to 1280×960, never upscales, and uses JPEG quality 0.92. **SFD detection stays bounded to 640×480**; map boxes back to original pixels for FAN and crops. After SFD input increased to 720p, a reported session was blocked by extra-face detections.
- Keep the 25 fps timestamp grid, alignment/normalization, BF16 weights, beam 30, and temperature 1.2 when making accuracy-preserving changes. A smaller beam, changed context, merged weights, or different gate can change predictions.
- The browser always enables speech gating and visual verification. Quiet/uncertain input holds earlier text. Do not bypass checks to make captions appear or suppress errors with canned transcripts.
- Each update replaces the current full-utterance draft. Pauses finalize once and preserve paragraphs. Do not join overlapping captions by fuzzy text matching, discard first/last words, or silently truncate long input.
- Cache only immutable observations/crops. Speech timestamps and state belong to each consumer. An overlapping window must not reuse old speech as new evidence.
- The removed early preview and extra decoder must stay out of the live path unless explicitly requested. `backend/preview_decoding.py` and `backend/decoder_graphs.py` are research utilities; old landmark activity modules are historical baselines.

## Checks and evidence

```bash
npm run build
.venv/bin/python -m pytest -q
```

The lightweight suite skips tests requiring Torch or pinned upstream sources. Use the model environment for decoder/model changes; see the development guide for adding test tooling without replacing its pinned NumPy/OpenCV stack. Test behavior and failures, not just implementation details.

GPU checks require an idle model: inspect `/api/status` (`ready` and `busy`) and `nvidia-smi` first. Do not stop unrelated GPU processes. Local replay scripts may depend on consented recordings and reports absent from a fresh clone. Do not create fake recordings to claim a real accuracy result.

Keep frames, captions and token IDs out of logs, reports, commits and external tools. Record only numeric timings, categorical failure reasons, and equality/presence flags. Tests with synthetic text are fine. Saved development examples are not held-out WER evidence; exact comparisons need identical input windows and awareness of thread-local cuDNN variation.

## Deployment and Git

- The configured remote is the destination for pushes requested by the user. Inspect its state before pushing; do not force-push or overwrite unrelated work.
- Keep `models/`, `vendor/`, virtual environments, local recordings, credentials and generated bundles out of Git. Installation scripts reproduce dependencies and weights.
- Follow [docs/OPERATIONS.md](docs/OPERATIONS.md) for a requested deployment. Check for active sessions before restarting and verify readiness afterward. Model loading takes time.
- For UI releases, build into `dist-next`, copy new assets first, retain existing hashed assets, then atomically replace `dist/index.html` so open browsers keep working.
- Keep the README short. Put setup and evaluation detail in linked guides; label historical experiments and avoid stale descriptions of removed controls.

# Development

## Lightweight UI/API setup

Use Node.js 22+, Python 3.10, and [uv](https://docs.astral.sh/uv/getting-started/installation/) (or an equivalent virtual environment with pip).

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
npm ci
bash scripts/dev.sh
```

Open **http://localhost:5173**. Vite proxies `/api` and WebSockets to port 8000. Camera preview works without weights; transcription remains unavailable until the real model is installed. `dev.sh` prefers `.venv-model` when present. Stop an existing backend before starting another on the same port.

For GPU inference, follow [MODEL.md](../MODEL.md#install). Do not install the lightweight OpenCV/NumPy requirements over the model environment's pinned versions.

## Checks

```bash
npm run build
.venv/bin/python -m pytest -q
```

Tests cover capture validation, sampling, face geometry, caches, speaking decisions, pause boundaries, revisions, transport and failure handling. Torch/source-dependent tests skip when dependencies are absent. The isolated test models never run in the live app.

For the full suite in the model environment, add only test tooling:

```bash
.venv-model/bin/python -m pip install 'pytest>=8,<10' 'httpx>=0.28,<1'
.venv-model/bin/python -m pytest -q
```

Frontend logic checks also run without a browser:

```bash
check_dir="$(mktemp -d)"
./node_modules/.bin/tsc src/capture.ts src/latency.ts --target ES2022 --module commonjs --outDir "$check_dir"
node scripts/check_capture.cjs "$check_dir/capture.js"
node scripts/check_browser_latency.cjs "$check_dir/latency.js"
```

For camera/UI changes, check Start/Stop, guidance recovery, mirroring, paragraphs, Copy/Save, and narrow layouts in a browser. Camera permission requires localhost or HTTPS. Confirm audio-free capture and both speech checks remain enabled.

## Real model evaluation

Automated tests do **not** measure word accuracy. Use real checkpoints and consented local video for inference checks. Do not run a competing session or GPU benchmark while someone is using the app.

The optional recorder is under **Help → Troubleshooting → Record examples**. It collects four eight-second conditions: quiet stillness, quiet movement, normal speech, and subtle speech. Examples stay in ignored `data/speaking-examples/`; the panel can delete them. No recordings or private reports are included in this repository.

```bash
.venv-model/bin/python scripts/evaluate_speaking_examples.py --take 1 --device cuda --output models/visual_speech/take1-report.json
.venv-model/bin/python scripts/evaluate_speaking_examples.py --take 2 --device cuda --output models/visual_speech/take2-report.json
.venv-model/bin/python scripts/check_speaking_gate.py --live --mixed
```

The live check requires a ready, idle localhost:8000 server and recorded examples. Offline gate checks additionally need saved reports and a matching frozen-selection report from the original experiment. Several benchmarks reference original recording IDs or ignored baseline snapshots: they are research tools, not fresh-clone smoke tests. Read their help and prerequisites first.

Reports omit captions and pixels. Development recordings test admission, continuity and regressions, not held-out word error rate. Resizing an old 640px recording tests cost, not the accuracy benefit of a native 720p camera.

## Architecture and research notes

`backend/app.py` owns a single reading/recording session. Frames are sampled onto a continuous 25 fps grid. The gate examines rolling context; admitted speech is decoded as a complete retained utterance and refined at a pause. The browser displays revisable text and committed paragraphs. See [MODEL.md](../MODEL.md) for preprocessing, thresholds and limitations.

[Archived HTML reports](../artifacts/README.md) describe dated experiments, not live telemetry, current features, or guaranteed performance.

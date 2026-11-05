# Still — camera to text

A local web app that reads visible speech with **FLARE-LLM Large**. Turn on your camera, speak, and see revisable captions. No microphone or external video API.

## Get started

You need **Node.js 22+**, **Python 3.10**, and an NVIDIA CUDA GPU. The runtime checks for at least **28 GiB of free VRAM** before loading; long phrases can need more.

1. [Install the model runtime and download its weights](MODEL.md#install).
2. From the repository root, run:

   ```bash
   npm ci
   npm run build
   .venv-model/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
   ```

3. Open **http://localhost:8000** and wait for **Ready to read**.

For access from another computer, use [HTTPS over your tailnet](docs/OPERATIONS.md#access-from-another-computer).

## Use it

- Select **Enable camera**, then **Start reading**. Keep one face visible, well lit, and facing the camera; speak naturally in English.
- Allow a few seconds for the first words. Recent text can change as more video arrives. Pauses finish statements and separate them into paragraphs.
- Select **Stop reading** and keep facing the camera until it finishes. Use **Copy** or **Save** to export the transcript.
- **Settings** contains mirror view and the maximum phrase length: 30 seconds by default, or 60. Sessions can continue across many pauses.

If nothing appears, check the guidance over the camera or below the panels. Move closer when asked. “More than one face detected” can also come from faces on pictures or screens. **Help → Troubleshooting** contains timing diagnostics and an optional example recorder.

## Know the limits

Lip reading is ambiguous: captions can be wrong, and very subtle speech can be missed. Review important text. This wraps an offline model with repeated updates, so it is not instantaneous. The authors' benchmark accuracy has **not** been reproduced on webcam input here.

Ordinary reading keeps camera frames in bounded memory. Only **Record examples** saves video locally, with an explicit recording action. Those recordings, downloaded models, and environments are excluded from Git. Speech detection and visual-evidence checks are always enabled in the browser app.

## More detail

- [Development and tests](docs/DEVELOPMENT.md)
- [Hosting, service management, and troubleshooting](docs/OPERATIONS.md)
- [Model setup, preprocessing, and evaluation notes](MODEL.md)
- [Agent working guide](AGENTS.md)

Built on [FLARE](https://github.com/jxzhanggg/FLARE), [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B), [face-alignment](https://github.com/1adrianb/face-alignment), and [Light-ASD](https://github.com/Junhua-Liao/Light-ASD). Upstream source and model weights retain their respective licenses and terms.

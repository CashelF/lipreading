# Running Still

## Local production server

Install the [model runtime](../MODEL.md#install), then:

```bash
npm ci
npm run build
.venv-model/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

The API serves `dist/` and loads the model in the background. At **http://localhost:8000/api/status**, `ready: true` means loading and warmup finished; `busy: true` means a reading or recording session owns the app. Initial loading can take roughly a minute on the development machine.

Only one session can use the backend at a time. This is a trusted, single-user app with origin checks but no public-user authentication. Keep the listener on loopback.

## Access from another computer

Use Tailscale Serve for HTTPS inside your tailnet. Replace the example hostname with your machine's actual tailnet hostname and use that exact origin when starting the backend:

```bash
LIPREAD_ALLOWED_ORIGINS=https://YOUR-MACHINE.YOUR-TAILNET.ts.net:8449 \
  .venv-model/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --no-access-log
```

In another terminal:

```bash
tailscale serve --bg --https=8449 http://127.0.0.1:8000
tailscale serve status
```

Open the resulting HTTPS URL from a device on the tailnet and grant camera permission. Use Serve, not public Funnel. `LIPREAD_ALLOWED_ORIGINS` accepts comma-separated exact origins; `.env` files are not loaded automatically. Remove only this app's proxy with `tailscale serve --https=8449 off`.

## Optional background service (Linux)

From the repository root, with the foreground backend stopped:

```bash
app_dir="$(pwd)"
tailnet_origin="https://YOUR-MACHINE.YOUR-TAILNET.ts.net:8449"
systemd-run --user --unit=still-lipreading \
  --property="WorkingDirectory=$app_dir" \
  --property=Restart=on-failure --property=RestartSec=5s \
  --setenv="LIPREAD_ALLOWED_ORIGINS=$tailnet_origin" \
  "$app_dir/.venv-model/bin/python" -m uvicorn backend.app:app \
  --host 127.0.0.1 --port 8000 --no-access-log
```

```bash
systemctl --user status still-lipreading
journalctl --user -u still-lipreading -f
systemctl --user restart still-lipreading
systemctl --user stop still-lipreading
```

This creates a **transient user service**, not a boot-enabled installation. Recreate it after reboot. For a lasting installation, install an equivalent user unit and configure user-session persistence for your host.

## Updating an existing deployment

Check `/api/status` before restarting; let an active session finish. Run relevant tests and build before publishing:

```bash
./node_modules/.bin/tsc -b
./node_modules/.bin/vite build --outDir dist-next
mkdir -p dist/assets
cp dist-next/assets/* dist/assets/
cp dist-next/index.html dist/index.html.next
mv dist/index.html.next dist/index.html
```

Retain older hashed assets for open browsers. Restart the backend when Python changes, and verify readiness, the served bundle, and real inference when model behavior changes. Plan combined API/UI changes so the published client is compatible with the running server. The proxy survives backend restarts.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Model unavailable | `/api/status` lists missing assets/packages or memory limits. Complete both setup scripts in MODEL.md. |
| Insufficient GPU memory | Check `nvidia-smi`. Loading requires 28 GiB free headroom. Free memory through the workload owner; the app never kills other processes. |
| No words | Read camera/session guidance. Tracking, speech admission or verification may be withholding text. Do not disable checks just to force output. |
| More than one face | Check people, pictures and screens in view. SFD uses a 640×480 detection view; mouth tracking retains native detail. |
| Move closer / lost tracking | Improve lighting, focus and framing. Larger images cannot recover detail absent from the original feed. |
| Slow first words | The initial window is about two seconds, followed by processing. See **Help → Troubleshooting → Update timing**. |
| Last words still changing | Stop collects 640 ms of extra context and finishes the utterance. Wait before leaving the camera. |
| Long phrase limit | Pause between statements, or select 60 seconds in Settings before starting. Long phrases can decode much more slowly. |
| Camera/connection denied remotely | Use HTTPS, allow camera access, and match `LIPREAD_ALLOWED_ORIGINS` to the frontend origin. |

Logs contain timing/gate diagnostics, not video or recognized words. Native 720p uses more bandwidth and host RAM; 2,001 cached RGB frames alone occupy about 5.2 GiB, plus JPEGs/crops. Caches release on session cleanup. Explicit eight-second recordings have a 128 MiB per-sample cap and remain local until deleted.

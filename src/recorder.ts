import { captureSize } from './capture';

// Shared capture loop for explicit local recordings (speaking examples and
// held-out sentences). Ordinary reading never uses this module.
export type RecordPhase = 'idle' | 'connecting' | 'countdown' | 'recording' | 'saving';
export type Recording = { stop: () => void; cancel: () => void; isSaving: () => boolean };
type Options = {
  video: HTMLVideoElement;
  path: string;
  start: Record<string, unknown>;
  maxMs: number;
  onPhase: (phase: RecordPhase, seconds: number) => void;
  onElapsed: (elapsedMs: number) => void;
  onSaved: (message: Record<string, unknown>) => void;
  onError: (detail: string) => void;
};

export function startRecording({ video, path, start, maxMs, onPhase, onElapsed, onSaved, onError }: Options): Recording | null {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d', { alpha: false });
  if (!context) { onError('Camera recording is unavailable in this browser.'); return null; }
  onPhase('connecting', 0);
  const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}${path}`);
  const timers = new Set<ReturnType<typeof setTimeout>>();
  let canceled = false;
  let capturing = false;
  let encoding = false;
  let finishing = false;
  let stopSent = false;
  let ready = false;
  let frameHandle = 0;
  let nextCapture = 0;
  let lastMediaTime = -1;
  const schedule = (callback: () => void, delay: number) => {
    const handle = setTimeout(() => { timers.delete(handle); if (!canceled) callback(); }, delay);
    timers.add(handle);
    return handle;
  };
  const cancelFrame = () => {
    if (video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(frameHandle);
    else cancelAnimationFrame(frameHandle);
  };
  const cleanup = () => {
    canceled = true; capturing = false; cancelFrame();
    timers.forEach(clearTimeout); timers.clear();
    socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null;
    socket.close();
  };
  const fail = (detail: string) => {
    if (canceled) return;
    cleanup(); onError(detail);
  };
  const sendStop = () => {
    if (canceled || encoding || stopSent) return;
    if (socket.readyState !== WebSocket.OPEN) { fail('The recording connection was lost. Please try again.'); return; }
    stopSent = true;
    socket.send(JSON.stringify({ type: 'stop' }));
    schedule(() => fail('Saving took too long. Close this panel and reopen it to check whether the recording was saved.'), 20000);
  };
  const finish = () => {
    if (!capturing) return;
    capturing = false; finishing = true; cancelFrame(); onPhase('saving', 0);
    // A JPEG already being encoded must precede the stop message on the socket.
    sendStop();
  };
  const frame = (now: number) => {
    if (canceled || !capturing) return;
    frameHandle = video.requestVideoFrameCallback ? video.requestVideoFrameCallback(frame) : requestAnimationFrame(frame);
    if (video.readyState < 2 || now < nextCapture || video.currentTime === lastMediaTime || encoding) return;
    lastMediaTime = video.currentTime;
    nextCapture = nextCapture ? nextCapture + 40 : now + 40;
    if (now - nextCapture > 40) nextCapture = now + 40;
    if (socket.readyState !== WebSocket.OPEN) { fail('The recording connection was lost. Please try again.'); return; }
    if (socket.bufferedAmount > 2 * 1024 * 1024) { fail('The connection could not keep up. Please try recording again.'); return; }
    const { width, height } = captureSize(video.videoWidth, video.videoHeight);
    if (!width || !height) return;
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    context.drawImage(video, 0, 0, width, height);
    encoding = true;
    canvas.toBlob(async (blob) => {
      try {
        if (canceled) return;
        if (!blob) { fail('Could not capture a camera frame. Please try again.'); return; }
        const jpeg = await blob.arrayBuffer();
        if (canceled) return;
        if (socket.readyState !== WebSocket.OPEN) { fail('The recording connection was lost. Please try again.'); return; }
        const packet = new Uint8Array(8 + jpeg.byteLength);
        new DataView(packet.buffer).setFloat64(0, now, true);
        packet.set(new Uint8Array(jpeg), 8);
        socket.send(packet);
      } catch { fail('Could not capture the camera frames. Please try again.'); }
      finally { encoding = false; if (finishing) sendStop(); }
    }, 'image/jpeg', 0.92);
  };
  const begin = () => {
    const began = performance.now();
    capturing = true; onPhase('recording', Math.ceil(maxMs / 1000));
    const tick = () => {
      if (!capturing) return;
      onElapsed(Math.min(maxMs, performance.now() - began));
      schedule(tick, 100);
    };
    tick();
    schedule(finish, maxMs);
    frameHandle = video.requestVideoFrameCallback ? video.requestVideoFrameCallback(frame) : requestAnimationFrame(frame);
  };
  const countdown = (remaining: number) => {
    if (remaining === 0) { begin(); return; }
    onPhase('countdown', remaining);
    schedule(() => countdown(remaining - 1), 1000);
  };
  const connectionTimer = schedule(() => fail('Could not start recording. Check the connection and try again.'), 15000);
  socket.onopen = () => socket.send(JSON.stringify({ type: 'start', ...start }));
  socket.onmessage = (event) => {
    let message: unknown;
    try { message = JSON.parse(event.data); } catch { fail('The recording server returned an invalid response.'); return; }
    if (!message || typeof message !== 'object') return;
    const result = message as Record<string, unknown>;
    if (result.type === 'ready' && !ready) {
      ready = true; clearTimeout(connectionTimer); timers.delete(connectionTimer); countdown(3);
    } else if (result.type === 'saved') {
      cleanup(); onSaved(result);
    } else if (result.type === 'error') fail(typeof result.detail === 'string' ? result.detail : 'Could not save the recording.');
  };
  socket.onerror = () => fail('Could not connect to the recording server. Please try again.');
  socket.onclose = () => fail('The recording connection closed before saving was confirmed. Reopen this panel to check saved recordings.');
  return { stop: finish, cancel: cleanup, isSaving: () => stopSent };
}

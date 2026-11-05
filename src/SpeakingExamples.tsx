import { useCallback, useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { Check, Circle, LoaderCircle, MicOff, Trash2, X } from 'lucide-react';
import { captureSize } from './capture';

const EXAMPLES = [
  { label: 'quiet_still', title: 'Quiet, relaxed', prompt: 'Relax your mouth and stay quiet. Blink and breathe normally; there is no need to freeze.' },
  { label: 'quiet_motion', title: 'Quiet, moving', prompt: 'Stay quiet while gently turning or nodding your head. Blink normally and keep your face visible.' },
  { label: 'speech_normal', title: 'Natural speech', prompt: 'Say a few sentences at your usual pace for the whole recording. Any words are fine.' },
  { label: 'speech_subtle', title: 'Subtle speech', prompt: 'Speak or mouth a few sentences with the small, natural movements you want the app to catch. Do not exaggerate.' },
] as const;
type Label = typeof EXAMPLES[number]['label'];
type Example = { id: string; label: Label; take: 1 | 2; frames: number; duration_seconds: number };
type Phase = 'idle' | 'connecting' | 'countdown' | 'recording' | 'saving';
type Props = { videoRef: RefObject<HTMLVideoElement | null>; cameraOn: boolean; onClose: () => void };

function isExample(value: unknown): value is Example {
  if (!value || typeof value !== 'object') return false;
  const item = value as Record<string, unknown>;
  return typeof item.id === 'string' && EXAMPLES.some((example) => example.label === item.label)
    && (item.take === 1 || item.take === 2) && typeof item.frames === 'number' && typeof item.duration_seconds === 'number';
}

export default function SpeakingExamples({ videoRef, cameraOn, onClose }: Props) {
  const [samples, setSamples] = useState<Example[]>([]);
  const [loading, setLoading] = useState(true);
  const [label, setLabel] = useState<Label>('quiet_still');
  const [take, setTake] = useState<1 | 2>(1);
  const [phase, setPhase] = useState<Phase>('idle');
  const [seconds, setSeconds] = useState(3);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [deleting, setDeleting] = useState<string | null>(null);
  const previewRef = useRef<HTMLVideoElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const runRef = useRef<{ cancel: () => void; isSaving: () => boolean } | null>(null);
  const active = phase !== 'idle';
  const selected = EXAMPLES.find((example) => example.label === label)!;
  const matching = samples.filter((sample) => sample.label === label && sample.take === take);
  const completeCount = EXAMPLES.reduce((total, example) => total + [1, 2].filter((round) => samples.some((sample) => sample.label === example.label && sample.take === round)).length, 0);

  const cancelRecording = useCallback(() => {
    runRef.current?.cancel();
    setPhase('idle');
    setNotice('Recording canceled. The unfinished example was discarded.');
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void fetch('/api/examples', { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('Could not load saved examples. Close this panel and try again.');
        const items: unknown = await response.json();
        if (!Array.isArray(items) || !items.every(isExample)) throw new Error('The server returned an invalid examples list.');
        setSamples(items);
      }).catch((cause) => {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Could not load examples.');
      }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => { controller.abort(); runRef.current?.cancel(); };
  }, []);

  useEffect(() => {
    const preview = previewRef.current;
    if (preview && cameraOn) {
      preview.srcObject = videoRef.current?.srcObject ?? null;
      void preview.play().catch(() => { /* The original camera preview remains available. */ });
    }
    // After stop is sent, saving no longer depends on the camera. Wait for its
    // acknowledgement instead of incorrectly reporting that a saved clip vanished.
    if (!cameraOn && runRef.current && !runRef.current.isSaving()) {
      cancelRecording();
      setError('The camera disconnected. Enable it again before recording.');
    }
    return () => { if (preview) preview.srcObject = null; };
  }, [cameraOn, videoRef, cancelRecording]);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();
    const listener = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && closeRef.current && !closeRef.current.disabled) onClose();
      if (event.key !== 'Tab') return;
      const controls = dialogRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), [tabindex="0"]');
      const first = controls?.[0];
      const last = controls?.[controls.length - 1];
      if (!first || !last) { event.preventDefault(); dialogRef.current?.focus(); return; }
      // Recording can disable the button that just had focus. Tab must stay in
      // this dialog even while every control except Cancel is unavailable.
      if (!Array.from(controls ?? []).includes(document.activeElement as HTMLElement)) {
        event.preventDefault(); (event.shiftKey ? last : first).focus(); return;
      }
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', listener);
    return () => { document.removeEventListener('keydown', listener); previous?.focus(); };
  }, [onClose]);

  const record = () => {
    const video = videoRef.current;
    if (runRef.current || !cameraOn || !video || video.readyState < 2 || matching.length || loading || deleting) return;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) { setError('Camera recording is unavailable in this browser.'); return; }
    setError(''); setNotice(''); setPhase('connecting'); setProgress(0);
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/examples`);
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
      runRef.current = null;
    };
    const fail = (detail: string) => {
      if (canceled) return;
      cleanup(); setPhase('idle'); setError(detail);
    };
    const sendStop = () => {
      if (canceled || encoding || stopSent) return;
      if (socket.readyState !== WebSocket.OPEN) { fail('The recording connection was lost. Please try again.'); return; }
      stopSent = true;
      socket.send(JSON.stringify({ type: 'stop' }));
      schedule(() => fail('Saving took too long. Close this panel and reopen it to check whether the example was saved.'), 20000);
    };
    const finish = () => {
      capturing = false; finishing = true; cancelFrame(); setPhase('saving'); setSeconds(0); setProgress(1);
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
      capturing = true; setPhase('recording'); setSeconds(8);
      const tick = () => {
        const elapsed = performance.now() - began;
        if (!capturing) return;
        setSeconds(Math.max(0, Math.ceil((8000 - elapsed) / 1000)));
        setProgress(Math.min(1, elapsed / 8000));
        schedule(tick, 100);
      };
      tick();
      schedule(finish, 8000);
      frameHandle = video.requestVideoFrameCallback ? video.requestVideoFrameCallback(frame) : requestAnimationFrame(frame);
    };
    const countdown = (remaining: number) => {
      if (remaining === 0) { begin(); return; }
      setPhase('countdown'); setSeconds(remaining);
      schedule(() => countdown(remaining - 1), 1000);
    };
    runRef.current = { cancel: cleanup, isSaving: () => stopSent };
    const connectionTimer = schedule(() => fail('Could not start recording. Check the connection and try again.'), 15000);
    socket.onopen = () => socket.send(JSON.stringify({ type: 'start', label, take }));
    socket.onmessage = (event) => {
      let message: unknown;
      try { message = JSON.parse(event.data); } catch { fail('The recording server returned an invalid response.'); return; }
      if (!message || typeof message !== 'object') return;
      const result = message as Record<string, unknown>;
      if (result.type === 'ready' && !ready) {
        ready = true; clearTimeout(connectionTimer); timers.delete(connectionTimer); countdown(3);
      } else if (result.type === 'saved' && isExample(message)) {
        if (message.label !== label || message.take !== take) { fail('The recording label did not match. Close this panel and check the saved examples.'); return; }
        cleanup(); setPhase('idle'); setSamples((current) => [...current.filter((sample) => sample.id !== message.id), message]);
        setNotice(`${selected.title}, take ${take}, saved on this machine. Select the next example when you are ready.`);
      } else if (result.type === 'error') fail(typeof result.detail === 'string' ? result.detail : 'Could not save the recording.');
    };
    socket.onerror = () => fail('Could not connect to the recording server. Please try again.');
    socket.onclose = () => fail('The recording connection closed before saving was confirmed. Reopen this panel to check saved examples.');
  };

  const deleteSample = async (sample: Example) => {
    if (active || deleting) return;
    setDeleting(sample.id); setError(''); setNotice('');
    try {
      const response = await fetch(`/api/examples/${encodeURIComponent(sample.id)}`, { method: 'DELETE' });
      if (!response.ok) throw new Error('Could not delete this example. Please try again.');
      setSamples((current) => current.filter((item) => item.id !== sample.id));
      setNotice('Example deleted. You can record that take again.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not delete this example.'); }
    finally { setDeleting(null); }
  };

  return <div className="modal-backdrop examples-backdrop">
    <section ref={dialogRef} tabIndex={-1} className="examples-modal" role="dialog" aria-modal="true" aria-labelledby="examples-title">
      <div className="examples-heading"><div><div className="eyebrow">A BETTER SPEECH FILTER</div><h2 id="examples-title">Show us quiet. Show us speech.</h2></div><button ref={closeRef} className="icon-button" onClick={onClose} disabled={active || !!deleting} aria-label="Close examples"><X size={20} /></button></div>
      <p className="examples-intro">Record four short examples, then a second take to check the result. About 90 seconds in all. These help us test the filter; recording them does not change it automatically.</p>
      <div className="examples-layout">
        <div className="examples-capture">
          <div className="examples-preview"><video ref={previewRef} muted playsInline aria-label="Example recording camera preview" />{!cameraOn && <span>Enable your camera in the studio first.</span>}{(phase === 'countdown' || phase === 'recording') && <div className={`examples-timer ${phase === 'recording' ? 'recording' : ''}`}><strong>{seconds}</strong><span>{phase === 'countdown' ? 'GET READY' : 'RECORDING'}</span></div>}<span className="examples-audio"><MicOff size={12} /> No audio</span></div>
          <div className="examples-prompt"><span className="info-label">{selected.title.toUpperCase()} · TAKE {take}</span><p>{selected.prompt}</p>{label.startsWith('speech_') && <small>Keep going until the timer finishes. Short pauses between phrases are fine.</small>}</div>
          <progress className="examples-progress" max={1} value={progress} aria-label="Recording progress" />
          <div className="examples-record-actions">{active ? <><span className="examples-phase" role="status">{phase === 'connecting' ? 'Connecting…' : phase === 'countdown' ? `Starting in ${seconds}…` : phase === 'recording' ? `${seconds} seconds remaining` : 'Saving on this machine…'}</span><button className="examples-secondary" onClick={cancelRecording} disabled={phase === 'saving'}>Cancel</button></> : <><span className="examples-phase">{matching.length ? 'This take is saved.' : '3-second countdown · 8-second clip'}</span><button className="read-button" onClick={record} disabled={!cameraOn || loading || !!deleting || matching.length > 0}><Circle size={12} fill="currentColor" />Record example</button></>}</div>
        </div>
        <div className="examples-choices">
          <div className="examples-takes" role="group" aria-label="Recording take"><button aria-pressed={take === 1} disabled={active || !!deleting} onClick={() => { setTake(1); setNotice(''); }}>Take 1 <span>For tuning</span></button><button aria-pressed={take === 2} disabled={active || !!deleting} onClick={() => { setTake(2); setNotice(''); }}>Take 2 <span>For checking</span></button></div>
          <div className="examples-list">{EXAMPLES.map((example, index) => {
            const saved = samples.filter((sample) => sample.label === example.label && sample.take === take);
            return <div className={`examples-item ${label === example.label ? 'selected' : ''}`} key={example.label}><button className="examples-select" aria-pressed={label === example.label} disabled={active || !!deleting} onClick={() => { setLabel(example.label); setNotice(''); }}><span className={`examples-number ${saved.length ? 'saved' : ''}`}>{saved.length ? <Check size={14} /> : index + 1}</span><span>{example.title}<small>{saved.length ? 'Saved on this machine' : '8 seconds'}</small></span></button>{saved.map((sample) => <button key={sample.id} className="icon-button" disabled={active || !!deleting} onClick={() => void deleteSample(sample)} aria-label={`Delete ${example.title.toLowerCase()}, take ${take}`} title="Delete this example">{deleting === sample.id ? <LoaderCircle size={15} className="spin" /> : <Trash2 size={15} />}</button>)}</div>;
          })}</div>
          <p className="examples-count" role="status">{loading ? 'Loading saved examples…' : `${completeCount} of 8 examples saved`}</p>
          {completeCount === 8 && <p className="examples-complete"><Check size={15} /> All set. Let me know in chat so I can check the detector against your examples.</p>}
        </div>
      </div>
      {error && <p className="examples-error" role="alert">{error}</p>}
      {notice && <p className="examples-notice" role="status">{notice}</p>}
      <p className="examples-privacy">Only clips you explicitly record here are saved. They stay on this inference machine until you delete them; no microphone is used. Ordinary reading still keeps video in memory only.</p>
    </section>
  </div>;
}

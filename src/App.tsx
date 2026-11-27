import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ArrowDownToLine, ArrowUpRight, Camera, CameraOff, Check, ChevronDown, CircleHelp, Copy, Expand, Focus, LoaderCircle, MicOff, Play, Settings2, ShieldCheck, Square, Trash2, X } from 'lucide-react';
import SpeakingExamples from './SpeakingExamples';
import EvalSentences from './EvalSentences';
import LatencyPanel from './LatencyPanel';
import { LatencyTracker, type LatencySample } from './latency';
import { captureSize } from './capture';

type ModelStatus = { ready: boolean; model: string; detail: string; device?: string; loading?: boolean; state?: string; max_utterance_seconds?: number; timing_probes?: boolean };
type Transcript = { session_id: number; text: string; stable_text: string; utterances: string[]; draft_text: string; revision: number; final: boolean; latency_ms: number; timestamp: string };
type Mode = 'idle' | 'connecting' | 'reading' | 'stopping';
type ActivityState = 'still' | 'motion' | 'uncertain' | 'bypassed';
type Activity = { segment_id: number; state: ActivityState; suppress: boolean; detail: string; camera_hint?: 'move_closer' | 'face_not_visible' | 'multiple_faces' };
type UtteranceState = 'draft' | 'refining' | 'committed' | 'limit';
type PendingFrame = { session: number; socket: WebSocket; settled: Promise<void>; settle: () => void };
const ACTIVITY_LABELS: Record<ActivityState, string> = { still: 'Waiting for speech', motion: 'Visual speech detected', uncertain: 'Checking for speech', bypassed: 'Speech filter off' };
const UTTERANCE_STORAGE_KEY = 'still.utteranceSeconds';
function savedUtteranceSeconds(): number {
  try { return window.localStorage.getItem(UTTERANCE_STORAGE_KEY) === '60' ? 60 : 30; }
  catch { return 30; }
}
const INITIAL_STATUS: ModelStatus = { ready: false, model: 'FLARE-LLM Large', detail: 'Connecting to the inference server…' };

export default function App() {
  const [model, setModel] = useState<ModelStatus>(INITIAL_STATUS);
  const [retrying, setRetrying] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState('');
  const [mode, setMode] = useState<Mode>('idle');
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [quality, setQuality] = useState('Enable your camera, then select Start reading.');
  const [utteranceSeconds, setUtteranceSeconds] = useState(savedUtteranceSeconds);
  const [activity, setActivity] = useState<Activity | null>(null);
  const [utteranceState, setUtteranceState] = useState<UtteranceState | null>(null);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [recorder, setRecorder] = useState<'examples' | 'sentences' | null>(null);
  const showExamples = recorder !== null;
  const [mirrored, setMirrored] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const [latencySamples, setLatencySamples] = useState<LatencySample[]>([]);
  const latencyTracker = useRef(new LatencyTracker());
  const pendingDisplay = useRef<{ session: number; sample: LatencySample } | null>(null);
  const pingTimer = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const paintTimers = useRef<number[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const followTranscriptRef = useRef(true);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const readingRef = useRef(false);
  const sessionRef = useRef(0);
  const cameraRequestRef = useRef(0);
  const sessionPrefix = useRef(0);
  const stopTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const contextTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const frameDrainTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const pendingFrameRef = useRef<PendingFrame | null>(null);
  const drainingSessionRef = useRef<number | null>(null);
  const stopRequested = useRef(false);
  const transcriptText = transcripts.map((transcript) => transcript.text).filter(Boolean).join('\n\n');
  const busy = mode !== 'idle';
  const closeExamples = useCallback(() => setRecorder(null), []);
  const activityLabel = activity ? ACTIVITY_LABELS[activity.state] : null;

  useEffect(() => {
    try { window.localStorage.setItem(UTTERANCE_STORAGE_KEY, String(utteranceSeconds)); } catch { /* Optional preference storage. */ }
  }, [utteranceSeconds]);

  const refreshStatus = useCallback(async (signal?: AbortSignal) => {
    try {
      const response = await fetch('/api/status', { signal });
      if (!response.ok) throw new Error('Server unavailable');
      const status = await response.json() as ModelStatus;
      setModel(status);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === 'AbortError') return;
      setModel({ ...INITIAL_STATUS, state: 'unavailable', detail: 'Cannot reach the server. Check the connection and try again.' });
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refreshStatus(controller.signal);
    const interval = setInterval(() => void refreshStatus(controller.signal), 10000);
    return () => { controller.abort(); clearInterval(interval); };
  }, [refreshStatus]);

  const retryModel = async () => {
    setRetrying(true);
    try {
      const response = await fetch('/api/model/load', { method: 'POST' });
      if (!response.ok) throw new Error('Model request failed');
      await refreshStatus();
    } catch { setError('Could not load the model. Check that the inference server is running.'); }
    finally { setRetrying(false); }
  };

  useEffect(() => {
    if (!showHelp) return;
    const previous = document.activeElement as HTMLElement | null;
    const listener = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setShowHelp(false);
      if (event.key !== 'Tab') return;
      const controls = Array.from(document.querySelectorAll<HTMLElement>('.help-modal button, .help-modal a, .help-modal summary')).filter((element) => element.getClientRects().length > 0);
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', listener);
    return () => { document.removeEventListener('keydown', listener); previous?.focus(); };
  }, [showHelp]);

  const endSession = useCallback(() => {
    readingRef.current = false;
    clearInterval(pingTimer.current);
    clearTimeout(stopTimer.current);
    clearTimeout(contextTimer.current);
    clearTimeout(frameDrainTimer.current);
    drainingSessionRef.current = null;
    pendingFrameRef.current?.settle();
    pendingFrameRef.current = null;
    stopRequested.current = false;
    setMode('idle');
    const socket = socketRef.current;
    socketRef.current = null;
    socket?.close();
    // An interrupted session keeps its last text, but no longer shows a live cursor.
    const endedSession = sessionPrefix.current;
    setTranscripts((current) => current.map((transcript) => transcript.session_id === endedSession
      ? { ...transcript, final: true } : transcript));
  }, []);

  const stopReading = useCallback(() => {
    if (stopRequested.current) return;
    stopRequested.current = true;
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      setMode('stopping');
      const stoppingSession = sessionRef.current;
      const finishCapture = async () => {
        readingRef.current = false;
        drainingSessionRef.current = stoppingSession;
        const pending = pendingFrameRef.current;
        if (pending?.session === stoppingSession && pending.socket === socket) {
          let deadline: ReturnType<typeof setTimeout> | undefined;
          try {
            await Promise.race([pending.settled, new Promise<never>((_, reject) => {
              deadline = setTimeout(() => reject(new Error('Camera encoding timed out')), 2000);
              frameDrainTimer.current = deadline;
            })]);
          } catch {
            if (socket === socketRef.current && stoppingSession === sessionRef.current) {
              setError('The last camera frame could not finish encoding. The session ended with your existing transcript kept.');
              endSession();
            }
            return;
          } finally {
            clearTimeout(deadline);
            if (frameDrainTimer.current === deadline) frameDrainTimer.current = undefined;
          }
        }
        // A previous session's delayed JPEG must never send a stop to a new one.
        if (socket !== socketRef.current || stoppingSession !== sessionRef.current) return;
        drainingSessionRef.current = null;
        if (socket.readyState !== WebSocket.OPEN) { endSession(); return; }
        socket.send(JSON.stringify({ type: 'stop' }));
        clearTimeout(stopTimer.current);
        stopTimer.current = setTimeout(() => { setError('The model took too long to finish. Your existing transcript has been kept.'); endSession(); }, 120000);
      };
      // The speech detector needs 480 ms of future image context. Keep the
      // camera running briefly so speech at the Stop click can be evaluated.
      if (readingRef.current && videoRef.current?.readyState === 4) {
        contextTimer.current = setTimeout(() => void finishCapture(), 640);
      } else void finishCapture();
    } else endSession();
  }, [endSession]);

  const releaseCamera = useCallback(() => {
    ++cameraRequestRef.current;
    streamRef.current?.getTracks().forEach((track) => { track.onended = null; track.stop(); });
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
  }, []);

  const enableCamera = useCallback(async (selectedId = deviceId) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access needs a supported browser on localhost or an HTTPS connection.');
      return;
    }
    const request = ++cameraRequestRef.current;
    setCameraLoading(true);
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 25, max: 30 }, ...(selectedId ? { deviceId: { exact: selectedId } } : { facingMode: 'user' }) },
      });
      if (request !== cameraRequestRef.current) { stream.getTracks().forEach((track) => track.stop()); return; }
      streamRef.current?.getTracks().forEach((track) => { track.onended = null; track.stop(); });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
      if (request !== cameraRequestRef.current) { stream.getTracks().forEach((track) => track.stop()); return; }
      setCameraOn(true);
      const available = await navigator.mediaDevices.enumerateDevices();
      if (request !== cameraRequestRef.current) return;
      setDevices(available.filter((device) => device.kind === 'videoinput'));
      const settings = stream.getVideoTracks()[0].getSettings();
      if (settings.deviceId) setDeviceId(settings.deviceId);
      stream.getVideoTracks()[0].onended = () => { setError('The camera was disconnected. Reconnect it to continue.'); stopReading(); releaseCamera(); };
    } catch (cause) {
      if (request !== cameraRequestRef.current) return;
      const name = cause instanceof DOMException ? cause.name : '';
      setError(name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access in your browser, then try again.' : name === 'NotFoundError' ? 'No camera was found. Connect a camera and try again.' : name === 'NotReadableError' ? 'The camera is in use by another app. Close it there and try again.' : 'Could not start this camera. Check your browser permissions and camera connection.');
    } finally { if (request === cameraRequestRef.current) setCameraLoading(false); }
  }, [deviceId, releaseCamera, stopReading]);

  useEffect(() => {
    if (!cameraOn) return;
    const video = videoRef.current!;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) return;
    let canceled = false;
    let nextCapture = 0;
    let lastMediaTime = -1;
    let frameHandle = 0;
    const frame = (now: number) => {
      if (canceled) return;
      frameHandle = video.requestVideoFrameCallback ? video.requestVideoFrameCallback(frame) : requestAnimationFrame(frame);
      if (video.readyState < 2 || now < nextCapture || video.currentTime === lastMediaTime) return;
      lastMediaTime = video.currentTime;
      nextCapture = nextCapture ? nextCapture + 40 : now + 40;
      if (now - nextCapture > 40) nextCapture = now + 40;
      const socket = socketRef.current;
      if (!readingRef.current || !socket || socket.readyState !== WebSocket.OPEN || pendingFrameRef.current) return;
      if (socket.bufferedAmount > 2 * 1024 * 1024) { setQuality('Connection is catching up. Keep your face still and well lit.'); return; }
      const { width, height } = captureSize(video.videoWidth, video.videoHeight);
      if (!width || !height) return;
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      context.drawImage(video, 0, 0, width, height);
      const session = sessionRef.current;
      let settle!: () => void;
      const settled = new Promise<void>((resolve) => { settle = resolve; });
      const pending: PendingFrame = { session, socket, settled, settle };
      pendingFrameRef.current = pending;
      const completeFrame = () => {
        if (pendingFrameRef.current === pending) pendingFrameRef.current = null;
        settle();
      };
      const canSend = () => session === sessionRef.current && socket === socketRef.current
        && socket.readyState === WebSocket.OPEN && (!canceled || stopRequested.current)
        && (readingRef.current || drainingSessionRef.current === session);
      const encodingFailed = () => {
        if (socket === socketRef.current && session === sessionRef.current) {
          setError('A camera frame could not be encoded. Restart reading to continue; your existing transcript has been kept.');
          endSession();
        }
      };
      try { canvas.toBlob(async (blob) => {
        try {
          if (!canSend()) return;
          if (!blob) throw new Error('No camera frame was encoded');
          const jpeg = await blob.arrayBuffer();
          if (!canSend()) return;
          const packet = new Uint8Array(8 + jpeg.byteLength);
          new DataView(packet.buffer).setFloat64(0, now, true);
          packet.set(new Uint8Array(jpeg), 8);
          latencyTracker.current.frame(now, performance.now());
          socket.send(packet);
        } catch { encodingFailed(); }
        finally { completeFrame(); }
      }, 'image/jpeg', 0.92); }
      catch { encodingFailed(); completeFrame(); }
    };
    frameHandle = video.requestVideoFrameCallback ? video.requestVideoFrameCallback(frame) : requestAnimationFrame(frame);
    return () => { canceled = true; if (video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(frameHandle); else cancelAnimationFrame(frameHandle); };
  }, [cameraOn, endSession]);

  useEffect(() => {
    if (mode !== 'reading') return;
    const interval = setInterval(() => setElapsed((value) => value + 1), 1000);
    return () => clearInterval(interval);
  }, [mode]);

  useEffect(() => {
    // Avoid a queued scrolling animation pulling the reader back down after
    // they have scrolled up to review an earlier part of the transcript.
    if (followTranscriptRef.current && transcriptRef.current) transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
  }, [transcripts]);

  useLayoutEffect(() => {
    const pending = pendingDisplay.current;
    if (!pending) return;
    pendingDisplay.current = null;
    for (const id of paintTimers.current) cancelAnimationFrame(id);
    // The second rAF runs after the first paint opportunity following React's
    // commit. This is an estimate, not a hardware display timestamp.
    const first = requestAnimationFrame(() => {
      const second = requestAnimationFrame(() => {
        if (pending.session !== sessionRef.current || document.visibilityState !== 'visible') return;
        latencyTracker.current.displayed(pending.sample, performance.now());
        setLatencySamples(latencyTracker.current.samples);
      });
      paintTimers.current = [second];
    });
    paintTimers.current = [first];
  }, [transcripts]);

  useEffect(() => () => {
    ++cameraRequestRef.current;
    clearInterval(pingTimer.current);
    for (const id of paintTimers.current) cancelAnimationFrame(id);
    readingRef.current = false;
    clearTimeout(stopTimer.current);
    clearTimeout(contextTimer.current);
    clearTimeout(frameDrainTimer.current);
    drainingSessionRef.current = null;
    pendingFrameRef.current?.settle();
    pendingFrameRef.current = null;
    socketRef.current?.close();
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  const startReading = () => {
    if (!model.ready || !cameraOn || busy || showExamples) return;
    setError('');
    setMode('connecting');
    setElapsed(0);
    setActivity(null);
    setUtteranceState(null);
    latencyTracker.current.reset();
    pendingDisplay.current = null;
    setLatencySamples([]);
    for (const id of paintTimers.current) cancelAnimationFrame(id);
    stopRequested.current = false;
    followTranscriptRef.current = true;
    ++sessionRef.current;
    sessionPrefix.current += 100000;
    const prefix = sessionPrefix.current;
    let latestRevision = -1;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${window.location.host}/api/stream`);
    socketRef.current = socket;
    socket.onopen = () => {
      if (socket !== socketRef.current) return;
      socket.send(JSON.stringify({ type: 'start', speech_gate: true, verify_visual: true, max_utterance_seconds: utteranceSeconds }));
      readingRef.current = true;
      const ping = () => {
        if (!model.timing_probes || socket !== socketRef.current || socket.readyState !== WebSocket.OPEN || stopRequested.current) return;
        const now = performance.now();
        latencyTracker.current.ping(now);
        socket.send(JSON.stringify({ type: 'ping', client_time_ms: now }));
      };
      ping();
      pingTimer.current = setInterval(ping, 5000);
      setMode('reading');
      setQuality('Speak naturally in short phrases, pausing briefly between them.');
    };
    socket.onmessage = (event) => {
      if (socket !== socketRef.current) return;
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      if (message.type === 'transcript') {
        if (typeof message.text !== 'string' || typeof message.stable_text !== 'string' || typeof message.draft_text !== 'string'
            || !Number.isSafeInteger(message.revision) || message.revision <= latestRevision) return;
        latestRevision = message.revision;
        const timing = latencyTracker.current.receive(message, performance.now());
        if (timing) pendingDisplay.current = { session: sessionRef.current, sample: timing };
        setTranscripts((current) => {
          // Each message replaces the current session snapshot. The backend
          // replaces the active utterance draft; appending updates would repeat words.
          const stable = message.stable_text.trim();
          const draft = message.draft_text.trim();
          // Preserve explicit pause boundaries, including identical statements.
          // Older servers can still supply the flat stable-text field.
          const utterances: string[] = Array.isArray(message.utterances)
            && message.utterances.every((part: unknown) => typeof part === 'string' && part.trim())
            && message.utterances.join(' ') === stable
            ? message.utterances : stable ? [stable] : [];
          const text = [...utterances, draft].filter(Boolean).join('\n\n');
          if (!text) return current.filter((transcript) => transcript.session_id !== prefix);
          const previous = current.find((transcript) => transcript.session_id === prefix);
          const result: Transcript = { session_id: prefix, text, stable_text: stable, utterances, draft_text: draft, revision: message.revision,
            final: message.final === true, latency_ms: message.latency_ms,
            timestamp: previous?.timestamp ?? new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) };
          return previous ? current.map((transcript) => transcript.session_id === prefix ? result : transcript) : [...current, result];
        });
      } else if (message.type === 'pong') {
        latencyTracker.current.pong(message.client_time_ms, performance.now());
      } else if (message.type === 'activity') {
        if (typeof message.state === 'string' && Object.hasOwn(ACTIVITY_LABELS, message.state)) {
          setActivity({ segment_id: prefix + message.segment_id, state: message.state as ActivityState, suppress: message.suppress === true, detail: typeof message.detail === 'string' ? message.detail : '', camera_hint: ['move_closer', 'face_not_visible', 'multiple_faces'].includes(message.camera_hint) ? message.camera_hint : undefined });
        }
      } else if (message.type === 'utterance') {
        if (['draft', 'refining', 'committed', 'limit'].includes(message.state)) {
          setUtteranceState(message.state as UtteranceState);
          if (typeof message.detail === 'string') setQuality(message.detail);
        }
      } else if (message.type === 'error') { setError(message.detail); endSession(); }
      else if (message.type === 'quality' || message.type === 'status') { if (typeof message.detail === 'string' && !message.detail.includes('fps captured')) setQuality(message.detail.replaceAll('utterance', 'sentence').replaceAll('Utterance', 'Sentence')); }
      else if (message.type === 'stopped') { endSession(); setQuality('Session complete. Your transcript stays here until you clear it.'); }
    };
    socket.onerror = () => { if (socket === socketRef.current) setError('Could not connect to the inference server. Check that the backend is running.'); };
    socket.onclose = () => {
      if (socket !== socketRef.current) return;
      if (readingRef.current) setError('Connection to the model was lost. Your existing transcript has been kept.');
      endSession();
      void refreshStatus();
    };
  };

  const copyTranscript = async () => {
    try { await navigator.clipboard.writeText(transcriptText); setCopied(true); setTimeout(() => setCopied(false), 1800); }
    catch { setError('Clipboard access was unavailable. Use Download to save your transcript.'); }
  };
  const downloadTranscript = () => {
    const url = URL.createObjectURL(new Blob([transcriptText], { type: 'text/plain;charset=utf-8' }));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = `still-transcript-${new Date().toISOString().slice(0, 10)}.txt`; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const toggleCamera = () => {
    if (cameraOn) { if (busy) stopReading(); releaseCamera(); }
    else void enableCamera();
  };
  const duration = `${Math.floor(elapsed / 60).toString().padStart(2, '0')}:${(elapsed % 60).toString().padStart(2, '0')}`;
  const cameraHint = cameraOn && mode === 'reading' && activity?.state === 'uncertain' ? activity.camera_hint : undefined;

  const feedback = !model.ready ? model.detail
    : !cameraOn ? 'Enable your camera, then select Start reading.'
    : mode === 'stopping' ? 'Finishing your last words…'
    : !busy ? transcripts.length ? 'Your transcript is ready to copy or save.' : 'Ready when you are. Select Start reading.'
    : activity?.state === 'uncertain' ? activity.detail
    : activity?.state === 'still' ? 'Waiting for speech. Speak when you’re ready.'
    : quality;

  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="/" aria-label="Still home"><span className="brand-mark"><i /><i /><i /></span><span>still<span className="brand-period">.</span></span></a>
        <span className="header-divider" /><span className="header-label">Camera to text</span>
        <div className="header-actions">
          <span className="private-badge"><ShieldCheck size={14} /> No microphone needed</span>
          <button className="icon-button help-button" onClick={() => setShowHelp(true)} aria-label="Help"><CircleHelp size={20} /></button>
        </div>
      </header>

      <main>
        <div className="page-heading">
          <div><div className="eyebrow"><span /> LIVE LIP READING</div><h1>Turn speech into text.</h1><p>Face the camera, speak naturally, and pause between sentences.</p></div>
          <div className="model-pill"><span className={`status-dot ${model.ready ? 'green' : 'amber'}`} /><span>{model.ready ? 'Ready to read' : model.loading ? 'Starting up…' : model.state === 'unavailable' ? 'Not ready' : 'Connecting…'}</span></div>
        </div>

        {error && <div className="error-banner" role="alert"><span>{error}</span><button className="icon-button" aria-label="Dismiss error" onClick={() => setError('')}><X size={17} /></button></div>}

        <section className="studio-grid" aria-label="Lip-reading workspace">
          <div className="camera-panel">
            <div className="panel-heading"><span className="panel-title"><Camera size={17} /> Camera</span><span className="camera-state"><span className={`status-dot ${cameraOn ? 'green' : ''}`} />{cameraOn ? 'On' : 'Off'}</span></div>
            <div className={`camera-stage ${cameraOn ? 'is-on' : ''}`} ref={stageRef}>
              <video ref={videoRef} muted playsInline className={mirrored ? 'mirrored' : ''} aria-label="Live camera preview" />
              <div className="stage-topline"><span className="video-tag"><span className={`status-dot ${mode === 'reading' ? 'green' : ''}`} />{mode === 'reading' ? 'READING' : 'CAMERA VIEW'}</span><span className="video-tag"><MicOff size={12} /> NO AUDIO</span></div>
              {!cameraOn && <div className="camera-placeholder">
                <div className="face-guide"><span className="corner top-left" /><span className="corner top-right" /><span className="corner bottom-left" /><span className="corner bottom-right" /><svg viewBox="0 0 180 210" fill="none" aria-hidden="true"><path d="M90 31c-29 0-45 23-43 57l4 24c4 25 20 45 39 45s35-20 39-45l4-24c2-34-14-57-43-57Z" stroke="currentColor" strokeWidth="1.3" /><path d="M66 151v17c-28 5-45 15-48 33m96-50v17c28 5 45 15 48 33" stroke="currentColor" strokeWidth="1.3" /><path d="M73 126c10 6 24 6 34 0" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /><path d="M87 91l-4 20h12" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" /><path d="M65 85h10m30 0h10" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /><rect x="61" y="115" width="58" height="24" rx="9" stroke="#afcb95" strokeWidth="1.2" strokeDasharray="3 4" /></svg></div>
                <h2>Start with your camera.</h2><p>Keep your face visible and well lit.</p>
                <button className="camera-enable" onClick={() => void enableCamera()} disabled={cameraLoading}>{cameraLoading ? <LoaderCircle size={16} className="spin" /> : <Camera size={16} />}{cameraLoading ? 'Opening camera…' : 'Enable camera'}<ArrowUpRight size={15} /></button>
              </div>}
              {cameraOn && <div className="framing-guide" aria-hidden="true"><span className="corner top-left" /><span className="corner top-right" /><span className="corner bottom-left" /><span className="corner bottom-right" /></div>}
              {cameraHint && <div className="camera-guidance" role="status"><Focus size={20} aria-hidden="true" /><div><strong>{cameraHint === 'move_closer' ? 'Move closer' : cameraHint === 'multiple_faces' ? 'More than one face detected' : 'Face the camera'}</strong><span>{cameraHint === 'move_closer' ? 'Your face is too small to read clearly.' : cameraHint === 'multiple_faces' ? 'Keep one face in view. Check pictures and screens behind you, too.' : 'Keep your whole face visible and well lit.'}</span></div></div>}
              <div className="stage-bottomline"><span><Focus size={13} /> Keep your face inside the frame</span><button className="stage-icon" aria-label="Expand camera preview" onClick={() => { if (!document.fullscreenElement) void stageRef.current?.requestFullscreen().catch(() => setError('Fullscreen is unavailable in this browser.')); else void document.exitFullscreen(); }}><Expand size={17} /></button></div>
            </div>
            <div className="camera-controls">
              <div className="camera-source">
                <button className={`camera-toggle ${cameraOn ? 'active' : ''}`} onClick={toggleCamera} disabled={cameraLoading || showExamples} aria-label={cameraOn ? 'Turn camera off' : 'Turn camera on'}>{cameraOn ? <Camera size={18} /> : <CameraOff size={18} />}</button>
                {devices.length > 1 && <div className="device-select-wrap"><select aria-label="Camera device" value={deviceId} disabled={!cameraOn || busy || showExamples} onChange={(event) => { setDeviceId(event.target.value); void enableCamera(event.target.value); }}>{devices.map((device, index) => <option key={device.deviceId} value={device.deviceId}>{device.label || `Camera ${index + 1}`}</option>)}</select><ChevronDown size={12} /></div>}
              </div>
              <div className="record-controls">
                <button className="icon-button settings-toggle" aria-label="Settings" aria-expanded={showSettings} aria-controls="reading-settings" onClick={() => setShowSettings(!showSettings)}><Settings2 size={19} /></button>
                <button className={`read-button ${busy ? 'is-reading' : ''}`} disabled={showExamples || mode === 'stopping' || (!busy && (!cameraOn || !model.ready))} onClick={busy ? stopReading : startReading}>{mode === 'connecting' || mode === 'stopping' ? <LoaderCircle size={15} className="spin" /> : busy ? <Square size={13} fill="currentColor" /> : <Play size={14} fill="currentColor" />}{mode === 'connecting' ? 'Connecting…' : mode === 'stopping' ? 'Finishing…' : busy ? 'Stop reading' : 'Start reading'}</button>
              </div>
            </div>
            {showSettings && <div className="settings-panel reading-settings" id="reading-settings">
              <label><input type="checkbox" checked={mirrored} onChange={(event) => setMirrored(event.target.checked)} /> Mirror camera view</label>
              <label className="phrase-length"><span>Maximum phrase length</span><select aria-label="Maximum phrase length" aria-describedby="phrase-length-note" value={utteranceSeconds} disabled={busy || showExamples} onChange={(event) => setUtteranceSeconds(Number(event.target.value))}><option value={30}>30 seconds</option><option value={60}>60 seconds</option></select></label>
              <p id="phrase-length-note">{busy ? 'Stop reading to change phrase length.' : utteranceSeconds === 60 ? 'Longer phrases take more time to finish. Pause between sentences.' : 'A pause starts a new sentence. Your session can keep going.'}</p>
            </div>}
          </div>

          <div className="transcript-panel">
            <div className="panel-heading"><span className="panel-title"><span className="transcript-icon"><i /><i /><i /></span>Transcript</span><span className={`session-badge ${busy ? 'active' : ''}`}>{busy ? <><span className="status-dot green" />{duration}</> : 'ENGLISH'}</span></div>
            <div className="transcript-content" ref={transcriptRef} aria-live="polite" aria-relevant="additions text" onScroll={(event) => {
              const panel = event.currentTarget;
              followTranscriptRef.current = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 80;
            }}>
              {transcripts.length > 0 ? <div className="transcript-sessions">{transcripts.map((transcript) => {
                const live = busy && transcript.session_id === sessionPrefix.current && !transcript.final;
                const drafting = live && !!transcript.draft_text;
                return <div className="transcript-session" key={transcript.session_id}>
                  <div className="transcript-session-meta"><time>{transcript.timestamp}</time>{live && <span>{drafting ? utteranceState === 'refining' ? 'Finishing sentence…' : 'Updating…' : 'Listening…'}</span>}</div>
                  {transcript.utterances.map((text, index) => <p className="transcript-utterance" key={index}><span className="transcript-stable">{text}</span></p>)}
                  {transcript.draft_text && <p className="transcript-utterance" key="draft"><span className="transcript-draft">{transcript.draft_text}</span>{drafting && <span className="text-cursor" aria-hidden="true" />}</p>}
                </div>;
              })}</div> : <div className="transcript-empty"><div className="empty-line-art"><span /><span /><span /><span /></div><h2>Your words appear here.</h2><p>{busy ? 'Text appears after a short delay.' : 'Enable your camera, then start reading.'}</p>{busy && <span className="waiting-label"><span className="status-dot green" />{activityLabel ?? 'Waiting for speech'}</span>}</div>}
            </div>
            <div className="transcript-footnote"><p>{busy ? utteranceState === 'refining' ? 'Finishing your sentence…' : 'Recent words may change when you pause.' : 'Review the transcript for accuracy.'}</p></div>
            <div className="transcript-controls">
              <button className="icon-button transcript-action" aria-label="Clear transcript" title={busy ? 'Stop reading to clear the transcript' : 'Clear transcript'} disabled={!transcripts.length || busy} onClick={() => { followTranscriptRef.current = true; setTranscripts([]); }}><Trash2 size={15} /><span>Clear</span></button>
              <div><button className="icon-button transcript-action" aria-label="Copy transcript" disabled={!transcriptText} onClick={() => void copyTranscript()}>{copied ? <Check size={16} /> : <Copy size={16} />}<span>{copied ? 'Copied' : 'Copy'}</span></button><button className="icon-button transcript-action" aria-label="Download transcript" disabled={!transcriptText} onClick={downloadTranscript}><ArrowDownToLine size={17} /><span>Save</span></button></div>
            </div>
          </div>
        </section>

        <div className="session-feedback" role="status"><span className={`status-dot ${model.ready ? 'green' : 'amber'}`} /><p>{feedback}</p>{!model.ready && <button className="refresh-button" onClick={() => void retryModel()} disabled={retrying || model.loading} aria-label="Retry loading model">{retrying || model.loading ? 'Loading…' : 'Retry'}<ArrowUpRight size={13} /></button>}</div>
      </main>
      {recorder === 'examples' && <SpeakingExamples videoRef={videoRef} cameraOn={cameraOn} onClose={closeExamples} />}
      {recorder === 'sentences' && <EvalSentences videoRef={videoRef} cameraOn={cameraOn} onClose={closeExamples} />}
      {showHelp && <div className="modal-backdrop" onClick={() => setShowHelp(false)}><section className="help-modal" role="dialog" aria-modal="true" aria-labelledby="help-title" onClick={(event) => event.stopPropagation()}>
        <button className="icon-button modal-close" aria-label="Close help" onClick={() => setShowHelp(false)} autoFocus><X size={20} /></button>
        <h2 id="help-title">Reading with Still</h2>
        <ol>
          <li><strong>Get into frame.</strong><span>Enable your camera. Keep one face visible, with good light on your mouth.</span></li>
          <li><strong>Start reading.</strong><span>Speak naturally in English. Text follows after a short delay.</span></li>
          <li><strong>Pause between sentences.</strong><span>Recent words adjust as you speak and settle when you pause. Stay in frame briefly after pressing Stop.</span></li>
        </ol>
        <p className="help-note">Each phrase can run up to {utteranceSeconds} seconds; pauses let your session continue. You can change phrase length in Settings. Lip reading can misinterpret words, so review your transcript.</p>
        <p className="help-note">Powered by {model.model || 'FLARE-LLM Large'} on this machine. Camera frames go to the inference server; your microphone is never used.</p>
        <details className="help-tools"><summary>Troubleshooting</summary><p>Record examples if you need help with missed speech or unwanted words. Record sentences to measure word accuracy.</p><span className="examples-open-row"><button className="examples-open" disabled={busy || cameraLoading} onClick={() => { setShowHelp(false); setRecorder('examples'); }}>Record examples <ArrowUpRight size={13} /></button><button className="examples-open" disabled={busy || cameraLoading} onClick={() => { setShowHelp(false); setRecorder('sentences'); }}>Record sentences <ArrowUpRight size={13} /></button></span><LatencyPanel samples={latencySamples} /></details>
        <button className="read-button help-done" onClick={() => setShowHelp(false)}>Got it <Check size={15} /></button>
      </section></div>}
    </div>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { Circle, LoaderCircle, MicOff, SkipForward, Square, Undo2, X } from 'lucide-react';
import { startRecording } from './recorder';
import type { RecordPhase, Recording } from './recorder';
import { useDialogFocus } from './dialog';

// Held-out read sentences for word error rate. Recordings are never used to
// tune the app; scripts/evaluate_sentences.py scores them (docs/EVALUATION.md).
type Prompt = { id: string; text: string };
type Saved = { id: string; prompt_id: string; speaker: string; frames: number; duration_seconds: number };
type Props = { videoRef: RefObject<HTMLVideoElement | null>; cameraOn: boolean; onClose: () => void };
const MAX_MS = 15000;
const MIN_MS = 1500;
const SPEAKER_KEY = 'still.eval.speaker';

function isSaved(value: unknown): value is Saved {
  if (!value || typeof value !== 'object') return false;
  const item = value as Record<string, unknown>;
  return typeof item.id === 'string' && typeof item.prompt_id === 'string' && typeof item.speaker === 'string'
    && typeof item.frames === 'number' && typeof item.duration_seconds === 'number';
}

function storedSpeaker() {
  try { return localStorage.getItem(SPEAKER_KEY) ?? ''; } catch { return ''; }
}

export default function EvalSentences({ videoRef, cameraOn, onClose }: Props) {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [saved, setSaved] = useState<Saved[]>([]);
  const [loading, setLoading] = useState(true);
  const [speaker, setSpeaker] = useState(storedSpeaker);
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState<RecordPhase>('idle');
  const [seconds, setSeconds] = useState(3);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [lastId, setLastId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const previewRef = useRef<HTMLVideoElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const doneRef = useRef<HTMLButtonElement>(null);
  const runRef = useRef<Recording | null>(null);
  const active = phase !== 'idle';
  const validSpeaker = /^[A-Za-z0-9_-]{1,16}$/.test(speaker);
  const done = useMemo(() => new Set(saved.filter((item) => item.speaker === speaker).map((item) => item.prompt_id)), [saved, speaker]);
  const prompt = prompts[index];
  const recorded = prompt ? done.has(prompt.id) : false;

  const nextOpen = useCallback((from: number, finished: Set<string>) => {
    for (let step = 0; step < prompts.length; step += 1) {
      const candidate = (from + step) % prompts.length;
      if (!finished.has(prompts[candidate].id)) return candidate;
    }
    return from;
  }, [prompts]);

  useEffect(() => {
    const controller = new AbortController();
    void fetch('/api/eval-sentences', { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('Could not load the sentence list. Close this panel and try again.');
        const body: unknown = await response.json();
        const data = body as { prompts?: unknown; recordings?: unknown };
        if (!Array.isArray(data.prompts) || !Array.isArray(data.recordings) || !data.recordings.every(isSaved)) throw new Error('The server returned an invalid sentence list.');
        setPrompts(data.prompts as Prompt[]); setSaved(data.recordings);
      }).catch((cause) => {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Could not load sentences.');
      }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => { controller.abort(); runRef.current?.cancel(); };
  }, []);

  // Start each speaker at their first unrecorded sentence.
  useEffect(() => { if (prompts.length && validSpeaker) setIndex((current) => nextOpen(current, done)); }, [prompts, validSpeaker]);

  useEffect(() => { try { localStorage.setItem(SPEAKER_KEY, speaker); } catch { /* Speaker code is a convenience only. */ } }, [speaker]);

  const cancelRecording = useCallback(() => {
    runRef.current?.cancel(); runRef.current = null;
    setPhase('idle'); setNotice('Recording canceled. Nothing was saved.');
  }, []);

  useEffect(() => {
    const preview = previewRef.current;
    if (preview && cameraOn) {
      preview.srcObject = videoRef.current?.srcObject ?? null;
      void preview.play().catch(() => { /* The studio preview remains available. */ });
    }
    if (!cameraOn && runRef.current && !runRef.current.isSaving()) {
      cancelRecording(); setError('The camera disconnected. Enable it again before recording.');
    }
    return () => { if (preview) preview.srcObject = null; };
  }, [cameraOn, videoRef, cancelRecording]);

  useEffect(() => { if (phase === 'recording') doneRef.current?.focus(); }, [phase]);

  useDialogFocus(dialogRef, closeRef, onClose);

  const record = () => {
    const video = videoRef.current;
    if (runRef.current || !prompt || recorded || !validSpeaker || !cameraOn || !video || video.readyState < 2 || loading || deleting) return;
    const promptId = prompt.id;
    setError(''); setNotice(''); setElapsed(0);
    runRef.current = startRecording({
      video, path: '/api/eval-sentences', start: { prompt_id: promptId, speaker }, maxMs: MAX_MS,
      onPhase: (next, remaining) => { setPhase(next); setSeconds(remaining); },
      onElapsed: setElapsed,
      onSaved: (message) => {
        runRef.current = null; setPhase('idle');
        if (!isSaved(message) || message.prompt_id !== promptId || message.speaker !== speaker) { setError('The saved recording did not match this sentence. Close this panel and reopen it.'); return; }
        const next = [...saved.filter((item) => item.id !== message.id), message];
        setSaved(next); setLastId(message.id);
        setIndex((current) => nextOpen(current + 1, new Set(next.filter((item) => item.speaker === speaker).map((item) => item.prompt_id))));
        setNotice('Saved on this machine. Here is the next sentence.');
      },
      onError: (detail) => { runRef.current = null; setPhase('idle'); setError(detail); },
    });
  };

  const finish = () => {
    if (elapsed < MIN_MS) return;
    runRef.current?.stop();
  };

  const undoLast = async () => {
    const item = saved.find((entry) => entry.id === lastId);
    if (!item || active || deleting) return;
    setDeleting(true); setError(''); setNotice('');
    try {
      const response = await fetch(`/api/eval-sentences/${encodeURIComponent(item.id)}`, { method: 'DELETE' });
      if (!response.ok) throw new Error('Could not delete that recording. Please try again.');
      setSaved((current) => current.filter((entry) => entry.id !== item.id));
      setLastId(null);
      const position = prompts.findIndex((entry) => entry.id === item.prompt_id);
      if (position >= 0) setIndex(position);
      setNotice('Deleted. Record that sentence again when you are ready.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not delete that recording.'); }
    finally { setDeleting(false); }
  };

  return <div className="modal-backdrop examples-backdrop">
    <section ref={dialogRef} tabIndex={-1} className="examples-modal" role="dialog" aria-modal="true" aria-labelledby="sentences-title">
      <div className="examples-heading"><div><div className="eyebrow">ACCURACY CHECK</div><h2 id="sentences-title">Read sentences aloud.</h2></div><button ref={closeRef} className="icon-button" onClick={onClose} disabled={active || deleting} aria-label="Close sentence recording"><X size={20} /></button></div>
      <p className="examples-intro">Read each sentence once, the way you would say it to a friend. These clips are a held-out test of word accuracy: they are never used to tune the app. Any number helps; all {prompts.length || 100} take about 20 minutes.</p>
      <div className="examples-layout">
        <div className="examples-capture">
          <div className="examples-preview"><video ref={previewRef} muted playsInline aria-label="Sentence recording camera preview" />{!cameraOn && <span>Enable your camera in the studio first.</span>}{(phase === 'countdown' || phase === 'recording') && <div className={`examples-timer ${phase === 'recording' ? 'recording' : ''}`}><strong>{phase === 'recording' ? Math.floor(elapsed / 1000) : seconds}</strong><span>{phase === 'countdown' ? 'GET READY' : 'RECORDING'}</span></div>}<span className="examples-audio"><MicOff size={12} /> No audio</span></div>
          <progress className="examples-progress" max={MAX_MS} value={phase === 'recording' ? elapsed : 0} aria-label="Recording length" />
          <div className="examples-record-actions">{active
            ? <><span className="examples-phase" role="status">{phase === 'connecting' ? 'Connecting…' : phase === 'countdown' ? `Starting in ${seconds}…` : phase === 'recording' ? 'Read the sentence, close your mouth, then press Done.' : 'Saving on this machine…'}</span>
                <span className="sentence-actions"><button className="examples-secondary" onClick={cancelRecording} disabled={phase === 'saving'}>Cancel</button>{phase === 'recording' && <button ref={doneRef} className="read-button" onClick={finish}><Square size={11} fill="currentColor" />Done</button>}</span></>
            : <><span className="examples-phase">{recorded ? 'Already recorded by this speaker.' : '3-second countdown · up to 15 seconds'}</span>
                <span className="sentence-actions"><button className="examples-secondary" onClick={() => { setIndex(nextOpen(index + 1, done)); setNotice(''); }} disabled={loading || !prompts.length}><SkipForward size={12} /> Skip</button><button className="read-button" onClick={record} disabled={!cameraOn || loading || deleting || recorded || !validSpeaker || !prompt}><Circle size={12} fill="currentColor" />Record</button></span></>}
          </div>
        </div>
        <div className="examples-choices">
          <label className="sentence-speaker"><span className="info-label">SPEAKER CODE</span><input value={speaker} maxLength={16} disabled={active} placeholder="e.g. CF" onChange={(event) => setSpeaker(event.target.value.trim())} aria-describedby="speaker-note" /><small id="speaker-note">Initials or any short code, one per person. Letters, digits, - and _.</small></label>
          <div className="examples-prompt sentence-prompt"><span className="info-label">{prompt ? `SENTENCE ${index + 1} OF ${prompts.length}` : 'SENTENCE'}</span><p>{loading ? 'Loading…' : prompt?.text}</p><small>Start with your mouth closed. Speak normally; do not exaggerate. If you stumble, cancel and record again.</small></div>
          <p className="examples-count" role="status">{loading ? 'Loading saved recordings…' : validSpeaker ? `${done.size} of ${prompts.length} recorded by ${speaker}` : 'Enter a speaker code to start.'}</p>
          {lastId && !active && <button className="examples-secondary sentence-undo" onClick={() => void undoLast()} disabled={deleting}>{deleting ? <LoaderCircle size={12} className="spin" /> : <Undo2 size={12} />} Delete last recording</button>}
        </div>
      </div>
      {error && <p className="examples-error" role="alert">{error}</p>}
      {notice && <p className="examples-notice" role="status">{notice}</p>}
      <p className="examples-privacy">Only sentences you record here are saved. They stay on this inference machine until deleted; no microphone is used. Ordinary reading still keeps video in memory only.</p>
    </section>
  </div>;
}

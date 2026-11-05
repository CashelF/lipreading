/** Numeric-only, bounded browser timing. No video or transcript content. */
export type LatencySample = {
  frameTimestampMs: number; receivedMs: number; displayMs: number;
  encodingMs: number; serverMs: number; queueMs: number;
  networkRoundTripMs: number | null; recognition: boolean;
  stages: Record<string, number>;
};

const STAGES = ['frame_decode', 'activity_preprocessing', 'recognition_preprocessing',
  'preprocessing', 'upload', 'inference', 'grounding'] as const;
const finiteTime = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;

export class LatencyTracker {
  private frames = new Map<number, number>();
  samples: LatencySample[] = [];
  roundTripMs: number | null = null;
  private pingMs: number | null = null;

  reset() { this.frames.clear(); this.samples = []; this.roundTripMs = null; this.pingMs = null; }

  frame(capturedMs: number, sentMs: number) {
    if (!finiteTime(capturedMs) || !finiteTime(sentMs) || sentMs < capturedMs) return;
    this.frames.set(capturedMs, sentMs - capturedMs);
    while (this.frames.size > 2200) this.frames.delete(this.frames.keys().next().value!);
  }

  ping(now: number) { this.pingMs = now; }
  pong(sent: unknown, now: number) {
    if (finiteTime(sent) && sent === this.pingMs && now >= sent) {
      this.roundTripMs = now - sent;
      this.pingMs = null;
    }
  }

  receive(message: { window_end_ms?: unknown; latency_ms?: unknown; timings_ms?: unknown }, now: number): LatencySample | null {
    const frame = message.window_end_ms;
    if (!finiteTime(frame) || !finiteTime(message.latency_ms) || !finiteTime(now) || now < frame) return null;
    const encodingMs = this.frames.get(frame);
    if (encodingMs === undefined) return null; // Old session or an unmeasured frame.
    const raw = message.timings_ms;
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
    const timings = raw as Record<string, unknown>;
    const stages: Record<string, number> = {};
    for (const key of STAGES) if (finiteTime(timings[key])) stages[key] = timings[key];
    return { frameTimestampMs: frame, receivedMs: now - frame, displayMs: now - frame,
      encodingMs, serverMs: message.latency_ms,
      queueMs: finiteTime(timings.queue) ? timings.queue : 0,
      networkRoundTripMs: this.roundTripMs,
      recognition: (stages.inference ?? 0) > 0 || timings.recognition_reused === 1, stages };
  }

  displayed(sample: LatencySample, now: number) {
    if (!finiteTime(now) || now < sample.frameTimestampMs + sample.receivedMs) return;
    this.samples = [...this.samples.slice(-119), { ...sample, displayMs: now - sample.frameTimestampMs }];
  }
}

export function percentile(values: number[], fraction: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  if (fraction === .5 && sorted.length % 2 === 0) return (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
  return sorted[Math.max(0, Math.min(sorted.length - 1, Math.ceil(fraction * sorted.length) - 1))];
}

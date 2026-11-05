import { type LatencySample, percentile } from './latency';

const ms = (value: number | null | undefined) => value == null ? '—' : value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;

export default function LatencyPanel({ samples }: { samples: LatencySample[] }) {
  if (!samples.length) return null;
  const reading = samples.filter((sample) => sample.recognition);
  const latest = reading.at(-1) ?? samples.at(-1)!;
  const distribution = reading.map((sample) => sample.displayMs);
  const download = () => {
    const blob = new Blob([JSON.stringify({
      description: 'Latest-frame capture to transcript display estimate; not word-aligned speech latency. Display uses the animation frame after the first paint opportunity. RTT includes server dispatch. No video or transcript text.',
      samples,
    }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url; link.download = 'still-timings.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <details className="latency-panel">
    <summary>Update timing <span>{ms(latest.displayMs)} <small>{reading.length ? 'latest speech update' : 'activity check'}</small></span></summary>
    <p>Age of the newest video frame when the transcript is displayed. This includes processing and your connection; words earlier in the video can be older.</p>
    <dl>
      <div><dt>Median speech update</dt><dd>{ms(percentile(distribution, .5))}</dd></div>
      <div><dt>95th percentile</dt><dd>{ms(percentile(distribution, .95))}</dd></div>
      <div><dt>Camera encoding</dt><dd>{ms(latest.encodingMs)}</dd></div>
      <div><dt>Server queue</dt><dd>{ms(latest.queueMs)}</dd></div>
      <div><dt>Server processing</dt><dd>{ms(latest.serverMs)}</dd></div>
      <div><dt>Connection round trip</dt><dd>{ms(latest.networkRoundTripMs)}</dd></div>
    </dl>
    <p>{reading.length} speech {reading.length === 1 ? 'update' : 'updates'} in the last {samples.length} {samples.length === 1 ? 'sample' : 'samples'}. Display timing is a browser paint estimate; camera sensor exposure is not included.</p>
    <button className="examples-open" onClick={download}>Download numeric timings</button>
  </details>;
}

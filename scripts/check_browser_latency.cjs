// Run against the isolated JS emitted from src/latency.ts; never needs a browser.
const assert = require('node:assert/strict');
const { LatencyTracker, percentile } = require(process.argv[2]);
const tracker = new LatencyTracker();
tracker.frame(1000, 1012);
tracker.ping(1015);
tracker.pong(1015, 1065);
const message = { window_end_ms: 1000, latency_ms: 500,
  timings_ms: { queue: 100, preprocessing: 200, inference: 180, grounding: 120,
    injected: 'private text', negative: -1 } };
const sample = tracker.receive(message, 1700);
assert.equal(sample.receivedMs, 700);
assert.equal(sample.encodingMs, 12);
assert.equal(sample.serverMs, 500);
assert.equal(sample.queueMs, 100);
assert.equal(sample.networkRoundTripMs, 50);
assert.equal(sample.recognition, true);
assert.equal(sample.stages.injected, undefined);
tracker.displayed(sample, 1720);
assert.equal(tracker.samples[0].displayMs, 720);
assert.equal(sample.displayMs, 700);
assert.equal(tracker.receive({ ...message, window_end_ms: 999 }, 1700), null);
assert.equal(tracker.receive(message, 500), null);
assert.equal(tracker.receive({ ...message, latency_ms: NaN }, 1700), null);
for (let i = 0; i < 150; i++) tracker.displayed(sample, 1720 + i);
assert.equal(tracker.samples.length, 120);
tracker.reset();
assert.equal(tracker.samples.length, 0);
assert.equal(tracker.receive(message, 1700), null);
assert.equal(tracker.roundTripMs, null);
tracker.pong(1015, 2000);
assert.equal(tracker.roundTripMs, null);
assert.equal(percentile([30, 10, 20], .5), 20);
assert.equal(percentile([30, 10, 20, 40], .5), 25);
assert.equal(percentile([], .5), null);
assert.equal(percentile([10, 20, 30, 40], .95), 40);
console.log('Browser latency checks passed: same-clock timing, paint estimate, bounded samples, session reset, numeric-only output.');

// Pass the isolated CommonJS output of src/capture.ts as the first argument.
const assert = require('node:assert/strict');
const { captureSize } = require(process.argv[2]);
for (const [input, output] of [
  [[1280, 720], [1280, 720]],
  [[640, 360], [640, 360]],
  [[320, 240], [320, 240]],
  [[1920, 1080], [1280, 720]],
  [[1600, 1200], [1280, 960]],
  [[1080, 1920], [540, 960]],
  [[0, 720], [0, 0]],
  [[NaN, 720], [0, 0]],
]) {
  const { width, height } = captureSize(...input);
  assert.deepEqual([width, height], output);
}
console.log('Capture geometry passed: native detail, server bounds, aspect ratio, no upscaling.');

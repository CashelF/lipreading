// Preserve camera detail within the server's image bounds. Never enlarge a
// smaller feed: extra interpolated pixels would hide the face-quality limit.
export function captureSize(sourceWidth: number, sourceHeight: number) {
  if (!Number.isFinite(sourceWidth) || !Number.isFinite(sourceHeight)
      || sourceWidth <= 0 || sourceHeight <= 0) return { width: 0, height: 0 };
  const scale = Math.min(1280 / sourceWidth, 960 / sourceHeight, 1);
  return { width: Math.round(sourceWidth * scale), height: Math.round(sourceHeight * scale) };
}

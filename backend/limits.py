"""Shared bounds for full-utterance recognition and its retained camera context."""

MAX_UTTERANCE_SECONDS = 60
DEFAULT_UTTERANCE_SECONDS = 30
UTTERANCE_OPTIONS = (30, 60)
MAX_UTTERANCE_MS = MAX_UTTERANCE_SECONDS * 1000.
# Allow an in-flight decode to finish while new camera frames arrive.
HISTORY_SECONDS = 80
HISTORY_RETENTION_MS = HISTORY_SECONDS * 1000.
MAX_HISTORY_FRAMES = HISTORY_SECONDS * 40 + 2
# Covers the maximum admitted JPEG size at the maximum admitted capture rate.
MAX_HISTORY_BYTES = MAX_HISTORY_FRAMES * 512_000
MAX_PREPROCESS_FRAMES = HISTORY_SECONDS * 25 + 1
UTTERANCE_CACHE_FRAMES = MAX_UTTERANCE_SECONDS * 25 + 2

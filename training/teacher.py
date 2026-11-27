"""Whisper as a teacher: what it hears in a clip's soundtrack, as one code every 20 ms.

After FLARE (arXiv 2610.19559): the lipreader learns to predict these codes from the silent
video, which tells it frame by frame what is being said. Text alone gives one signal per clip,
and a lipreader trained from scratch on that never starts. Whisper is frozen and is used only
in training; the app still never hears audio.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

CODES = 8192
SAMPLES_PER_FRAME = 640  # 16 kHz audio against 25 fps video: two codes per video frame.


class WhisperCodes(nn.Module):
    """Frozen Whisper encoder plus a fixed random quantizer (as in BEST-RQ) over its features."""

    def __init__(self, encoder: nn.Module, mel_filters, *, code_dim: int = 16, seed: int = 0):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        width = encoder.layer_norm.normalized_shape[0]
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer('filters', torch.as_tensor(mel_filters, dtype=torch.float32).T.contiguous(), persistent=False)
        self.register_buffer('window', torch.hann_window(400), persistent=False)
        # The four buffers below define the codes, so they are saved with every checkpoint.
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('std', torch.ones(width))
        self.register_buffer('projection', nn.init.xavier_uniform_(torch.empty(width, code_dim), generator=generator))
        self.register_buffer('codebook', F.normalize(torch.randn(CODES, code_dim, generator=generator)))

    @classmethod
    def pretrained(cls, name: str = 'openai/whisper-small') -> 'WhisperCodes':
        from transformers import WhisperFeatureExtractor, WhisperModel
        return cls(WhisperModel.from_pretrained(name).encoder, WhisperFeatureExtractor.from_pretrained(name).mel_filters)

    def code_state(self) -> dict:
        return {name: getattr(self, name) for name in ('mean', 'std', 'projection', 'codebook')}

    @torch.no_grad()
    def features(self, audio: torch.Tensor) -> torch.Tensor:
        """int16 audio (B, 640 T) to Whisper's last-layer features (B, 2 T, width).

        Whisper pads every input to 30 seconds; here it sees only the batch's own length, which
        costs a fraction of the compute and still gives consistent targets.
        """
        spectrum = torch.stft(audio.float() / 32768, 400, 160, window=self.window, return_complex=True)[..., :-1].abs() ** 2
        log = (self.filters @ spectrum).clamp_min(1e-10).log10()
        log = torch.maximum(log, log.amax((1, 2), keepdim=True) - 8)
        with torch.autocast(audio.device.type, torch.bfloat16, enabled=audio.is_cuda):
            x = F.gelu(self.encoder.conv2(F.gelu(self.encoder.conv1((log + 4) / 4)))).transpose(1, 2)
            x = x + self.encoder.embed_positions.weight[:x.shape[1]]
            for layer in self.encoder.layers:
                x = layer(x, None, None)[0]
            return self.encoder.layer_norm(x).float()

    @torch.no_grad()
    def fit(self, batches) -> None:
        """Set the feature mean and spread from a few (audio, frame counts) batches. Without this the codes bunch up."""
        kept = []
        for audio, frames in batches:
            features = self.features(audio)
            real = torch.arange(features.shape[1], device=audio.device)[None] < 2 * frames.to(audio.device)[:, None]
            kept.append(features[real])
        kept = torch.cat(kept)
        self.mean.copy_(kept.mean(0))
        self.std.copy_(kept.std(0).clamp_min(1e-5))

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """int16 audio (B, 640 T) to codes (B, T, 2): the two 20 ms codes under each video frame."""
        x = F.normalize((self.features(audio) - self.mean) / self.std @ self.projection, dim=-1)
        return (x @ self.codebook.T).argmax(-1).view(audio.shape[0], -1, 2)  # Nearest code on the unit sphere.

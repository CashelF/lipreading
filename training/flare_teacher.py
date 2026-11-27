"""FLARE's visual encoder as a teacher: its per-frame features for a clip, from the lip stream alone.

The small lipreader learns to reproduce these features from the same video. They are what
FLARE's own decoder reads text from, so matching them is most of the way to reading lips.
Only ``extract_features.py`` runs FLARE; training reads what it saved. Needs the app's model
setup (scripts/setup_model.py): ``vendor/flare_runtime`` and ``models/FLARE-checkpoints``.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]


class NoFace(torch.nn.Module):
    """Stands in for FLARE's face front end when there is no face crop: zeros of the shape it would give."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, video):  # (B, 1, T, h, w)
        return video.new_zeros(video.shape[0], self.width, video.shape[2])


class FlareEncoder:
    def __init__(self, device, root: Path = ROOT):
        import numpy as np
        from omegaconf import OmegaConf, open_dict
        sys.path.insert(0, str(root / 'vendor'))
        from flare_runtime.models.hubert_distill_frm_whisper_offline_bestrq_dual_frontend import AVHubertConfig, AVHubertModelDual
        base = root / 'models/FLARE-checkpoints/exp_22_2_large'
        # These pickle checkpoints are solely the official release that scripts/setup_model.py downloads.
        visual = torch.load(base / 'checkpoints/checkpoint_52_800000.pt', map_location='cpu', weights_only=False)
        config = OmegaConf.create(visual['cfg'])
        config.model = OmegaConf.merge(OmegaConf.structured(AVHubertConfig), config.model)
        with open_dict(config.model):
            config.model.resnet_weights = None
        with open_dict(config.task):
            config.task.setdefault('sample_rate', 16000)
        # Upstream reads its Whisper statistics from a file while building; the checkpoint holds them.
        with tempfile.TemporaryDirectory(prefix='flare-stat-') as folder:
            np.save(Path(folder) / 'whisper.npy', visual['model']['whisper_stat'].cpu().numpy())
            os.environ['FLARE_WHISPER_STAT'] = str(Path(folder) / 'whisper.npy')
            encoder = AVHubertModelDual(config.model, config.task, [])
        del visual
        tuned = torch.load(base / 'finetuned/checkpoints/checkpoint_last.pt', map_location='cpu', weights_only=False)['model']
        prefix = 'encoder.w2v_model.'
        encoder.load_state_dict({name[len(prefix):]: value for name, value in tuned.items() if name.startswith(prefix)}, strict=True)
        encoder.remove_pretraining_modules()
        # The clips are mouth crops, so FLARE's face stream is switched off (its mask is 0 below).
        # Skip computing it: that is half the cost, and the output is the same.
        encoder.feature_extractor_video_face = NoFace(encoder.feature_extractor_video_face.proj.out_features)
        self.encoder = encoder.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)

    @torch.no_grad()
    def __call__(self, video: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """Normalized video (B, T, 88, 88) and frame counts to features (B, T, 768), each vector standardized."""
        lips = video[:, None].to(torch.bfloat16)
        present = torch.ones(len(video), device=video.device)
        source = {'audio': None, 'video': (lips[..., :1, :1], lips), 'face_lip_mask': (0 * present, present)}
        padding = torch.arange(video.shape[1], device=video.device)[None] >= frames.to(video.device)[:, None]
        features, _ = self.encoder.extract_finetune(source, padding_mask=padding)
        return torch.nn.functional.layer_norm(features.float(), features.shape[-1:])

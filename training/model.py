"""Lip clips to text: FLARE's ConvNeXt lip front end, a small Conformer, CTC and attention heads."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.models import Conformer

TEACHER_DIM = 768  # Width of FLARE's encoder output.


class ChannelNorm(nn.Module):
    """Layer normalization over the channel axis (axis 1) of an image or video tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(channels)), nn.Parameter(torch.zeros(channels)), eps

    def forward(self, x):
        x = x - x.mean(1, keepdim=True)
        x = x / torch.sqrt(x.pow(2).mean(1, keepdim=True) + self.eps)
        shape = (1, -1) + (1,) * (x.dim() - 2)
        return self.weight.view(shape) * x + self.bias.view(shape)


class Block(nn.Module):
    """ConvNeXt block: depthwise convolution, then a pointwise expansion and projection."""

    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1, self.pwconv2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        y = self.norm(self.dwconv(x).permute(0, 2, 3, 1))
        return x + (self.gamma * self.pwconv2(F.gelu(self.pwconv1(y)))).permute(0, 3, 1, 2)


class Trunk(nn.Module):
    def __init__(self, dims=(96, 192, 384, 768), kernels=(7, 7, 5, 3)):
        super().__init__()
        self.downsample_layers = nn.ModuleList([nn.Identity()] + [
            nn.Sequential(ChannelNorm(dims[i]), nn.Conv2d(dims[i], dims[i + 1], 2, 2, padding=1 if i == 1 else 0))
            for i in range(3)])
        self.stages = nn.ModuleList(nn.Sequential(Block(dim, kernel), Block(dim, kernel)) for dim, kernel in zip(dims, kernels))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

    def forward(self, x):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
        return self.norm(x.mean((-2, -1)))


class Frontend(nn.Module):
    """One feature vector per video frame; the only temporal context is the 5-frame stem.

    The layout and parameter names are those of FLARE's lip front end (``ConvEncoderT`` in its
    ``convnext.py``: a 3D patch stem and a small ConvNeXt), so its trained weights load as they are.
    """
    dim = 768

    def __init__(self):
        super().__init__()
        self.frontend3D = nn.Sequential(nn.Conv3d(1, 96, (5, 4, 4), (1, 4, 4), (2, 0, 0), bias=False), ChannelNorm(96))
        self.trunk = Trunk()

    def forward(self, video):  # (B, T, H, W)
        batch, frames = video.shape[:2]
        y = self.frontend3D(video.unsqueeze(1)).transpose(1, 2).flatten(0, 1)  # (B*T, C, h, w)
        return self.trunk(y).view(batch, frames, self.dim)


def positions(length: int, dim: int, device) -> torch.Tensor:
    position = torch.arange(length, device=device).unsqueeze(1)
    scale = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim, device=device)
    table[:, 0::2], table[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
    return table


class LipReader(nn.Module):
    """FLARE's lip front end (kept as it is), then a small Conformer, a CTC head and an attention decoder.

    Training reads the front end's output from disk (``extract_features.py``), so every method below
    takes those features. On video, call ``front`` first.
    """

    def __init__(self, vocab_size: int, *, dim: int = 256, layers: int = 12, heads: int = 4,
                 decoder_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.vocab_size, self.dim = vocab_size, dim
        self.sos_eos = vocab_size - 1
        self.frontend = Frontend().requires_grad_(False)
        self.project = nn.Sequential(nn.Linear(Frontend.dim, dim), nn.LayerNorm(dim))
        self.encoder = Conformer(dim, heads, 4 * dim, layers, 31, dropout=dropout)
        self.ctc = nn.Linear(dim, vocab_size)
        self.teacher = nn.Linear(dim, TEACHER_DIM)  # Training only: FLARE's encoder output for each frame.
        self.embed = nn.Embedding(vocab_size, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, decoder_layers, norm=nn.LayerNorm(dim))
        self.output = nn.Linear(dim, vocab_size)

    @torch.no_grad()
    def front(self, video):
        """Normalized video (B, T, 88, 88) to features (B, T, 768), standardized per frame as they are on disk."""
        return F.layer_norm(self.frontend(video).float(), (Frontend.dim,))

    def encode(self, features, lengths):
        """features: (B, T, 768). Returns encoder states, their lengths and a padding mask."""
        features = features[:, :int(lengths.max())]
        padding = torch.arange(features.shape[1], device=features.device)[None] >= lengths[:, None]
        x = self.project(features).transpose(0, 1)  # (T, B, D)
        for layer in self.encoder.conformer_layers:
            # torchaudio's ConformerLayer.forward, with one addition: padded frames are zeroed just before
            # the depthwise convolution, as its own edge padding would be. Without that they leak into the
            # real frames beside them, and a clip batched with longer clips is encoded differently from
            # the same clip alone.
            x = layer.ffn1(x) * 0.5 + x
            attended, _ = layer.self_attn(*[layer.self_attn_layer_norm(x)] * 3, key_padding_mask=padding, need_weights=False)
            x = layer.self_attn_dropout(attended) + x
            pointwise, gate, *rest = layer.conv_module.sequential
            y = gate(pointwise(layer.conv_module.layer_norm(x).permute(1, 2, 0)))  # (B, D, T)
            for step in rest:
                y = step(y.masked_fill(padding[:, None], 0)) if step is rest[0] else step(y)
            x = y.permute(2, 0, 1) + x
            x = layer.final_layer_norm(layer.ffn2(x) * 0.5 + x)
        return x.transpose(0, 1), lengths, padding

    def decode_step(self, tokens, states, padding):
        """Next-token logits for every position of ``tokens`` (B, U)."""
        length = tokens.shape[1]
        x = self.embed(tokens) * math.sqrt(self.dim) + positions(length, self.dim, tokens.device)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), 1)
        y = self.decoder(x, states, tgt_mask=causal, memory_key_padding_mask=padding)
        return self.output(y)

    def forward(self, features, lengths, targets, target_lengths, *, teacher=None, teacher_weight: float = 0.0,
                ctc_weight: float = 0.1, smoothing: float = 0.1):
        """targets: (B, U) padded with 0; teacher: FLARE's standardized encoder features (B, T, 768), or None.

        Returns (loss, ctc_loss, attention_loss, teacher_loss): the text losses per token, the teacher loss per frame.
        """
        states, lengths, padding = self.encode(features, lengths)
        log_probs = self.ctc(states).float().log_softmax(-1).transpose(0, 1)
        # Both text losses are per token, so the weights mean what they say whatever the clip length.
        ctc = F.ctc_loss(log_probs, targets, lengths, target_lengths, blank=0, reduction='sum', zero_infinity=True)
        ctc = ctc / target_lengths.sum()
        batch, longest = targets.shape
        sos = torch.full((batch, 1), self.sos_eos, dtype=targets.dtype, device=targets.device)
        valid = torch.arange(longest + 1, device=targets.device)[None] <= target_lengths[:, None]
        decoder_input = torch.cat([sos, targets], 1)
        expected = torch.cat([targets, torch.zeros_like(sos)], 1)
        expected[torch.arange(batch), target_lengths] = self.sos_eos
        expected = expected.masked_fill(~valid, -100)
        logits = self.decode_step(decoder_input, states, padding).float()
        attention = F.cross_entropy(logits.flatten(0, 1), expected.flatten(), ignore_index=-100,
                                    label_smoothing=smoothing)
        copied = ctc.new_zeros(())
        if teacher is not None:
            # Text gives one signal per clip. The teacher gives a full target for every frame, computed
            # from the same video, and that is what gets a small model reading: distance plus direction.
            predicted, wanted = self.teacher(states).float()[~padding], teacher[:, :states.shape[1]][~padding]
            copied = F.l1_loss(predicted, wanted) + 1 - F.cosine_similarity(predicted, wanted, dim=-1).mean()
        loss = ctc_weight * ctc + (1 - ctc_weight) * attention + teacher_weight * copied
        return loss, ctc.detach(), attention.detach(), copied.detach()

    @torch.no_grad()
    def greedy(self, features, lengths, max_tokens: int = 120) -> list[list[int]]:
        states, _, padding = self.encode(features, lengths)
        batch = features.shape[0]
        tokens = torch.full((batch, 1), self.sos_eos, dtype=torch.long, device=features.device)
        done = torch.zeros(batch, dtype=torch.bool, device=features.device)
        for _ in range(max_tokens):
            following = self.decode_step(tokens, states, padding)[:, -1].argmax(-1)
            following = following.masked_fill(done, self.sos_eos)
            tokens = torch.cat([tokens, following[:, None]], 1)
            done |= following == self.sos_eos
            if done.all():
                break
        results = []
        for row in tokens[:, 1:].tolist():
            results.append(row[:row.index(self.sos_eos)] if self.sos_eos in row else row)
        return results

    @torch.no_grad()
    def beam_search(self, features, length, *, beam: int = 10, ctc_weight: float = 0.1, max_tokens: int = 120,
                    length_bonus: float = 0.5) -> list[int]:
        """One clip. Attention beam search, then rescoring of finished beams with the CTC head."""
        states, lengths, padding = self.encode(features[None], length[None])
        ctc_log_probs = self.ctc(states).float().log_softmax(-1).transpose(0, 1)
        beams, finished = [([self.sos_eos], 0.0)], []
        for _ in range(max_tokens):
            tokens = torch.tensor([prefix for prefix, _ in beams], device=features.device)
            scores = self.decode_step(tokens, states.expand(len(beams), -1, -1),
                                      padding.expand(len(beams), -1))[:, -1].float().log_softmax(-1)
            best, index = scores.topk(beam, -1)
            candidates = [(prefix + [int(token)], total + float(score))
                          for (prefix, total), row_scores, row_index in zip(beams, best, index)
                          for score, token in zip(row_scores, row_index)]
            candidates.sort(key=lambda item: item[1], reverse=True)
            beams = []
            for prefix, total in candidates[:beam]:
                (finished if prefix[-1] == self.sos_eos else beams).append((prefix, total))
            if not beams or len(finished) >= beam:
                break
        finished = finished or beams
        rescored = []
        for prefix, total in finished:
            words = [token for token in prefix[1:] if token != self.sos_eos]
            score = total + length_bonus * len(words)
            if words and ctc_weight:
                target = torch.tensor([words], device=features.device)
                ctc = -F.ctc_loss(ctc_log_probs, target, lengths, torch.tensor([len(words)], device=features.device),
                                  blank=0, reduction='sum', zero_infinity=True)
                score = (1 - ctc_weight) * score + ctc_weight * float(ctc)
            rescored.append((score, words))
        return max(rescored, key=lambda item: item[0])[1]

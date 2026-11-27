"""Lip clips to text: 3D-conv + ResNet18 front end, Conformer encoder, CTC and attention heads."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.models import Conformer


class BasicBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.shortcut = None
        if stride != 1 or inputs != outputs:
            self.shortcut = nn.Sequential(nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.BatchNorm2d(outputs))

    def forward(self, x):
        y = F.silu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.silu(y + (x if self.shortcut is None else self.shortcut(x)))


class Frontend(nn.Module):
    """One feature vector per video frame; the only temporal context is the 5-frame stem."""

    def __init__(self, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, width, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False), nn.BatchNorm3d(width), nn.SiLU(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)))
        layers, inputs = [], width
        for index, outputs in enumerate((width, 2 * width, 4 * width, 8 * width)):
            layers += [BasicBlock(inputs, outputs, 1 if index == 0 else 2), BasicBlock(outputs, outputs, 1)]
            inputs = outputs
        self.trunk = nn.Sequential(*layers)
        self.dim = inputs

    def forward(self, video):  # (B, T, H, W)
        batch, frames = video.shape[:2]
        y = self.stem(video.unsqueeze(1)).transpose(1, 2).flatten(0, 1)  # (B*T, C, h, w)
        y = self.trunk(y.contiguous(memory_format=torch.channels_last))
        return y.mean((2, 3)).view(batch, frames, self.dim)


def positions(length: int, dim: int, device) -> torch.Tensor:
    position = torch.arange(length, device=device).unsqueeze(1)
    scale = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim, device=device)
    table[:, 0::2], table[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
    return table


class LipReader(nn.Module):
    def __init__(self, vocab_size: int, *, width: int = 64, dim: int = 256, layers: int = 12, heads: int = 4,
                 decoder_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.vocab_size, self.dim = vocab_size, dim
        self.sos_eos = vocab_size - 1
        self.frontend = Frontend(width)
        self.project = nn.Linear(self.frontend.dim, dim)
        self.encoder = Conformer(dim, heads, 4 * dim, layers, 31, dropout=dropout)
        self.ctc = nn.Linear(dim, vocab_size)
        self.embed = nn.Embedding(vocab_size, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, decoder_layers, norm=nn.LayerNorm(dim))
        self.output = nn.Linear(dim, vocab_size)

    def encode(self, video, lengths):
        """video: normalized float (B, T, H, W). Returns encoder states and a padding mask."""
        video = video[:, :int(lengths.max())]  # The encoder masks by the longest clip, not the tensor width.
        states, lengths = self.encoder(self.project(self.frontend(video)), lengths)
        padding = torch.arange(states.shape[1], device=states.device)[None] >= lengths[:, None]
        return states, lengths, padding

    def decode_step(self, tokens, states, padding):
        """Next-token logits for every position of ``tokens`` (B, U)."""
        length = tokens.shape[1]
        x = self.embed(tokens) * math.sqrt(self.dim) + positions(length, self.dim, tokens.device)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), 1)
        y = self.decoder(x, states, tgt_mask=causal, memory_key_padding_mask=padding)
        return self.output(y)

    def forward(self, video, lengths, targets, target_lengths, *, ctc_weight: float = 0.1, smoothing: float = 0.1):
        """targets: (B, U) padded with 0. Returns (loss, ctc_loss, attention_loss)."""
        states, lengths, padding = self.encode(video, lengths)
        log_probs = self.ctc(states).float().log_softmax(-1).transpose(0, 1)
        ctc = F.ctc_loss(log_probs, targets, lengths, target_lengths, blank=0, reduction='sum', zero_infinity=True)
        ctc = ctc / video.shape[0]
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
        return ctc_weight * ctc + (1 - ctc_weight) * attention, ctc.detach(), attention.detach()

    @torch.no_grad()
    def greedy(self, video, lengths, max_tokens: int = 120) -> list[list[int]]:
        states, _, padding = self.encode(video, lengths)
        batch = video.shape[0]
        tokens = torch.full((batch, 1), self.sos_eos, dtype=torch.long, device=video.device)
        done = torch.zeros(batch, dtype=torch.bool, device=video.device)
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
    def beam_search(self, video, length, *, beam: int = 10, ctc_weight: float = 0.1, max_tokens: int = 120,
                    length_bonus: float = 0.5) -> list[int]:
        """One clip. Attention beam search, then rescoring of finished beams with the CTC head."""
        states, lengths, padding = self.encode(video[None], length[None])
        ctc_log_probs = self.ctc(states).float().log_softmax(-1).transpose(0, 1)
        beams, finished = [([self.sos_eos], 0.0)], []
        for _ in range(max_tokens):
            tokens = torch.tensor([prefix for prefix, _ in beams], device=video.device)
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
                target = torch.tensor([words], device=video.device)
                ctc = -F.ctc_loss(ctc_log_probs, target, lengths, torch.tensor([len(words)], device=video.device),
                                  blank=0, reduction='sum', zero_infinity=True)
                score = (1 - ctc_weight) * score + ctc_weight * float(ctc)
            rescored.append((score, words))
        return max(rescored, key=lambda item: item[0])[1]

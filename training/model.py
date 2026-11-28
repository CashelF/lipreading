"""Lip clips to text: FLARE's lip front end, some of FLARE's encoder layers and a small Conformer, CTC and attention heads."""
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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps

    def forward(self, x):
        y = x.float()
        return self.weight * (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class GatedMLP(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(dim, 4 * dim, bias=False), nn.Linear(dim, 4 * dim, bias=False)
        self.down_proj = nn.Linear(4 * dim, dim, bias=False)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.down_proj(self.dropout1(F.silu(self.gate_proj(x)) * self.up_proj(x))))


class RotaryAttention(nn.Module):
    """Self-attention with rotary positions; keys and values use fewer heads than queries and are shared."""

    def __init__(self, dim: int, heads: int, shared_heads: int, dropout: float, theta: float = 1e6):
        super().__init__()
        self.heads, self.shared_heads, self.head_dim, self.theta = heads, shared_heads, dim // heads, theta
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj, self.v_proj = nn.Linear(dim, shared_heads * self.head_dim), nn.Linear(dim, shared_heads * self.head_dim)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

    def rotate(self, x):
        """Turn each pair of channels by an angle that grows with the frame's position."""
        count = x.shape[-2]
        rate = 1.0 / self.theta ** (torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim)
        angle = torch.outer(torch.arange(count, device=x.device).float(), rate).repeat(1, 2)
        first, second = x.chunk(2, -1)
        return x * angle.cos().to(x.dtype) + torch.cat([-second, first], -1) * angle.sin().to(x.dtype)

    def forward(self, x, padding):
        batch, count, _ = x.shape
        split = lambda y, heads: y.view(batch, count, heads, self.head_dim).transpose(1, 2)
        query = self.rotate(split(self.q_proj(x), self.heads))
        key = self.rotate(split(self.k_proj(x), self.shared_heads)).repeat_interleave(self.heads // self.shared_heads, 1)
        value = split(self.v_proj(x), self.shared_heads).repeat_interleave(self.heads // self.shared_heads, 1)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=~padding[:, None, None])
        return self.dropout1(self.o_proj(attended.transpose(1, 2).reshape(batch, count, -1)))


class DepthwiseConv(nn.Module):
    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim)

    def forward(self, x):
        return self.conv(F.pad(x, (self.conv.kernel_size[0] // 2,) * 2))


class ConvBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        # The slots without weights keep FLARE's numbering, so its checkpoint loads by name.
        self.net = nn.ModuleList([RMSNorm(dim), nn.Linear(dim, 2 * dim), nn.Identity(), nn.Identity(), DepthwiseConv(dim, 15),
                                  nn.Identity(), RMSNorm(dim), nn.Identity(), nn.Linear(dim, dim), nn.Dropout(dropout)])

    def forward(self, x, padding):
        kept, gate = self.net[1](self.net[0](x)).transpose(1, 2).chunk(2, 1)
        # FLARE does not zero padded frames here, so they leak into real ones; the app never pads.
        # Zeroing them makes a batched clip equal the same clip alone, which is what FLARE was run as.
        y = self.net[4]((kept * gate.sigmoid()).masked_fill(padding[:, None], 0))
        y = self.net[6](y.transpose(1, 2))
        return self.net[9](self.net[8](y * y.sigmoid()))


class FlareLayer(nn.Module):
    """One layer of FLARE's encoder (``Qwen2DecoderLayer`` in its ``qwen_conformer.py``), parameter names included."""

    def __init__(self, dim: int = 768, heads: int = 12, shared_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.self_attn = RotaryAttention(dim, heads, shared_heads, dropout)
        self.mlp1, self.mlp2, self.conv = GatedMLP(dim, dropout), GatedMLP(dim, dropout), ConvBlock(dim, dropout)
        self.input_layernorm1, self.input_layernorm2, self.input_layernorm3 = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
        self.post_layernorm = RMSNorm(dim)

    def forward(self, x, padding):
        x = x + 0.5 * self.mlp1(self.input_layernorm1(x))
        x = x + self.self_attn(self.input_layernorm2(x), padding)
        x = x + self.conv(x, padding)
        x = x + 0.5 * self.mlp2(self.input_layernorm3(x))
        return self.post_layernorm(x)


FLARE_DEPTH = 18


class FlareStack(nn.Module):
    """The first ``count`` of FLARE's 18 encoder layers, with the steps FLARE runs before them.

    The layers keep FLARE's weights and are never trained, so the output is exactly what FLARE has
    at that depth: a prefix is a working encoder, where layers picked from the middle are not.
    """

    def __init__(self, count: int):
        super().__init__()
        self.lip_proj = nn.Linear(TEACHER_DIM, TEACHER_DIM)
        self.mix_norm, self.mix_proj = nn.LayerNorm(3 * TEACHER_DIM), nn.Linear(3 * TEACHER_DIM, TEACHER_DIM)
        self.layers = nn.ModuleList(FlareLayer() for _ in range(count))
        self.requires_grad_(False)

    def load_flare(self, state: dict):
        """state: what extract_features.py saved as encoder.pt (all 18 layers)."""
        kept = lambda name: not name.startswith('layers.') or int(name.split('.')[1]) < len(self.layers)
        self.load_state_dict({name: value for name, value in state.items() if kept(name)})

    def train(self, mode: bool = True):
        return super().train(False)  # Frozen: never drop activations, whatever the model around it does.

    @torch.no_grad()
    def forward(self, features, padding):
        lip = self.lip_proj(features)
        # FLARE joins three streams here: audio, face and lip. The first two are absent, as in its lip-only use.
        x = self.mix_proj(self.mix_norm(torch.cat([torch.zeros_like(lip), torch.zeros_like(lip), lip], -1)))
        for layer in self.layers:
            x = layer(x, padding)
        return x


def positions(length: int, dim: int, device) -> torch.Tensor:
    position = torch.arange(length, device=device).unsqueeze(1)
    scale = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim, device=device)
    table[:, 0::2], table[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
    return table


class LipReader(nn.Module):
    """FLARE's lip front end and the first ``flare_layers`` of its 18 encoder layers (all kept as they
    are), then a small Conformer of ``layers`` layers, a CTC head and an attention decoder.

    ``flare_layers`` sets the size: each FLARE layer is 17M parameters. With 0 the 40M model's
    Conformer has to learn all of FLARE's encoder; with 18 nothing of it is left to learn. Training
    reads the front end's output from disk (``extract_features.py``), so every method below takes
    those features. On video, call ``front`` first.
    """

    def __init__(self, vocab_size: int, *, flare_layers: int = 0, dim: int = 256, layers: int = 12, heads: int = 4,
                 decoder_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.vocab_size, self.dim = vocab_size, dim
        self.sos_eos = vocab_size - 1
        self.frontend = Frontend().requires_grad_(False)
        self.flare = FlareStack(flare_layers) if flare_layers else None
        self.whole_flare = flare_layers == FLARE_DEPTH
        self.project = nn.Sequential(nn.Linear(Frontend.dim, dim), nn.LayerNorm(dim))
        self.encoder = Conformer(dim, heads, 4 * dim, layers, 31, dropout=dropout)
        self.ctc = nn.Linear(dim, vocab_size)
        # Training only: FLARE's encoder output for each frame, unless the model holds all of FLARE's encoder.
        self.teacher = None if self.whole_flare else nn.Linear(dim, TEACHER_DIM)
        self.embed = nn.Embedding(vocab_size, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, decoder_layers, norm=nn.LayerNorm(dim))
        self.output = nn.Linear(dim, vocab_size)

    @torch.no_grad()
    def front(self, video):
        """Normalized video (B, T, 88, 88) to features (B, T, 768), standardized per frame as they are on disk."""
        return F.layer_norm(self.frontend(video).float(), (Frontend.dim,))

    def encode(self, features, lengths, flare=None):
        """features: (B, T, 768). Returns encoder states, their lengths and a padding mask. ``flare``
        supplies the FLARE layers' output when it is already known: with all 18 layers it is FLARE's
        encoder output, which training reads from disk."""
        features = features[:, :int(lengths.max())]
        padding = torch.arange(features.shape[1], device=features.device)[None] >= lengths[:, None]
        if self.flare is not None:
            deep = self.flare(features, padding) if flare is None else flare[:, :features.shape[1]]
            features = F.layer_norm(deep.float(), (TEACHER_DIM,))  # Standardized per frame, as the stored features are.
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
        """targets: (B, U) padded with 0; teacher: FLARE's standardized encoder output (B, T, 768), or None.

        Returns (loss, ctc_loss, attention_loss, teacher_loss): the text losses per token, the teacher loss per frame.
        """
        states, lengths, padding = self.encode(features, lengths, teacher if self.whole_flare else None)
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
        if teacher is not None and not self.whole_flare:
            # Text gives one signal per clip. The teacher gives a full target for every frame, computed
            # from the same video, and that is what gets a small model reading: distance plus direction.
            predicted, wanted = self.teacher(states).float()[~padding], teacher[:, :states.shape[1]][~padding]
            copied = F.l1_loss(predicted, wanted) + 1 - F.cosine_similarity(predicted, wanted, dim=-1).mean()
        loss = ctc_weight * ctc + (1 - ctc_weight) * attention + teacher_weight * copied
        return loss, ctc.detach(), attention.detach(), copied.detach()

    @torch.no_grad()
    def greedy(self, features, lengths, max_tokens: int = 120, flare=None) -> list[list[int]]:
        states, _, padding = self.encode(features, lengths, flare)
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

"""A pixel-space diffusion transformer whose FFNs become LLM-style MoE layers after the first blocks.

Images are cut into patches of raw pixels (no VAE). Timestep and class enter through AdaLN-Zero.
Positions are 2D sin-cos embeddings of each patch's *relative* location, so the same model reads
the half-resolution stage and the full-resolution stage. A wide prediction head (the paper uses a
DDT head) maps backbone features and the noisy input patches back to pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .moe import MoE, SwiGLU


@dataclass
class Config:
    image_size: int = 32
    channels: int = 3
    patch: int = 4
    dim: int = 384
    depth: int = 8
    heads: int = 6
    dense_blocks: int = 2  # the first blocks keep a dense FFN, as in LLM MoEs
    ffn_hidden: int = 1024  # dense FFN width
    experts: int = 32  # 0 makes every block dense: the controlled baseline
    active: int = 4
    routed_hidden: int = 192
    shared_hidden: int = 288  # shared + active x routed ~ dense width: matched active parameters
    head_dim: int = 512
    head_depth: int = 1
    classes: int = 10


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: nn.Module):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn = ffn
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c):
        s1, g1, b1, s2, g2, b2 = self.ada(F.silu(c)).chunk(6, dim=-1)
        b, n, d = x.shape
        q, k, v = self.qkv(modulate(self.norm1(x), b1, s1)).view(b, n, 3, self.heads, d // self.heads).unbind(2)
        att = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + g1.unsqueeze(1) * self.proj(att.transpose(1, 2).reshape(b, n, d))
        return x + g2.unsqueeze(1) * self.ffn(modulate(self.norm2(x), b2, s2))


class PixelDiT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        pdim = cfg.channels * cfg.patch ** 2
        self.embed = nn.Linear(pdim, cfg.dim)
        self.t_mlp = nn.Sequential(nn.Linear(256, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.y_embed = nn.Embedding(cfg.classes + 1, cfg.dim)  # last index = no label (guidance)
        self.blocks = nn.ModuleList([
            Block(cfg.dim, cfg.heads, SwiGLU(cfg.dim, cfg.ffn_hidden) if i < cfg.dense_blocks or not cfg.experts
                  else MoE(cfg.dim, cfg.experts, cfg.active, cfg.routed_hidden, cfg.shared_hidden))
            for i in range(cfg.depth)])
        # Wide prediction head: backbone features plus the noisy patches, at a larger width.
        self.head_feat = nn.Linear(cfg.dim, cfg.head_dim)
        self.head_in = nn.Linear(pdim, cfg.head_dim)
        self.head_cond = nn.Linear(cfg.dim, cfg.head_dim)
        self.head = nn.ModuleList([Block(cfg.head_dim, cfg.head_dim // 64, SwiGLU(cfg.head_dim, 2 * cfg.head_dim))
                                   for _ in range(cfg.head_depth)])
        self.out_norm = nn.RMSNorm(cfg.head_dim, elementwise_affine=False)
        self.out_ada = nn.Linear(cfg.head_dim, 2 * cfg.head_dim)
        self.out = nn.Linear(cfg.head_dim, pdim)
        for layer in (self.out_ada, self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict the stage endpoint for noisy images x [B, C, H, W] at global timesteps t [B]."""
        b, ch, h, w = x.shape
        p = self.cfg.patch
        patches = x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).reshape(b, -1, ch * p * p)
        c = self.t_mlp(timestep_embedding(t * 1000, 256)) + self.y_embed(y)
        tokens = self.embed(patches) + positions(h // p, w // p, self.cfg.dim, x.device)
        for block in self.blocks:
            tokens = block(tokens, c)
        ch_ = self.head_cond(c)
        z = self.head_feat(tokens) + self.head_in(patches)
        for block in self.head:
            z = block(z, ch_)
        shift, scale = self.out_ada(F.silu(ch_)).chunk(2, dim=-1)
        out = self.out(modulate(self.out_norm(z), shift, scale))
        return out.view(b, h // p, w // p, ch, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, ch, h, w)

    def moe_layers(self) -> list[MoE]:
        return [m for m in self.modules() if isinstance(m, MoE)]

    def active_params(self) -> int:
        """Parameters one image token passes through (routed experts count A of E)."""
        total = sum(p.numel() for p in self.parameters())
        for m in self.moe_layers():
            total -= m.expert_params() * (m.experts - m.active)
        return total


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    freqs = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def positions(rows: int, cols: int, dim: int, device) -> torch.Tensor:
    """2D sin-cos embeddings of patch centers in [0, 1], so both resolutions share coordinates."""
    ys = (torch.arange(rows, device=device) + 0.5) / rows
    xs = (torch.arange(cols, device=device) + 0.5) / cols
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), -1).reshape(-1, 2) * 64
    freqs = torch.exp(-math.log(10000) * torch.arange(dim // 4, device=device) / (dim // 4))
    args = grid[:, :, None] * freqs  # [N, 2, dim/4]
    return torch.cat([args.sin(), args.cos()], dim=-1).reshape(-1, dim)

"""The LLM-style mixture-of-experts layer of PixelMoE.

Token-choice routing with sigmoid scores, fine-grained routed experts plus one always-on shared
expert, and bias-based load balancing: a per-expert bias steers which experts get picked, is
nudged after every step toward uniform load, and never enters the loss or the routing weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class MoE(nn.Module):
    def __init__(self, dim: int, experts: int = 32, active: int = 4, routed_hidden: int = 192,
                 shared_hidden: int = 288, gamma: float = 1e-3):
        super().__init__()
        self.router = nn.Linear(dim, experts, bias=False)
        # All routed experts as stacked SwiGLU weights, so they run as two batched matmuls.
        self.w_gate_up = nn.Parameter(torch.empty(experts, dim, 2 * routed_hidden).uniform_(-1, 1) / math.sqrt(dim))
        self.w_down = nn.Parameter(torch.empty(experts, routed_hidden, dim).uniform_(-1, 1) / math.sqrt(routed_hidden))
        self.shared = SwiGLU(dim, shared_hidden)
        self.register_buffer("bias", torch.zeros(experts))  # balancing bias: selection only
        self.register_buffer("load", torch.zeros(experts))  # load fraction F of the last batch
        self.experts, self.active, self.gamma, self.scale = experts, active, gamma, math.sqrt(active)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        scores = torch.sigmoid(self.router(x).float())  # router in fp32
        ids = torch.topk(scores + self.bias, self.active, dim=-1).indices  # [N, A]
        weights = scores.gather(1, ids)
        weights = self.scale * weights / weights.sum(-1, keepdim=True)

        flat = ids.flatten()
        counts = torch.bincount(flat, minlength=self.experts)
        if self.training:
            self.load = counts.float() / flat.numel()  # each token spreads 1/A over its A experts

        # Scatter (token, expert) pairs into a padded [E, capacity, dim] buffer, run every expert at once.
        order = flat.argsort()
        expert = flat[order]
        token = order // self.active
        offsets = torch.cumsum(counts, 0) - counts
        slot = torch.arange(len(order), device=x.device) - offsets[expert]
        buf = x.new_zeros(self.experts, int(counts.max()), x.shape[-1])
        buf[expert, slot] = x[token]
        gate, up = torch.bmm(buf, self.w_gate_up.to(x.dtype)).chunk(2, dim=-1)
        y = torch.bmm(F.silu(gate) * up, self.w_down.to(x.dtype))[expert, slot]
        out = torch.zeros_like(x).index_add_(0, token, y * weights.flatten()[order].unsqueeze(1).to(x.dtype))
        return (out + self.shared(x)).reshape(shape)

    def expert_params(self) -> int:
        """Parameters of one routed expert."""
        return (self.w_gate_up[0].numel() + self.w_down[0].numel())

    @torch.no_grad()
    def update_bias(self) -> None:
        """b <- b - gamma (F - Q) / RMS(F - Q), with Q uniform (appendix A of the paper)."""
        delta = self.load - 1.0 / self.experts
        self.bias -= self.gamma * delta / (delta.pow(2).mean().sqrt() + 1e-6)

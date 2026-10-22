"""Pixel-space hierarchical flow matching: denoise at half resolution first, then at full resolution.

Stage 0 runs from pure noise (t = 1) to a still-noisy half-resolution image (t = t_e0). Stage 1 runs
from a noisy, blurred full-resolution image (t = t_s1) to the clean image (t = 0). The model predicts
each stage's endpoint; the loss is the velocity-space MSE, which is the endpoint error over sigma^2.
Between the stages the sampler upsamples, rescales, and adds block-correlated noise so the state
matches exactly the Stage 1 starting distribution seen in training (appendix B of the paper).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

T_S1 = 6 / 7  # Stage 1 starts here ...
T_E0 = T_S1 / (2 - T_S1)  # ... which fixes the Stage 0 endpoint: 0.75
SIGMA_MIN = 0.05
STAGES = {0: (1.0, T_E0), 1: (T_S1, 0.0)}  # (t at stage start, t at stage end)


def down(x):
    return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True)


def up(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


def logit_normal(n, mu, sd, device):
    return torch.sigmoid(mu + sd * torch.randn(n, device=device))


def loss(model, x0: torch.Tensor, y: torch.Tensor, noise: float = 1.0, mu_g=0.8, sd_g=0.8, mu_l=0.6, sd_l=0.8):
    """Stage-aware flow-matching loss on clean images x0 in [-1, 1]."""
    b, dev = len(x0), x0.device
    # Stage allocation: a global Logit-Normal draw, the overlap of the two stages split evenly.
    tg = logit_normal(b, mu_g, sd_g, dev)
    stage = torch.where(tg > T_S1, 0, torch.where(tg < T_E0, 1, torch.randint(0, 2, (b,), device=dev)))
    total = x0.new_zeros(())
    for s in (0, 1):
        idx = (stage == s).nonzero().squeeze(1)
        if not len(idx):
            continue
        x, lab = x0[idx], y[idx]
        if s == 0:
            eps = noise * torch.randn_like(down(x))
            start, end = eps, (1 - T_E0) * down(x) + T_E0 * eps
        else:
            eps = noise * torch.randn_like(x)
            start, end = (1 - T_S1) * up(down(x)) + T_S1 * eps, x
        sigma = logit_normal(len(idx), mu_l, sd_l, dev)  # stage-relative time
        xs = sigma.view(-1, 1, 1, 1) * start + (1 - sigma.view(-1, 1, 1, 1)) * end
        t_start, t_end = STAGES[s]
        pred = model(xs, t_end + sigma * (t_start - t_end), lab)
        err = ((pred.float() - end) ** 2).mean((1, 2, 3)) / sigma.clamp_min(SIGMA_MIN) ** 2
        total = total + err.sum()
    return total / b


@torch.no_grad()
def sample(model, y: torch.Tensor, size: int, steps: int = 40, cfg: float = 1.5, noise: float = 1.0,
           null: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """Generate images for labels y: Heun steps at half resolution, transition, Heun steps at full."""
    dev = y.device
    null = model.cfg.classes if null is None else null
    x = noise * torch.randn(len(y), model.cfg.channels, size // 2, size // 2, device=dev, generator=generator)
    for s in (0, 1):
        if s == 1:
            # Rescale the coarse endpoint and add 2x2-block-correlated noise (rho = -1/3).
            z = torch.randn(len(y), model.cfg.channels, size, size, device=dev, generator=generator)
            z = z - up(F.avg_pool2d(z, 2))  # zero mean within each 2x2 block: correlation -1/3
            # alpha * eps' with alpha = sqrt(3)/2 * t_s1 and eps' = noise * 2/sqrt(3) * z
            x = (1 - T_S1) / (1 - T_E0) * up(x) + T_S1 * noise * z
        n = max(1, round(steps * (0.25 if s == 0 else 0.75)))
        sigmas = torch.linspace(1, 0, n + 1, device=dev)
        for i in range(n):
            s0, s1 = sigmas[i], sigmas[i + 1]
            v = _velocity(model, x, s0, s, y, cfg, null)
            x_next = x + (s1 - s0) * v
            if s1 > 0:  # Heun correction
                x_next = x + (s1 - s0) * 0.5 * (v + _velocity(model, x_next, s1, s, y, cfg, null))
            x = x_next
    return x.clamp(-1, 1)


def _velocity(model, x, sigma, stage, y, cfg, null):
    t_start, t_end = STAGES[stage]
    t = (t_end + sigma * (t_start - t_end)).expand(len(x))
    if cfg == 1.0:
        pred = model(x, t, y).float()
    else:
        both = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, torch.full_like(y, null)])).float()
        cond, uncond = both.chunk(2)
        pred = uncond + cfg * (cond - uncond)
    return (x - pred) / sigma.clamp_min(SIGMA_MIN)

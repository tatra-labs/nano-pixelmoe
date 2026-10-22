"""Where do the experts work? Per-patch expert maps on real images, and expert use across noise levels.

    python analyze.py runs/moe -o assets/experts.png
"""

import argparse
from pathlib import Path

import matplotlib
import torch

from pixelmoe import Config, PixelDiT, flow
from train import cifar10

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def routes(model, x, t, y, layer):
    """Top-1 expert of every patch in one MoE layer: [B, rows, cols]."""
    moe = model.moe_layers()[layer]
    seen = {}
    hook = moe.router.register_forward_hook(lambda m, i, o: seen.update(scores=torch.sigmoid(o.float()) + moe.bias))
    model(x, t, y)
    hook.remove()
    side = x.shape[-1] // model.cfg.patch
    return seen["scores"].argmax(-1).view(len(x), side, side)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--data", default="data/cifar10-train.parquet")
    p.add_argument("-o", "--out", default="assets/experts.png")
    p.add_argument("--layer", type=int, default=-1)
    args = p.parse_args()

    ckpt = torch.load(Path(args.run) / "ema.pt", map_location="cuda", weights_only=True)
    model = PixelDiT(Config(**ckpt["config"])).cuda().eval()
    model.load_state_dict(ckpt["model"])
    images, labels = cifar10(Path(args.data))
    pick = [next(i for i in range(len(labels)) if labels[i] == c) for c in (0, 1, 3, 5, 7, 8)]
    x0 = torch.tensor(images[pick], device="cuda").permute(0, 3, 1, 2).float() / 127.5 - 1
    y = torch.tensor(labels[pick], device="cuda")

    # Stage 1 input at a moderate noise level: a blurred, noisy version of each image.
    sigma = 0.3
    torch.manual_seed(0)
    start = (1 - flow.T_S1) * flow.up(flow.down(x0)) + flow.T_S1 * torch.randn_like(x0)
    xs = sigma * start + (1 - sigma) * x0
    maps = routes(model, xs, torch.full((len(x0),), sigma * flow.T_S1, device="cuda"), y, args.layer).cpu()

    # Expert use against the global timestep, over 512 images.
    n_exp = model.cfg.experts
    sel = torch.randint(0, len(labels), (512,))
    xb = torch.tensor(images[sel.numpy()], device="cuda").permute(0, 3, 1, 2).float() / 127.5 - 1
    yb = torch.tensor(labels[sel.numpy()], device="cuda")
    ts = torch.linspace(0.02, 0.84, 12)
    usage = []
    for t in ts:
        sig = float(t) / flow.T_S1
        noisy = sig * ((1 - flow.T_S1) * flow.up(flow.down(xb)) + flow.T_S1 * torch.randn_like(xb)) + (1 - sig) * xb
        r = routes(model, noisy, torch.full((len(xb),), float(t), device="cuda"), yb, args.layer).flatten()
        usage.append(torch.bincount(r, minlength=n_exp).float() / len(r))
    usage = torch.stack(usage).cpu()

    fig = plt.figure(figsize=(12, 4.6))
    grid = fig.add_gridspec(2, 7, width_ratios=[1] * 6 + [2.6], hspace=0.08, wspace=0.08)
    cmap = plt.get_cmap("tab20", n_exp)
    for i in range(6):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(((x0[i].permute(1, 2, 0).cpu() + 1) / 2).clamp(0, 1))
        ax.axis("off")
        ax = fig.add_subplot(grid[1, i])
        ax.imshow(maps[i], cmap=cmap, vmin=0, vmax=n_exp - 1, interpolation="nearest")
        ax.axis("off")
    ax = fig.add_subplot(grid[:, 6])
    ax.imshow(usage.T, aspect="auto", cmap="magma", origin="lower",
              extent=[float(ts[0]), float(ts[-1]), -0.5, n_exp - 0.5])
    ax.set_xlabel("noise level t")
    ax.set_ylabel("expert")
    ax.set_title("top-1 expert share per noise level", fontsize=10)
    fig.suptitle("PixelMoE routing: each patch's top expert (bottom), and how expert use shifts with noise (right)",
                 fontsize=11)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()

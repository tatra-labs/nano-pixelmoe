"""Side-by-side samples and loss curves for a dense run and a PixelMoE run.

    python report.py runs/dense runs/moe -o assets
"""

import argparse
import json
from pathlib import Path

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def label(run: Path) -> str:
    cfg = json.loads((run / "config.json").read_text())
    kind = "dense" if not cfg["config"]["experts"] else f"PixelMoE E{cfg['config']['experts']}A{cfg['config']['active']}"
    return f"{kind}: {cfg['params'] / 1e6:.0f}M params, {cfg['active_params'] / 1e6:.0f}M active"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dense")
    p.add_argument("moe")
    p.add_argument("-o", "--out", default="assets")
    args = p.parse_args()
    runs, out = [Path(args.dense), Path(args.moe)], Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    for ax, run in zip(axes, runs):
        grid = sorted(run.glob("samples_*.png"))[-1]
        ax.imshow(Image.open(grid))
        fid = sorted(run.glob("fid-cfg*.json"))
        score = f", FID {json.loads(fid[0].read_text())['fid']}" if fid else ""
        ax.set_title(label(run) + score, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"CIFAR-10, class-conditional, pixel space, same steps ({grid.stem.split('_')[1].lstrip('0')})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=130)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for run, color in zip(runs, ("#8A8F99", "#1F5FBF")):
        rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
        ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], color=color, lw=1.8, label=label(run))
    ax.set_yscale("log")
    ax.set_xlabel("training step (batch 128)")
    ax.set_ylabel("flow-matching loss (running mean)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=150)
    print(out / "samples.png", out / "loss.png")


if __name__ == "__main__":
    main()

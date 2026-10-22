"""Train PixelMoE (or its dense twin) on CIFAR-10, class-conditional, on one GPU.

    python train.py --out runs/moe                 # PixelMoE, E32A4
    python train.py --out runs/dense --experts 0   # dense baseline with matched active parameters
"""

import argparse
import copy
import io
import json
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from pixelmoe import Config, PixelDiT, flow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data/cifar10-train.parquet")
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--active", type=int, default=4)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--sample-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    images, labels = cifar10(Path(args.data))
    images = torch.tensor(images, device=dev).permute(0, 3, 1, 2)  # uint8 [50000, 3, 32, 32]
    labels = torch.tensor(labels, device=dev)

    cfg = Config(experts=args.experts, active=args.active)
    model = PixelDiT(cfg).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total = sum(p.numel() for p in model.parameters())
    info = {"config": asdict(cfg), "params": total, "active_params": model.active_params(), **vars(args)}
    (out / "config.json").write_text(json.dumps(info, indent=2))
    print(f"{total / 1e6:.1f}M parameters, {model.active_params() / 1e6:.1f}M active per image token")

    log = open(out / "log.jsonl", "a")
    t0, running = time.time(), None
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(images), (args.batch,), device=dev)
        x = images[idx].float() / 127.5 - 1
        x = torch.where(torch.rand(len(x), 1, 1, 1, device=dev) < 0.5, x.flip(3), x)  # horizontal flip
        y = torch.where(torch.rand(len(idx), device=dev) < 0.1, cfg.classes, labels[idx])  # label dropout
        for group in opt.param_groups:
            group["lr"] = args.lr * min(1.0, step / args.warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow.loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for moe in model.moe_layers():
            moe.update_bias()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - args.ema)
            for e, m in zip(ema.buffers(), model.buffers()):
                e.copy_(m)  # the balancing biases follow the live model
        running = loss.item() if running is None else 0.99 * running + 0.01 * loss.item()
        if step % 100 == 0:
            loads = [m.load.max().item() * m.experts for m in model.moe_layers()]
            row = {"step": step, "loss": round(running, 5), "seconds": round(time.time() - t0, 1),
                   "max_load": round(max(loads), 3) if loads else None}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % 1000 == 0:
                print(row, flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            ema_grid(ema, out / f"samples_{step:06d}.png")
            torch.save({"model": ema.state_dict(), "config": asdict(cfg), "step": step}, out / "ema.pt")


CIFAR10_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet"


def cifar10(path: Path):
    """CIFAR-10 train set as uint8 [50000, 32, 32, 3] and labels, downloaded once and cached as .npz."""
    cache = path.with_suffix(".npz")
    if not cache.exists():
        import pandas as pd

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(CIFAR10_URL, path)
        df = pd.read_parquet(path)
        images = np.stack([np.array(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in df["img"]])
        np.savez(cache, images=images, labels=df["label"].to_numpy())
    data = np.load(cache)
    return data["images"], data["labels"]


def ema_grid(model, path, per_class=10):
    y = torch.arange(model.cfg.classes, device="cuda").repeat_interleave(per_class)
    gen = torch.Generator("cuda").manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = flow.sample(model, y, model.cfg.image_size, generator=gen)
    save_image((x.float() + 1) / 2, path, nrow=per_class)


if __name__ == "__main__":
    main()

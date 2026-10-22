"""Generate a grid of images from a trained run.

    python sample.py runs/moe --classes 0 1 8 9 --per-class 8 --cfg 2.0 -o grid.png
"""

import argparse

import torch
from torchvision.utils import save_image

from pixelmoe import Config, PixelDiT, flow

CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--classes", type=int, nargs="+", default=list(range(10)), help=" ".join(
        f"{i}={n}" for i, n in enumerate(CIFAR10)))
    p.add_argument("--per-class", type=int, default=10)
    p.add_argument("--cfg", type=float, default=1.5, help="classifier-free guidance scale")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", default="samples.png")
    args = p.parse_args()

    ckpt = torch.load(f"{args.run}/ema.pt", map_location="cuda", weights_only=True)
    model = PixelDiT(Config(**ckpt["config"])).cuda().eval()
    model.load_state_dict(ckpt["model"])
    y = torch.tensor(args.classes, device="cuda").repeat_interleave(args.per_class)
    gen = torch.Generator("cuda").manual_seed(args.seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = flow.sample(model, y, model.cfg.image_size, args.steps, args.cfg, generator=gen)
    save_image((x.float() + 1) / 2, args.out, nrow=args.per_class)
    print(args.out)


if __name__ == "__main__":
    main()

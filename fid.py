"""FID (and Inception Score) of a trained run against the CIFAR-10 train set.

    python fid.py runs/moe --n 10000 --cfg 1.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch_fidelity
from PIL import Image

from pixelmoe import Config, PixelDiT, flow
from train import cifar10


def write_pngs(images: np.ndarray, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images):
        Image.fromarray(im).save(folder / f"{i:05d}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--data", default="data/cifar10-train.parquet")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch", type=int, default=500)
    args = p.parse_args()

    run = Path(args.run)
    reference = Path(args.data).parent / "cifar10-train-png"
    if not reference.exists():
        write_pngs(cifar10(Path(args.data))[0], reference)

    ckpt = torch.load(run / "ema.pt", map_location="cuda", weights_only=True)
    model = PixelDiT(Config(**ckpt["config"])).cuda().eval()
    model.load_state_dict(ckpt["model"])
    samples = run / f"samples-cfg{args.cfg:g}"
    labels = torch.arange(args.n, device="cuda") % model.cfg.classes
    gen = torch.Generator("cuda").manual_seed(0)
    images = []
    for i in range(0, args.n, args.batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = flow.sample(model, labels[i:i + args.batch], model.cfg.image_size, args.steps, args.cfg, generator=gen)
        images.append(((x.float() + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy())
    write_pngs(np.concatenate(images), samples)

    metrics = torch_fidelity.calculate_metrics(input1=str(samples), input2=str(reference), cuda=True, fid=True,
                                               isc=True, cache=True, input2_cache_name="cifar10-train-png",
                                               verbose=False)
    result = {"step": ckpt["step"], "cfg": args.cfg, "steps": args.steps, "samples": args.n,
              "fid": round(metrics["frechet_inception_distance"], 2),
              "inception_score": round(metrics["inception_score_mean"], 2)}
    (run / f"fid-cfg{args.cfg:g}.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()

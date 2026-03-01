"""
Generate side-by-side comparison grids: LR | SR | HR

Usage:
    python -m scripts.visualize --checkpoint checkpoints/best.pth \\
        --input-dir data/benchmark/Set14/HR \\
        --output-dir visualizations/ --scale 4
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dcno import DCNO
from models.diffusion import RectifiedFlow
from models.ema import EMA


def main():
    parser = argparse.ArgumentParser(description="Visualize DCNO SR results")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of HR images")
    parser.add_argument("--output-dir", type=str, default="visualizations")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    elif "config" in ckpt:
        cfg = ckpt["config"]

    scale = args.scale
    if cfg:
        scale = cfg["model"].get("scale_factor", scale)

    # Build model
    model_kwargs = {
        "in_channels": 3, "out_channels": 3,
        "hidden_dims": [64, 128, 256, 512], "depths": [2, 2, 4, 2],
        "dct_block_size": 8, "num_heads": 8, "mode_weighting": True,
        "scale_factor": scale, "dropout": 0.0, "transform_type": "dct",
    }
    if cfg:
        for k in model_kwargs:
            if k in cfg.get("model", {}):
                model_kwargs[k] = cfg["model"][k]

    model = DCNO(**model_kwargs).to(device)

    if "ema" in ckpt:
        ema = EMA(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    flow = RectifiedFlow(num_inference_steps=args.steps)

    # Process images
    to_tensor = transforms.ToTensor()
    os.makedirs(args.output_dir, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
    images = sorted([
        f for f in Path(args.input_dir).iterdir()
        if f.suffix.lower() in exts
    ])[:args.max_images]

    print(f"Processing {len(images)} images at x{scale}...")

    for img_path in images:
        hr = to_tensor(Image.open(img_path).convert("RGB")).to(device)
        _, H, W = hr.shape

        # Make LR
        lr = F.interpolate(
            hr.unsqueeze(0), size=(H // scale, W // scale),
            mode="bicubic", align_corners=False,
        ).clamp(0, 1)

        # SR
        block = 8
        H_pad = ((H + block - 1) // block) * block
        W_pad = ((W + block - 1) // block) * block

        with torch.no_grad():
            sr = flow.sample(model, lr, shape=(1, 3, H_pad, W_pad), device=device)
        sr = sr[:, :, :H, :W].clamp(0, 1)

        # Upsample LR for comparison
        lr_up = F.interpolate(lr, size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)

        # Grid: LR | SR | HR
        grid = torch.cat([lr_up[0], sr[0], hr], dim=-1)

        out_name = f"{img_path.stem}_comparison.png"
        save_image(grid, os.path.join(args.output_dir, out_name))
        print(f"  Saved: {out_name}")

    print(f"Done! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()

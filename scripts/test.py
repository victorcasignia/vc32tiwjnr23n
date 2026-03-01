"""
Testing / evaluation script for DCNO.

Evaluates trained DCNO on benchmark datasets with PSNR, SSIM, and LPIPS metrics.
Supports single-image inference and batch evaluation.

Usage:
    # Evaluate on benchmarks
    python -m scripts.test --config configs/dcno_div2k_x4.yaml \\
        --checkpoint checkpoints/best.pth \\
        --datasets set5 set14 bsd100 urban100

    # Single image super-resolution
    python -m scripts.test --checkpoint checkpoints/best.pth \\
        --input path/to/image.jpg --scale 4 --output sr_output.png

    # Adjust inference steps
    python -m scripts.test --config configs/dcno_div2k_x4.yaml \\
        --checkpoint checkpoints/best.pth --steps 20
"""

import argparse
import logging
import os
import sys
import math
import time
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dcno import DCNO
from models.diffusion import RectifiedFlow
from models.ema import EMA
from scripts.dataset import build_dataset


def _get_device(preference: str = "auto") -> torch.device:
    """Pick the best available device: cuda > mps > cpu."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, crop_border: int = 0) -> float:
    """
    PSNR on Y channel (ITU-R BT.601), matching standard SR evaluation.
    Inputs: (C, H, W) in [0, 1].
    """
    if crop_border > 0:
        pred = pred[:, crop_border:-crop_border, crop_border:-crop_border]
        target = target[:, crop_border:-crop_border, crop_border:-crop_border]

    # Convert to Y channel
    pred_y = _rgb_to_y(pred)
    target_y = _rgb_to_y(target)

    mse = F.mse_loss(pred_y, target_y).item()
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, crop_border: int = 0) -> float:
    """
    SSIM on Y channel, matching standard SR evaluation.
    Inputs: (C, H, W) in [0, 1].
    """
    if crop_border > 0:
        pred = pred[:, crop_border:-crop_border, crop_border:-crop_border]
        target = target[:, crop_border:-crop_border, crop_border:-crop_border]

    pred_y = _rgb_to_y(pred).squeeze(0).cpu().numpy()
    target_y = _rgb_to_y(target).squeeze(0).cpu().numpy()

    return _ssim_numpy(pred_y, target_y)


def _rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB to Y channel (ITU-R BT.601)."""
    # img: (3, H, W) in [0, 1]
    r, g, b = img[0:1], img[1:2], img[2:3]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0
    return y


def _ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two 2D arrays."""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    from scipy.ndimage import uniform_filter
    mu1 = uniform_filter(img1, size=11)
    mu2 = uniform_filter(img2, size=11)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = uniform_filter(img1 ** 2, size=11) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=11) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=11) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, lpips_fn=None) -> float:
    """Compute LPIPS if available."""
    if lpips_fn is None:
        return 0.0
    # lpips expects (B, C, H, W) in [-1, 1]
    pred_b = pred.unsqueeze(0) * 2 - 1
    target_b = target.unsqueeze(0) * 2 - 1
    with torch.no_grad():
        score = lpips_fn(pred_b, target_b)
    return score.item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    flow: RectifiedFlow,
    dataset_name: str,
    data_dir: str,
    scale_factor: int,
    device: torch.device,
    num_steps: int = 10,
    ode_solver: str = "euler",
    save_dir: Optional[str] = None,
    lpips_fn=None,
) -> Dict[str, float]:
    """Evaluate model on a benchmark dataset."""
    model.eval()

    ds = build_dataset(
        dataset_name, data_dir,
        scale_factor=scale_factor,
        patch_size=512,  # unused for test
        augment=False, split="test",
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    psnr_list = []
    ssim_list = []
    lpips_list = []
    times = []

    crop_border = scale_factor  # standard: crop `scale` pixels at borders

    for i, (hr, lr) in enumerate(tqdm(loader, desc=f"  {dataset_name}", leave=False)):
        hr, lr = hr.to(device), lr.to(device)
        B, C, H_hr, W_hr = hr.shape

        # Ensure HR dimensions are divisible by 8 for DCT
        block = 8
        H_hr_pad = ((H_hr + block - 1) // block) * block
        W_hr_pad = ((W_hr + block - 1) // block) * block

        t_start = time.time()

        # Generate SR
        target_shape = (B, C, H_hr_pad, W_hr_pad)
        sr = flow.sample(model, lr, shape=target_shape, num_steps=num_steps, device=device)
        sr = sr[:, :, :H_hr, :W_hr]  # crop to original HR size

        t_elapsed = time.time() - t_start
        times.append(t_elapsed)

        sr = sr.clamp(0, 1)

        # Metrics
        for b in range(B):
            psnr_val = compute_psnr(sr[b], hr[b], crop_border)
            ssim_val = compute_ssim(sr[b], hr[b], crop_border)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            if lpips_fn is not None:
                lpips_val = compute_lpips(sr[b], hr[b], lpips_fn)
                lpips_list.append(lpips_val)

        # Save SR images
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            from torchvision.utils import save_image
            save_image(sr[0], os.path.join(save_dir, f"{dataset_name}_{i:04d}_sr.png"))

    results = {
        "psnr": np.mean(psnr_list),
        "ssim": np.mean(ssim_list),
        "avg_time": np.mean(times),
        "num_images": len(psnr_list),
    }
    if lpips_list:
        results["lpips"] = np.mean(lpips_list)

    return results


def single_image_sr(
    model: nn.Module,
    flow: RectifiedFlow,
    input_path: str,
    output_path: str,
    scale_factor: int,
    device: torch.device,
    num_steps: int = 10,
):
    """Super-resolve a single image."""
    from torchvision import transforms
    from torchvision.utils import save_image
    from PIL import Image

    model.eval()

    # Load image
    img = Image.open(input_path).convert("RGB")
    lr = transforms.ToTensor()(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    _, _, H, W = lr.shape
    H_hr = H * scale_factor
    W_hr = W * scale_factor

    # Pad to multiples of 8
    block = 8
    H_pad = ((H_hr + block - 1) // block) * block
    W_pad = ((W_hr + block - 1) // block) * block

    log.info("Input: %d×%d → Output: %d×%d", H, W, H_hr, W_hr)
    log.info("Running %d-step ODE sampling...", num_steps)

    t0 = time.time()
    sr = flow.sample(model, lr, shape=(1, 3, H_pad, W_pad), num_steps=num_steps, device=device, show_progress=True)
    sr = sr[:, :, :H_hr, :W_hr].clamp(0, 1)
    elapsed = time.time() - t0

    save_image(sr[0], output_path)
    log.info("Saved to: %s (%.2fs)", output_path, elapsed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test DCNO")
    parser.add_argument("--config", type=str, default=None, help="YAML config (for dataset eval)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--datasets", nargs="+", default=["set5", "set14", "bsd100", "urban100"],
                        help="Benchmark datasets to evaluate")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--scale", type=int, default=None, help="SR scale factor")
    parser.add_argument("--steps", type=int, default=None, help="Override inference steps")
    parser.add_argument("--solver", type=str, default=None, choices=["euler", "midpoint", "adaptive"],
                        help="Override ODE solver")
    parser.add_argument("--input", type=str, default=None, help="Single image input path")
    parser.add_argument("--output", type=str, default="sr_output.png", help="Output path for single image")
    parser.add_argument("--save-dir", type=str, default=None, help="Save SR images to directory")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use EMA weights")
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS computation")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to run on (default: auto-detect)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = _get_device(args.device)

    # Load checkpoint
    log.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Get config from checkpoint or file
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    elif "config" in ckpt:
        cfg = ckpt["config"]
    else:
        raise ValueError("No config found. Provide --config or use a checkpoint that includes config.")

    # Overrides
    scale = args.scale or cfg["model"].get("scale_factor", 4)
    num_steps = args.steps or cfg["diffusion"].get("num_inference_steps", 10)
    solver = args.solver or cfg["diffusion"].get("ode_solver", "euler")
    data_dir = args.data_dir or cfg["data"].get("data_dir", "./data")

    # Build model
    model = DCNO(
        in_channels=cfg["model"].get("in_channels", 3),
        out_channels=cfg["model"].get("out_channels", 3),
        hidden_dims=cfg["model"].get("hidden_dims", [64, 128, 256, 512]),
        depths=cfg["model"].get("depths", [2, 2, 4, 2]),
        dct_block_size=cfg["model"].get("dct_block_size", 8),
        num_heads=cfg["model"].get("num_heads", 8),
        mode_weighting=cfg["model"].get("mode_weighting", True),
        scale_factor=scale,
        dropout=0.0,
        transform_type=cfg["model"].get("transform_type", "dct"),
    ).to(device)

    # Load weights (EMA if available)
    if args.use_ema and "ema" in ckpt:
        ema = EMA(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()
        log.info("  Using EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    # Diffusion
    flow = RectifiedFlow(
        num_inference_steps=num_steps,
        ode_solver=solver,
    )

    log.info("Scale: x%d | Steps: %d | Solver: %s", scale, num_steps, solver)

    # LPIPS
    lpips_fn = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex").to(device)
            log.info("  LPIPS: enabled (AlexNet)")
        except ImportError:
            log.info("  LPIPS: disabled (install with: pip install lpips)")

    # ---- Single image mode ----
    if args.input:
        single_image_sr(model, flow, args.input, args.output, scale, device, num_steps)
        return

    # ---- Benchmark evaluation ----
    log.info("")
    log.info("=" * 60)
    log.info("%-12s %8s %8s %8s %8s", "Dataset", "PSNR", "SSIM", "LPIPS", "Time")
    log.info("=" * 60)

    all_results = {}
    for ds_name in tqdm(args.datasets, desc="Benchmarks", unit="ds"):
        try:
            results = evaluate_dataset(
                model, flow, ds_name, data_dir, scale, device,
                num_steps=num_steps, ode_solver=solver,
                save_dir=args.save_dir, lpips_fn=lpips_fn,
            )
            all_results[ds_name] = results

            lpips_str = f"{results.get('lpips', 0):.4f}" if "lpips" in results else "  N/A "
            log.info(
                "%-12s %8.2f %8.4f %8s %7.2fs",
                ds_name, results['psnr'], results['ssim'], lpips_str, results['avg_time'],
            )
        except FileNotFoundError as e:
            log.warning("%s: SKIPPED — %s", ds_name, e)

    log.info("=" * 60)

    # Average across datasets
    if all_results:
        avg_psnr = np.mean([r["psnr"] for r in all_results.values()])
        avg_ssim = np.mean([r["ssim"] for r in all_results.values()])
        log.info("%-12s %8.2f %8.4f", "Average", avg_psnr, avg_ssim)


if __name__ == "__main__":
    main()

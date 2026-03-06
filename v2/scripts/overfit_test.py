"""
Overfit test: progressive difficulty check.

Runs three stages in sequence; each stage uses more images than the last.
A stage must reach >1 dB above the bicubic baseline to pass and unlock
the next stage.  Stops early on failure.

  Stage 1:  4 images  — sanity check (should pass easily)
  Stage 2: 16 images  — checks capacity to fit a small varied set
  Stage 3: 32 images  — harder diversity; still an overfit regime

Optimizer is set by OPTIMIZER constant below or via --optimizer CLI arg.

Run:
    python -m scripts.overfit_test                  # default (adamw)
    python -m scripts.overfit_test --optimizer musgd
    python -m scripts.overfit_test --stages 1       # only stage 1
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dwno import DWNOS, DWNOLoss
from scripts.dataset import SRDataset
from scripts.utils import psnr, ssim
import torch
import numpy as np
import random
import os

def set_seed_all(seed):
    """
    Sets the seed for pseudo-random number generators in:
    Python, NumPy, PyTorch (CPU and all GPUs).
    Also configures PyTorch to use deterministic algorithms.
    """
    print(f"Setting random seed to {seed}")
    
    # Set seeds for standard libraries
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # Set seeds for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU models
    
    # Configure deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enforce deterministic algorithms for native kernels if available (PyTorch 1.12+)
    # This might impact performance and may raise errors if no deterministic algo exists
    # torch.use_deterministic_algorithms(True) 
    
    # Handle DataLoader workers for full reproducibility
    # For fully deterministic data loading, a specific worker_init_fn is needed
    # or set num_workers=0 in your DataLoader.

# Example usage
SEED = 43
set_seed_all(SEED)


# ── defaults ──────────────────────────────────────────────────────────────
OPTIMIZER        = "adamw"   # "adam" | "adamw" | "musgd"
SCALE            = 4
PATCH_SIZE       = 128
LR_RATE          = 5e-4
N_STEPS          = 200
LOG_EVERY        = 10
PASS_THRESHOLD   = 1.0       # minimum gain (dB) over bicubic to pass a stage

# Stage definitions: (n_images, n_steps)
STAGES = [
    (4,  200),
    (16, 200),
    (32, 200),
]


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def _make_optimizer(model: nn.Module, name: str, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "musgd":
        from optim.musgd import MuSGD
        return MuSGD(model.parameters(), lr=lr, weight_decay=0)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)


# ---------------------------------------------------------------------------
# Single-stage runner
# ---------------------------------------------------------------------------

def run_stage(
    n_images: int,
    n_steps: int,
    optimizer_name: str,
    device: torch.device,
    scale: int       = SCALE,
    patch_size: int  = PATCH_SIZE,
    lr_rate: float   = LR_RATE,
    log_every: int   = LOG_EVERY,
) -> dict:
    """Train on n_images patches for n_steps steps. Returns result dict."""

    # ── Dataset ──────────────────────────────────────────────────────────
    ds = SRDataset(
        hr_dir     = "../data/DIV2K/DIV2K_train_HR",
        lr_dir     = "../data/DIV2K/DIV2K_train_LR_bicubic/X4",
        scale      = scale,
        patch_size = patch_size,
        augment    = False,
        cache      = False,
    )
    ds.hr_paths = ds.hr_paths[:n_images]
    loader      = torch.utils.data.DataLoader(ds, batch_size=n_images, shuffle=False, drop_last=True)
    _lr, _hr    = next(iter(loader))
    lr_img      = _lr.to(device)
    hr_img      = _hr.to(device)
    del loader, ds, _lr, _hr

    # ── Model ─────────────────────────────────────────────────────────────
    model = DWNOS(
        scale        = scale,
        channels     = 64,
        stage_depths = [4, 4, 4],
        levels       = 2,
        kernel_size  = 7,
        mlp_ratio    = 4.0,
        drop_path    = 0.0,
        filter_len   = 3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # ── Bicubic baseline ─────────────────────────────────────────────────
    with torch.no_grad():
        bicubic  = F.interpolate(lr_img, scale_factor=scale, mode="bicubic", align_corners=False)
        bic_psnr = psnr(bicubic.clamp(0, 1), hr_img)

    print(f"  Params: {n_params:.2f}M   Bicubic baseline: {bic_psnr:.2f} dB")

    # ── Training ──────────────────────────────────────────────────────────
    optimizer = _make_optimizer(model, optimizer_name, lr_rate)
    criterion = DWNOLoss(
        lambda_ssim=0.1,
        lambda_wave=0.15,
        lambda_orth=1e-4,
        wave_weight_edge=1.0,
        wave_weight_diag=2.0,
    ).to(device)

    model.train()
    print(f"  {'Step':>5} | {'Loss':>8} | {'L1':>8} | {'PSNR':>8} | {'SSIM':>7}")
    print("  " + "-" * 50)

    for step in range(1, n_steps + 1):
        sr         = model(lr_img)
        loss, info = criterion(sr, hr_img, model)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                sr_e = model(lr_img).clamp(0, 1)
                p    = psnr(sr_e, hr_img)
                s    = ssim(sr_e.float(), hr_img.float())
            print(f"  {step:5d} | {info['loss_total']:8.4f} | {info['loss_l1']:8.4f}"
                  f" | {p:8.2f} | {s:7.4f}")

    # ── Final eval ────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        sr_final   = model(lr_img).clamp(0, 1)
        final_psnr = psnr(sr_final, hr_img)
        final_ssim = ssim(sr_final.float(), hr_img.float())

    gain   = final_psnr - bic_psnr
    passed = gain >= PASS_THRESHOLD

    print(f"\n  Final PSNR : {final_psnr:.2f} dB  "
          f"(bicubic: {bic_psnr:.2f} dB,  gain: {gain:+.2f} dB)")
    print(f"  Final SSIM : {final_ssim:.4f}")

    # ── Save sample images ────────────────────────────────────────────────
    try:
        from torchvision.utils import save_image
        out_dir = Path(f"checkpoints/overfit/stage_{n_images}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(4, lr_img.shape[0])):
            lr_up = F.interpolate(lr_img[i:i+1], scale_factor=scale,
                                  mode="bicubic", align_corners=False).clamp(0, 1)
            row = torch.cat([lr_up, sr_final[i:i+1], hr_img[i:i+1]], dim=3)
            save_image(row[0], out_dir / f"sample_{i}_lr_sr_hr.png")
        print(f"  Images     → {out_dir.resolve()}")
    except Exception as e:
        print(f"  (image save skipped: {e})")

    return {
        "n_images":   n_images,
        "final_psnr": final_psnr,
        "bic_psnr":   bic_psnr,
        "gain":       gain,
        "ssim":       final_ssim,
        "passed":     passed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DWNO-SR overfit test")
    parser.add_argument("--optimizer", default=OPTIMIZER,
                        choices=["adam", "adamw", "musgd"],
                        help="Optimizer (default: %(default)s)")
    parser.add_argument("--stages", type=int, default=len(STAGES),
                        help=f"Number of stages to run 1-{len(STAGES)} (default: all)")
    parser.add_argument("--steps", type=int, default=N_STEPS,
                        help="Training steps per stage (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=LR_RATE,
                        help="Learning rate (default: %(default)s)")
    parser.add_argument("--device", default=None,
                        help="Device override: mps / cuda / cpu")
    args = parser.parse_args()

    default_device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(args.device or default_device)
    print(f"Device: {device}  |  Optimizer: {args.optimizer}  |  Steps/stage: {args.steps}")
    print("=" * 60)

    results    = []
    n_stages   = min(args.stages, len(STAGES))
    for idx, (n_images, _) in enumerate(STAGES[:n_stages]):
        print(f"\nStage {idx+1}/{n_stages}  —  {n_images} images × {args.steps} steps")
        print("-" * 60)

        result = run_stage(
            n_images       = n_images,
            n_steps        = args.steps,
            optimizer_name = args.optimizer,
            device         = device,
            lr_rate        = args.lr,
        )
        results.append(result)

        label = "✅ PASSED" if result["passed"] else "❌ FAILED"
        print(f"\n  {label}  gain={result['gain']:+.2f} dB  "
              f"(threshold: +{PASS_THRESHOLD:.1f} dB)\n")

        if not result["passed"]:
            print("Stopping early — network did not clear the threshold.")
            break

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("Summary")
    print("-" * 60)
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{tag}]  {r['n_images']:>2} images  "
              f"PSNR={r['final_psnr']:.2f} dB  "
              f"(bic={r['bic_psnr']:.2f})  "
              f"gain={r['gain']:+.2f} dB  "
              f"SSIM={r['ssim']:.4f}")

    all_passed = all(r["passed"] for r in results) and len(results) == n_stages
    if all_passed:
        print(f"\n✅ All {n_stages} stages passed — network handles varied data.")
    elif len(results) > 0 and results[-1]["passed"]:
        print(f"\n⚠️  Passed {len(results)}/{n_stages} stages.")
    else:
        print(f"\n❌ Failed at stage {len(results)} ({results[-1]['n_images']} images).")


if __name__ == "__main__":
    main()

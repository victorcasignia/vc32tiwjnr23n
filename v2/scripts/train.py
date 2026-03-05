"""
Training script for DWNO-SR.

Usage
-----
From v2/:
    python -m scripts.train --config configs/dwno_div2k_x4.yaml

Supports:
  - Mixed-precision (AMP) training
  - Gradient accumulation
  - Gradient clipping
  - EMA (exponential moving average) of weights
  - wandb logging
  - Checkpoint resume
  - Periodic validation with PSNR/SSIM
"""

import argparse
import copy
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Make sure we can import from v2/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DWNOS, DWNOLoss
from scripts.dataset import make_train_loader, make_val_loader
from scripts.utils import (
    AverageMeter,
    CosineWarmupScheduler,
    Timer,
    count_params,
    load_checkpoint,
    model_summary,
    psnr,
    save_checkpoint,
    ssim,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# wandb image panel helper
# ---------------------------------------------------------------------------

def make_wandb_image_panels(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    n: int = 4,
) -> list:
    """
    Build a list of wandb.Image objects showing LR | SR | HR side-by-side.
    Inputs are float tensors in [0, 1], shape (B, C, H, W).
    Returns up to `n` wandb.Image panels.
    """
    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage
    import numpy as np

    panels = []
    for i in range(min(n, lr.shape[0])):
        lr_i = lr[i].cpu().clamp(0, 1)
        sr_i = sr[i].cpu().clamp(0, 1)
        hr_i = hr[i].cpu().clamp(0, 1)

        lr_pil = TF.to_pil_image(lr_i)
        sr_pil = TF.to_pil_image(sr_i)
        hr_pil = TF.to_pil_image(hr_i)

        # Resize LR to HR size for visual comparison
        hr_w, hr_h = hr_pil.size
        lr_up = lr_pil.resize((hr_w, hr_h), PILImage.NEAREST)

        # Horizontal concat: LR(up) | SR | HR
        gap     = 4
        total_w = hr_w * 3 + gap * 2
        panel   = PILImage.new("RGB", (total_w, hr_h), color=(30, 30, 30))
        panel.paste(lr_up,  (0,              0))
        panel.paste(sr_pil, (hr_w + gap,     0))
        panel.paste(hr_pil, (hr_w * 2 + gap * 2, 0))

        img_psnr = psnr(sr[i:i+1].float(), hr[i:i+1].float())
        panels.append(wandb.Image(panel, caption=f"LR | SR | HR   PSNR={img_psnr:.2f}dB"))

    return panels


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential Moving Average of model weights.
    Useful for stabilising training — the EMA model is used for validation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)

    def eval_model(self) -> nn.Module:
        return self.shadow


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _center_crop_pair(
    lr: torch.Tensor,
    hr: torch.Tensor,
    hr_crop: int,
) -> tuple:
    """
    Center-crop hr to (hr_crop × hr_crop) and lr proportionally.
    The lr crop size is hr_crop // scale, where scale = hr.H / lr.H.
    If either spatial dim is smaller than the crop size it is left unchanged.
    """
    scale  = hr.shape[-2] // lr.shape[-2]          # infer upscale factor
    lc     = hr_crop // scale                       # LR crop size

    # HR crop
    H, W   = hr.shape[-2], hr.shape[-1]
    th, tw = min(hr_crop, H), min(hr_crop, W)
    top    = (H - th) // 2
    left   = (W - tw) // 2
    hr     = hr[..., top : top + th, left : left + tw]

    # LR crop (same centre)
    lH, lW = lr.shape[-2], lr.shape[-1]
    lth, ltw = min(lc, lH), min(lc, lW)
    ltop   = (lH - lth) // 2
    lleft  = (lW - ltw) // 2
    lr     = lr[..., ltop : ltop + lth, lleft : lleft + ltw]

    return lr, hr


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    amp: bool,
    num_sample_images: int = 4,
    crop_size: int = None,
) -> dict:
    """
    Validate the model.

    crop_size : if set, center-crops every HR image to (crop_size × crop_size)
                and the LR image proportionally before inference.  Useful when
                the validation set contains variable-resolution images.
    """
    model.eval()
    psnr_m = AverageMeter("PSNR")
    ssim_m = AverageMeter("SSIM")

    # Accumulate up to num_sample_images examples for image logging
    sample_lrs, sample_srs, sample_hrs = [], [], []

    for batch_idx, (lr, hr) in enumerate(tqdm(loader, desc="  Val", leave=False, ncols=80)):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        if crop_size is not None:
            lr, hr = _center_crop_pair(lr, hr, crop_size)

        with autocast('cuda', enabled=amp):
            sr = model(lr)

        sr_f = sr.clamp(0, 1).float()
        hr_f = hr.float()
        psnr_m.update(psnr(sr_f, hr_f), lr.shape[0])
        ssim_m.update(ssim(sr_f, hr_f), lr.shape[0])

        # Accumulate sample images until we have enough
        if num_sample_images > 0 and len(sample_lrs) < num_sample_images:
            need = num_sample_images - len(sample_lrs)
            sample_lrs.append(lr[:need].cpu())
            sample_srs.append(sr_f[:need].cpu())
            sample_hrs.append(hr_f[:need].cpu())

    sample_lr = torch.cat(sample_lrs, dim=0) if sample_lrs else None
    sample_sr = torch.cat(sample_srs, dim=0) if sample_srs else None
    sample_hr = torch.cat(sample_hrs, dim=0) if sample_hrs else None

    model.train()
    return {
        "psnr":       psnr_m.avg,
        "ssim":       ssim_m.avg,
        "sample_lr":  sample_lr,
        "sample_sr":  sample_sr,
        "sample_hr":  sample_hr,
    }


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    Build optimizer from config.

    cfg["optimizer"] : "adam" | "adamw" (default) | "musgd"
    cfg["lr"]        : base learning rate
    cfg["weight_decay"]: weight decay

    For adam / adamw, lifting-filter weights get 10× lower LR so the
    learned wavelets stay close to their bi-orthogonal initialization.
    musgd treats all parameters uniformly (Newton-Schulz handles each
    weight matrix in parameter-agnostic fashion).
    """
    name = cfg.get("optimizer", "adamw").lower()
    lr   = float(cfg.get("lr", 2e-4))
    wd   = float(cfg.get("weight_decay", 1e-2))

    if name == "musgd":
        from optim.musgd import MuSGD
        print(f"[Optim] MuSGD  lr={lr}  wd={wd}")
        return MuSGD(
            model.parameters(),
            lr            = lr,
            momentum      = float(cfg.get("momentum",      0.9)),
            muon_momentum = float(cfg.get("muon_momentum", 0.95)),
            weight_decay  = wd,
            nesterov      = cfg.get("nesterov", True),
        )

    # Split lifting-filter params to a lower LR group
    lift_params  = [p for n, p in model.named_parameters()
                    if "weight" in n and "lifting" in n.lower()]
    other_params = [p for n, p in model.named_parameters()
                    if not ("weight" in n and "lifting" in n.lower())]
    param_groups = [
        {"params": other_params, "lr": lr},
        {"params": lift_params,  "lr": lr * 0.1, "name": "lifting"},
    ]

    if name == "adam":
        print(f"[Optim] Adam   lr={lr}  wd={wd}")
        return optim.Adam(param_groups, weight_decay=wd)
    else:  # adamw (default)
        print(f"[Optim] AdamW  lr={lr}  wd={wd}")
        return optim.AdamW(param_groups, weight_decay=wd)


# ---------------------------------------------------------------------------
# Main Train Loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Train] Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader = make_train_loader(cfg)
    val_loader   = make_val_loader(cfg)
    print(f"[Train] Train batches: {len(train_loader)}  Val images: {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────────────
    model = DWNOS(
        scale        = cfg["scale"],
        in_channels  = cfg.get("in_channels", 3),
        channels     = cfg.get("channels", 64),
        stage_depths = cfg.get("stage_depths", [4, 4, 4]),
        levels       = cfg.get("levels", 2),
        kernel_size  = cfg.get("kernel_size", 7),
        mlp_ratio    = cfg.get("mlp_ratio", 4.0),
        drop_path    = cfg.get("drop_path", 0.05),
        filter_len   = cfg.get("filter_len", 3),
    ).to(device)
    model_summary(model)

    # EMA
    use_ema = cfg.get("ema", True)
    ema     = EMA(model, decay=cfg.get("ema_decay", 0.999)) if use_ema else None

    # ── Loss ─────────────────────────────────────────────────────────────
    criterion = DWNOLoss(
        lambda_ssim = cfg.get("lambda_ssim", 0.1),
        lambda_wave = cfg.get("lambda_wave", 0.05),
        lambda_orth = cfg.get("lambda_orth", 1e-4),
        wave_levels = cfg.get("wave_loss_levels", 3),
    ).to(device)

    # ── Optimiser ────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)

    # ── LR Scheduler ─────────────────────────────────────────────────────
    # Compute total_steps from actual training duration so warmup/decay
    # are proportional to real epoch count, not an arbitrary large number.
    _epochs      = cfg.get("epochs", 1000)
    _n_batches   = len(train_loader)
    total_steps  = _epochs * _n_batches
    # Warmup: config value if given, else 5% of total (min 50 steps)
    warmup_steps = max(50, cfg.get("warmup_steps", total_steps // 20))
    print(f"[Train] total_steps={total_steps}  warmup={warmup_steps} ({warmup_steps/_n_batches:.1f} epochs)")
    scheduler     = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

    # ── AMP Scaler ───────────────────────────────────────────────────────
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    scaler  = GradScaler('cuda', enabled=use_amp)

    # ── Grad accum ───────────────────────────────────────────────────────
    accum_steps = cfg.get("grad_accum", 1)
    clip_norm   = cfg.get("clip_norm", 1.0)

    # ── Resume ───────────────────────────────────────────────────────────
    ckpt_dir  = cfg.get("ckpt_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0
    global_step = 0
    best_psnr   = 0.0

    resume_path = cfg.get("resume")
    if resume_path and os.path.exists(resume_path):
        print(f"[Train] Resuming from {resume_path}")
        state = load_checkpoint(model, optimizer, scheduler, resume_path, device)
        start_epoch = state.get("epoch", 0) + 1
        global_step = state.get("step", 0)
        best_psnr   = state.get("best_psnr", 0.0)
        if ema:
            ema_path = resume_path.replace(".pth", "_ema.pth")
            if os.path.exists(ema_path):
                ema_state = torch.load(ema_path, map_location=device)
                ema.shadow.load_state_dict(ema_state)

    # ── wandb ────────────────────────────────────────────────────────────
    use_wandb = cfg.get("wandb", False) and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project = cfg.get("wandb_project", "dwno-sr"),
            name    = cfg.get("run_name", "dwno"),
            config  = cfg,
            resume  = "allow",
        )
        wandb.watch(model, log="gradients", log_freq=500)

    # ── Training ─────────────────────────────────────────────────────────
    epochs       = _epochs      # already computed above
    val_freq     = cfg.get("val_freq", 5)
    log_freq     = cfg.get("log_freq", 100)
    save_freq    = cfg.get("save_freq", 5)

    model.train()

    for epoch in range(start_epoch, epochs):
        loss_m   = AverageMeter("loss")
        l1_m     = AverageMeter("l1")
        ssim_m_  = AverageMeter("ssim")
        wave_m   = AverageMeter("wave")
        psnr_m_t = AverageMeter("psnr_train")

        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1:04d}",
            ncols=100,
        )

        for batch_idx, (lr, hr) in pbar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            # Forward  (model in AMP, loss in fp32 for numerical stability)
            with autocast('cuda', enabled=use_amp):
                sr = model(lr)
            loss, info = criterion(sr.float(), hr.float(), model)
            loss = loss / accum_steps

            # Backward
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                # Unscale + clip
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # Warn if scaler is skipping steps (inf/nan grads)
                if scaler.get_scale() < 1.0:
                    print(f"  [warn] GradScaler scale={scaler.get_scale():.1f}")

                if ema:
                    ema.update(model)

            # Metrics
            B = lr.shape[0]
            loss_m.update(info["loss_total"], B)
            l1_m.update(info["loss_l1"],    B)
            ssim_m_.update(info["loss_ssim"], B)
            wave_m.update(info["loss_wave"],  B)

            # Per-batch PSNR (computed on detached SR vs HR)
            with torch.no_grad():
                batch_psnr = psnr(sr.detach().clamp(0, 1).float(), hr.float())
            psnr_m_t.update(batch_psnr, B)

            pbar.set_postfix(
                loss=f"{loss_m.avg:.4f}",
                psnr=f"{psnr_m_t.avg:.2f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                refresh=False,
            )

            # Per-step wandb log
            if global_step % log_freq == 0 and use_wandb:
                wandb.log(
                    {
                        "train/loss":       info["loss_total"],
                        "train/l1":         info["loss_l1"],
                        "train/ssim_loss":  info["loss_ssim"],
                        "train/wave_loss":  info["loss_wave"],
                        "train/orth_loss":  info["loss_orth"],
                        "train/psnr":       batch_psnr,
                        "lr":               scheduler.get_last_lr()[0],
                        "step":             global_step,
                    }
                )

        # ── Per-epoch wandb log (train averages) ────────────────────────
        if use_wandb:
            wandb.log({
                "train_epoch/loss":       loss_m.avg,
                "train_epoch/l1":         l1_m.avg,
                "train_epoch/ssim_loss":  ssim_m_.avg,
                "train_epoch/wave_loss":  wave_m.avg,
                "train_epoch/psnr":       psnr_m_t.avg,
                "epoch": epoch + 1,
                "step":  global_step,
            })

        # ── Validation ────────────────────────────────────────────────────
        if (epoch + 1) % val_freq == 0:
            val_model = ema.eval_model() if ema else model
            val_stats = validate(
                val_model, val_loader, device, use_amp,
                num_sample_images=cfg.get("wandb_num_images", 4),
                crop_size=cfg.get("val_crop_size", None),
            )
            vp  = val_stats["psnr"]
            vs  = val_stats["ssim"]
            print(
                f"  [Val] Epoch {epoch+1}  PSNR={vp:.3f} dB  SSIM={vs:.4f}"
                + ("  ← best" if vp > best_psnr else "")
            )

            is_best = vp > best_psnr
            best_psnr = max(best_psnr, vp)

            if use_wandb:
                log_payload = {
                    "val/psnr":      vp,
                    "val/ssim":      vs,
                    "val/best_psnr": best_psnr,
                    "epoch":         epoch + 1,
                    "step":          global_step,
                }
                # Image panels: LR | SR | HR
                if (
                    val_stats["sample_lr"] is not None
                    and WANDB_AVAILABLE
                ):
                    panels = make_wandb_image_panels(
                        val_stats["sample_lr"],
                        val_stats["sample_sr"],
                        val_stats["sample_hr"],
                        n=cfg.get("wandb_num_images", 4),
                    )
                    log_payload["val/images"] = panels
                wandb.log(log_payload)

            if (epoch + 1) % save_freq == 0 or is_best:
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, global_step, vp, best_psnr,
                    ckpt_dir, is_best=is_best,
                )
                if ema and is_best:
                    torch.save(
                        ema.shadow.state_dict(),
                        os.path.join(ckpt_dir, "best_ema.pth"),
                    )

    if use_wandb:
        wandb.finish()

    print(f"\n[Train] Done. Best PSNR: {best_psnr:.3f} dB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train DWNO-SR")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--device", default=None, help="cuda / cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.resume:
        cfg["resume"] = args.resume
    if args.device:
        cfg["device"] = args.device

    train(cfg)

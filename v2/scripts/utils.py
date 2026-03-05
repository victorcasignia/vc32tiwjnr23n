"""
Utility functions: PSNR, SSIM metrics, checkpoint management, LR scheduling.
"""

import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
    """PSNR in dB.  Inputs should be float in [0, max_val]."""
    mse = F.mse_loss(sr, hr).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def ssim(sr: torch.Tensor, hr: torch.Tensor, window_size: int = 11) -> float:
    """Compute mean SSIM over a batch."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    B, C, H, W = sr.shape

    # Gaussian kernel
    sigma = 1.5
    coords = torch.arange(window_size, dtype=sr.dtype, device=sr.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(C, 1, window_size, window_size)
    pad    = window_size // 2

    mu_x  = F.conv2d(sr, kernel, padding=pad, groups=C)
    mu_y  = F.conv2d(hr, kernel, padding=pad, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x2 = F.conv2d(sr * sr, kernel, padding=pad, groups=C) - mu_x2
    sig_y2 = F.conv2d(hr * hr, kernel, padding=pad, groups=C) - mu_y2
    sig_xy = F.conv2d(sr * hr, kernel, padding=pad, groups=C) - mu_xy

    numer = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    denom = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
    return (numer / denom).mean().item()


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    val_psnr: float,
    best_psnr: float,
    ckpt_dir: str,
    is_best: bool = False,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "epoch":     epoch,
        "step":      step,
        "val_psnr":  val_psnr,
        "best_psnr": best_psnr,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }
    torch.save(state, os.path.join(ckpt_dir, "latest.pth"))
    if is_best:
        torch.save(state, os.path.join(ckpt_dir, "best.pth"))


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    ckpt_path: str,
    device: torch.device,
) -> dict:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state


# ---------------------------------------------------------------------------
# Learning Rate Schedule: Cosine with Warmup
# ---------------------------------------------------------------------------

class CosineWarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Linear warmup followed by cosine annealing.

    lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*(t-T_warm)/(T-T_warm)))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_min_frac: float = 0.01,   # lr_min = lr_max * lr_min_frac
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.lr_min_frac  = lr_min_frac
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            lr_min = base_lr * self.lr_min_frac
            if t < self.warmup_steps:
                lr = base_lr * (t + 1) / self.warmup_steps
            else:
                progress = (t - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ---------------------------------------------------------------------------
# Timing context
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# Model parameter count
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module):
    n = count_params(model)
    print(f"[Model] {model.__class__.__name__} — {n/1e6:.2f}M trainable parameters")

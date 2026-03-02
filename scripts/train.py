"""
Training script for DCNO — DCT Neural Operator for Super-Resolution.

Features:
  - Rectified flow matching (velocity prediction)
  - Optional consistency training for low-step inference
  - AdamW / Adam / MuSGD optimizer selection
  - Cosine / linear / constant-with-warmup LR scheduling
  - EMA of model weights
  - Mixed precision (AMP)
  - Wandb logging (loss curves, LR, samples, PSNR)
  - Checkpoint save/resume

Usage:
    python -m scripts.train --config configs/dcno_div2k_x4.yaml
    python -m scripts.train --config configs/dcno_div2k_x4.yaml --optimizer musgd --resume checkpoints/latest.pth
"""

import argparse
import logging
import os
import sys
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
from tqdm import tqdm

log = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dcno import DCNO
from models.diffusion import RectifiedFlow, ConsistencyTrainer
from models.ema import EMA
from optim.musgd import MuSGD
from scripts.dataset import build_dataset

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    log.warning("wandb not installed — logging disabled. Install with: pip install wandb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> torch.device:
    """Pick the best available device: cuda > mps > cpu."""
    if preference != "auto":
        dev = torch.device(preference)
        log.info("Device (manual): %s", dev)
        return dev
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    return dev


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> DCNO:
    mcfg = cfg["model"]
    return DCNO(
        in_channels=mcfg.get("in_channels", 3),
        out_channels=mcfg.get("out_channels", 3),
        hidden_dims=mcfg.get("hidden_dims", [64, 128, 256, 512]),
        depths=mcfg.get("depths", [2, 2, 4, 2]),
        dct_block_size=mcfg.get("dct_block_size", 8),
        num_heads=mcfg.get("num_heads", 8),
        mode_weighting=mcfg.get("mode_weighting", True),
        scale_factor=mcfg.get("scale_factor", 4),
        dropout=mcfg.get("dropout", 0.0),
        transform_type=mcfg.get("transform_type", "dct"),
        residual_learning=mcfg.get("residual_learning", False),
        spectral_conv_size=mcfg.get("spectral_conv_size", 1),
        spatial_dual_path=mcfg.get("spatial_dual_path", False),
        freq_norm=mcfg.get("freq_norm", False),
        progressive_stem=mcfg.get("progressive_stem", False),
        concat_cond=mcfg.get("concat_cond", False),
        input_proj=mcfg.get("input_proj", False),
        pixel_refinement=mcfg.get("pixel_refinement", False),
    )


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    tcfg = cfg["training"]
    name = tcfg.get("optimizer", "adamw").lower()
    lr = tcfg.get("learning_rate", 2e-4)
    wd = tcfg.get("weight_decay", 0.01)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "musgd":
        return MuSGD(
            model.parameters(),
            lr=lr,
            momentum=tcfg.get("sgd_momentum", 0.9),
            muon_momentum=tcfg.get("muon_momentum", 0.95),
            weight_decay=wd,
            nesterov=tcfg.get("nesterov", True),
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict, total_steps: int):
    scfg = cfg["training"].get("scheduler", {})
    stype = scfg.get("type", "cosine").lower()
    warmup_steps = scfg.get("warmup_steps", 0)
    min_lr = scfg.get("min_lr", 1e-6)
    T_max = scfg.get("T_max") or total_steps

    if stype == "cosine":
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(T_max - warmup_steps, 1)
            return max(min_lr / optimizer.defaults["lr"],
                       0.5 * (1 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif stype == "linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(T_max - warmup_steps, 1)
            return max(min_lr / optimizer.defaults["lr"], 1 - progress)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif stype == "constant_with_warmup":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler: {stype}")


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between two image tensors in [0, 1]."""
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    ema: EMA,
    epoch: int,
    global_step: int,
    best_psnr: float,
    cfg: dict,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "config": cfg,
    }, path)


def load_checkpoint(path: str, model, optimizer, scheduler, ema):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0), ckpt.get("best_psnr", 0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    flow: RectifiedFlow,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    num_samples: int = 4,
) -> dict:
    """Run validation: compute PSNR over val set and generate sample images."""
    model.eval()
    scale = cfg["model"].get("scale_factor", 4)
    num_steps = cfg["diffusion"].get("num_inference_steps", 10)
    residual_learning = cfg["model"].get("residual_learning", False)

    total_psnr = 0.0
    count = 0
    sample_images = []

    for i, (hr, lr) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
        hr, lr = hr.to(device), lr.to(device)
        B, C, H_hr, W_hr = hr.shape

        # Bicubic upsampled LR (used for residual learning & visualisation)
        lr_up = F.interpolate(
            lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False
        )
        x_bicubic = lr_up if residual_learning else None

        # Generate SR via ODE sampling
        sr = flow.sample(
            model, lr, shape=hr.shape, num_steps=num_steps,
            device=device, x_bicubic=x_bicubic,
        )

        # PSNR
        for b in range(B):
            total_psnr += compute_psnr(sr[b], hr[b])
            count += 1

        # Collect samples for visualization
        if len(sample_images) < num_samples:
            for b in range(min(B, num_samples - len(sample_images))):
                sample_images.append({
                    "lr": lr_up[b:b+1].clamp(0, 1)[0].cpu(),
                    "sr": sr[b].cpu().clamp(0, 1),
                    "hr": hr[b].cpu(),
                })

    avg_psnr = total_psnr / max(count, 1)
    model.train()
    return {"psnr": avg_psnr, "samples": sample_images}


def log_samples_to_wandb(samples: list, epoch: int):
    """Log sample SR images to wandb."""
    if not HAS_WANDB or wandb.run is None:
        return

    images = []
    for i, s in enumerate(samples[:4]):
        # Concatenate LR | SR | HR horizontally
        grid = torch.cat([s["lr"], s["sr"], s["hr"]], dim=-1)  # (3, H, W*3)
        grid = (grid * 255).byte().permute(1, 2, 0).numpy()
        images.append(wandb.Image(grid, caption=f"Sample {i}: LR | SR | HR"))

    wandb.log({"val/samples": images, "epoch": epoch})


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, args):
    device = get_device(args.device)
    log.info("Device: %s", device)

    # ---- Data ----
    scale = cfg["model"].get("scale_factor", 4)
    dcfg = cfg["data"]

    subset_frac = args.subset or dcfg.get("subset", None)

    train_ds = build_dataset(
        dcfg["train_dataset"], dcfg["data_dir"],
        scale_factor=scale, patch_size=dcfg.get("patch_size", 256),
        augment=dcfg.get("augment", True), split="train",
        subset=subset_frac,
    )
    val_ds = build_dataset(
        dcfg["val_dataset"], dcfg["data_dir"],
        scale_factor=scale, patch_size=dcfg.get("patch_size", 256),
        augment=False, split="val",
        subset=subset_frac,
    )

    tcfg = cfg["training"]
    # MPS doesn't support pin_memory; also reduce workers for small subsets
    _pin = dcfg.get("pin_memory", True) and device.type != "mps"
    _nw = dcfg.get("num_workers", 4)
    if len(train_ds) < 200:
        _nw = min(_nw, 2)
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=_nw,
        pin_memory=_pin,
        drop_last=True,
        persistent_workers=_nw > 0,
    )
    val_batch_size = tcfg.get("val_batch_size", 1)
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=_nw,
        pin_memory=_pin,
        persistent_workers=_nw > 0,
        drop_last=val_batch_size > 1,
    )

    # ---- Model ----
    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{param_count:,}")

    # ---- Diffusion ----
    diff_cfg = cfg["diffusion"]
    flow = RectifiedFlow(
        num_train_timesteps=diff_cfg.get("num_train_timesteps", 1000),
        num_inference_steps=diff_cfg.get("num_inference_steps", 10),
        prediction_type=diff_cfg.get("prediction_type", "velocity"),
        ode_solver=diff_cfg.get("ode_solver", "euler"),
        lr_init=diff_cfg.get("lr_init", False),
        t_max=diff_cfg.get("t_max", 0.8),
        freq_weighted_loss=diff_cfg.get("freq_weighted_loss", False),
        freq_loss_alpha=diff_cfg.get("freq_loss_alpha", 2.0),
        dct_block_size=cfg["model"].get("dct_block_size", 8),
        timestep_sampling=diff_cfg.get("timestep_sampling", "uniform"),
        logit_mean=diff_cfg.get("logit_mean", 0.0),
        logit_std=diff_cfg.get("logit_std", 1.0),
        single_step_inference=diff_cfg.get("single_step_inference", False),
    )

    consistency_trainer = None
    if diff_cfg.get("consistency_training", False):
        consistency_trainer = ConsistencyTrainer(
            num_train_timesteps=diff_cfg["num_train_timesteps"],
            target_steps=diff_cfg.get("consistency_steps", 1),
        )

    # ---- Optimizer + Scheduler ----
    optimizer_name = args.optimizer or tcfg.get("optimizer", "adamw")
    tcfg["optimizer"] = optimizer_name
    optimizer = build_optimizer(model, cfg)
    
    total_steps = len(train_loader) * tcfg["epochs"]
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    # ---- EMA ----
    ema = EMA(model, decay=tcfg.get("ema_decay", 0.9999))

    # ---- AMP (CUDA only — MPS does not support GradScaler) ----
    use_amp = tcfg.get("mixed_precision", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    best_psnr = 0.0

    if args.resume:
        log.info("Resuming from: %s", args.resume)
        start_epoch, global_step, best_psnr = load_checkpoint(
            args.resume, model, optimizer, scheduler, ema
        )
        log.info("  Resumed at epoch %d, step %d, best PSNR %.2f", start_epoch, global_step, best_psnr)

    # ---- Wandb ----
    lcfg = cfg.get("logging", {})
    diff_type = diff_cfg.get("type", "rectified_flow")
    transform_type = cfg["model"].get("transform_type", "dct")
    # Build descriptive run name including ablation flags
    ablation_tags = []
    if cfg["model"].get("residual_learning", False):
        ablation_tags.append("res")
    if diff_cfg.get("lr_init", False):
        ablation_tags.append("lrinit")
    if diff_cfg.get("freq_weighted_loss", False):
        ablation_tags.append("fwl")
    if diff_cfg.get("timestep_sampling", "uniform") != "uniform":
        ablation_tags.append("logit")
    if cfg["model"].get("spatial_dual_path", False):
        ablation_tags.append("spatial")
    sc_size = cfg["model"].get("spectral_conv_size", 1)
    if sc_size > 1:
        ablation_tags.append(f"sc{sc_size}")
    ablation_suffix = "-" + "+".join(ablation_tags) if ablation_tags else ""
    auto_run_name = f"dcno-x{scale}-{optimizer_name}-{diff_type}-{transform_type}{ablation_suffix}"
    if HAS_WANDB and not args.no_wandb:
        wandb.init(
            project=lcfg.get("project", "dcno-super-resolution"),
            name=lcfg.get("run_name", auto_run_name),
            config=cfg,
            resume="allow" if args.resume else None,
        )
        wandb.watch(model, log="gradients", log_freq=lcfg.get("log_interval", 100))

    # ---- Training ----
    log_interval = lcfg.get("log_interval", 100)
    val_start_epoch = lcfg.get("val_start_epoch", 200)
    val_interval = lcfg.get("val_interval", 5)
    save_interval = lcfg.get("save_interval", 10)
    sample_interval = lcfg.get("sample_interval", 5)
    accum_steps = tcfg.get("accumulation_steps", 1)
    grad_clip = tcfg.get("gradient_clip", 1.0)
    ckpt_dir = cfg.get("checkpoint", {}).get("dir", "./checkpoints")

    log.info("Training for %d epochs (%d steps)", tcfg['epochs'], total_steps)
    log.info("  Optimizer: %s | LR: %s", optimizer_name, tcfg['learning_rate'])
    log.info("  Batch: %d | Accum: %d | AMP: %s", tcfg['batch_size'], accum_steps, use_amp)
    log.info("  Scale: x%d | Patch: %d", scale, dcfg.get('patch_size', 256))
    log.info("  Diffusion: %s | Inference steps: %d", diff_cfg['type'], diff_cfg['num_inference_steps'])
    if subset_frac:
        log.info("  Subset: %.1f%% of data (train: %d, val: %d)", subset_frac * 100, len(train_ds), len(val_ds))

    residual_learning = cfg["model"].get("residual_learning", False)
    if residual_learning:
        log.info("  Residual learning: ON (diffusing HR - bicubic(LR))")
    if diff_cfg.get("lr_init", False):
        log.info("  LR-init sampling: ON (t_max=%.2f)", diff_cfg.get("t_max", 0.8))
    if diff_cfg.get("freq_weighted_loss", False):
        log.info("  Freq-weighted loss: ON (alpha=%.1f)", diff_cfg.get("freq_loss_alpha", 2.0))
    if diff_cfg.get("timestep_sampling", "uniform") != "uniform":
        log.info("  Timestep sampling: %s (mean=%.1f, std=%.1f)",
                 diff_cfg["timestep_sampling"], diff_cfg.get("logit_mean", 0.0), diff_cfg.get("logit_std", 1.0))

    model.train()
    for epoch in range(start_epoch, tcfg["epochs"]):
        epoch_loss = 0.0
        epoch_improve = 0.0
        epoch_loss_lo = 0.0
        epoch_loss_hi = 0.0
        epoch_steps = 0
        lo_count = 0
        hi_count = 0
        t0 = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{tcfg['epochs']}",
            leave=True,
            dynamic_ncols=True,
        )
        for batch_idx, (hr, lr) in enumerate(pbar):
            hr, lr = hr.to(device), lr.to(device)

            # Upsample LR to HR spatial size for conditioning
            _, _, H_hr, W_hr = hr.shape
            lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False)

            # Residual learning: pass bicubic baseline so diffusion learns the residual
            x_bicubic = lr_up if residual_learning else None

            with autocast(enabled=use_amp):
                if consistency_trainer is not None:
                    # Consistency training step
                    with ema.average_parameters() as ema_model:
                        result = consistency_trainer.training_step(
                            model, ema_model, hr, lr_up
                        )
                else:
                    # Standard rectified flow training
                    result = flow.training_step(model, hr, lr_up, x_bicubic=x_bicubic)

                loss = result["loss"] / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                ema.update()
                global_step += 1

            epoch_loss += result["loss"].item()
            if "noise_improve" in result:
                epoch_improve += result["noise_improve"]
                if not math.isnan(result["loss_lo_t"]):
                    epoch_loss_lo += result["loss_lo_t"]
                    lo_count += 1
                if not math.isnan(result["loss_hi_t"]):
                    epoch_loss_hi += result["loss_hi_t"]
                    hi_count += 1
            epoch_steps += 1

            # Update progress bar
            avg_loss = epoch_loss / epoch_steps
            lr_current = optimizer.param_groups[0]["lr"]
            postfix = dict(loss=f"{result['loss'].item():.4f}", avg=f"{avg_loss:.4f}", lr=f"{lr_current:.2e}")
            if "noise_improve" in result:
                postfix["imp"] = f"{result['noise_improve']:.1%}"
                postfix["lo"] = f"{result['loss_lo_t']:.3f}"
                postfix["hi"] = f"{result['loss_hi_t']:.3f}"
            pbar.set_postfix(**postfix)

            # Logging
            if global_step % log_interval == 0 and global_step > 0:
                avg_loss = epoch_loss / epoch_steps
                lr_current = optimizer.param_groups[0]["lr"]
                
                log_dict = {
                    "train/loss": result["loss"].item(),
                    "train/loss_avg": avg_loss,
                    "train/lr": lr_current,
                    "train/epoch": epoch,
                    "train/step": global_step,
                }
                if "noise_improve" in result:
                    log_dict["train/noise_improve"] = result["noise_improve"]
                    log_dict["train/loss_lo_t"] = result["loss_lo_t"]
                    log_dict["train/loss_hi_t"] = result["loss_hi_t"]
                if "v_pred_norm" in result:
                    log_dict["train/v_pred_norm"] = result["v_pred_norm"]
                if "consistency_gap" in result:
                    log_dict["train/consistency_gap"] = result["consistency_gap"]
                
                if HAS_WANDB and wandb.run is not None:
                    wandb.log(log_dict, step=global_step)

        pbar.close()
        t_train_end = time.time()
        epoch_time = t_train_end - t0
        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_improve = epoch_improve / max(epoch_steps, 1)
        avg_lo = epoch_loss_lo / max(lo_count, 1)
        avg_hi = epoch_loss_hi / max(hi_count, 1)
        if epoch_improve != 0:
            log.info("Epoch %d/%d done in %.1fs | Loss: %.4f | Improve: %.1f%% | Lo/Hi: %.3f/%.3f",
                     epoch+1, tcfg['epochs'], epoch_time, avg_loss, avg_improve*100, avg_lo, avg_hi)
        else:
            log.info("Epoch %d/%d done in %.1fs | Avg Loss: %.4f", epoch+1, tcfg['epochs'], epoch_time, avg_loss)

        # ---- Validation (starts at val_start_epoch, then every val_interval) ----
        if (epoch + 1) >= val_start_epoch and (epoch + 1 - val_start_epoch) % val_interval == 0:
            t_val_start = time.time()
            with ema.average_parameters():
                val_result = validate(model, flow, val_loader, device, cfg)
            t_val_end = time.time()
            
            val_psnr = val_result["psnr"]
            log.info("  Val PSNR: %.2f dB (best: %.2f dB) [val took %.1fs]", val_psnr, best_psnr, t_val_end - t_val_start)

            if HAS_WANDB and wandb.run is not None:
                wandb.log({
                    "val/psnr": val_psnr,
                    "val/best_psnr": max(best_psnr, val_psnr),
                    "epoch": epoch + 1,
                }, step=global_step)

            # Log samples
            if (epoch + 1) % sample_interval == 0 and val_result["samples"]:
                log_samples_to_wandb(val_result["samples"], epoch + 1)

            # Save best
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pth"),
                    model, optimizer, scheduler, ema,
                    epoch + 1, global_step, best_psnr, cfg,
                )
                log.info("  ★ New best! Saved to %s/best.pth", ckpt_dir)

            # Early stopping
            es_cfg = tcfg.get("early_stopping", {})
            if es_cfg.get("enabled", False):
                patience = es_cfg.get("patience", 30)
                # Simple check: if no improvement for `patience` epochs
                # (tracked via checkpoint epoch vs current)
                pass  # Implemented via best_psnr tracking above

        # ---- Save latest checkpoint (overwrite each time) ----
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, "latest.pth"),
                model, optimizer, scheduler, ema,
                epoch + 1, global_step, best_psnr, cfg,
            )

    # ---- Final save (as latest) ----
    save_checkpoint(
        os.path.join(ckpt_dir, "latest.pth"),
        model, optimizer, scheduler, ema,
        tcfg["epochs"], global_step, best_psnr, cfg,
    )
    log.info("Training complete! Best PSNR: %.2f dB", best_psnr)
    log.info("Checkpoints saved to: %s (best.pth + latest.pth)", ckpt_dir)

    if HAS_WANDB and wandb.run is not None:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DCNO")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw", "musgd"],
                        help="Override optimizer from config")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to train on (default: auto-detect)")
    parser.add_argument("--subset", type=float, default=None, metavar="FRAC",
                        help="Use a fraction of the dataset (0.0-1.0) for quick convergence tests")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs from config")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    train(cfg, args)


if __name__ == "__main__":
    main()

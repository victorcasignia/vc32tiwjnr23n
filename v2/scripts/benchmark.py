"""
Benchmark DWNO-SR checkpoints on classic SR datasets.

Usage (from v2/):
    python -m scripts.benchmark \
        --config configs/dwno_div2k_x2.yaml \
        --checkpoint checkpoints/div2k_x2/best_ema.pth \
        --bench-root ../data/benchmark

If --checkpoint is omitted, the script prefers:
  1) <ckpt_dir>/best_ema.pth
  2) <ckpt_dir>/best.pth

Outputs:
  - Per-dataset PSNR / SSIM
  - Overall average across all benchmark images
  - JSON summary in <ckpt_dir>/benchmark_x{scale}.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image
from torch.amp import autocast
from torch.profiler import ProfilerActivity, profile
from torchvision.transforms import functional as TF

# Ensure imports work when running as module from v2/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DWNOS
from scripts.utils import psnr, ssim


DATASETS = ["Set5", "Set14", "B100", "Urban100"]


def _load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img).unsqueeze(0).to(device)
    return t


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _estimate_gflops(
    model: DWNOS,
    sample_lr: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> Optional[float]:
    """Estimate forward-pass GFLOPs for one LR input shape via torch profiler.
    Returns None if FLOPs are unavailable for the current backend/build.
    """
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    try:
        _sync_if_cuda(device)
        with profile(activities=activities, with_flops=True) as prof:
            with autocast("cuda", enabled=amp and device.type == "cuda"):
                _ = model(sample_lr)
        _sync_if_cuda(device)

        total_flops = 0
        for event in prof.key_averages():
            event_flops = getattr(event, "flops", 0)
            if event_flops is not None:
                total_flops += int(event_flops)

        if total_flops <= 0:
            return None
        return float(total_flops) / 1e9
    except Exception:
        return None


def _size_stats(heights: List[int], widths: List[int]) -> Dict[str, float]:
    return {
        "min_h": int(min(heights)),
        "max_h": int(max(heights)),
        "avg_h": float(sum(heights) / len(heights)),
        "min_w": int(min(widths)),
        "max_w": int(max(widths)),
        "avg_w": float(sum(widths) / len(widths)),
    }


def _resolve_checkpoint(cfg: dict, ckpt_arg: str | None) -> Path:
    if ckpt_arg:
        ckpt = Path(ckpt_arg)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints"))
    for name in ("best_ema.pth", "best.pth", "latest.pth"):
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. Expected best_ema.pth or best.pth or latest.pth"
    )


def _load_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> DWNOS:
    model = DWNOS(
        scale=cfg["scale"],
        in_channels=cfg.get("in_channels", 3),
        channels=cfg.get("channels", 64),
        stage_depths=cfg.get("stage_depths", [4, 4, 4]),
        levels=cfg.get("levels", 2),
        kernel_size=cfg.get("kernel_size", 7),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        drop_path=cfg.get("drop_path", 0.05),
        filter_len=cfg.get("filter_len", 3),
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=True)
    else:
        model.load_state_dict(state, strict=True)

    model.eval()
    return model


def _match_lr_for_hr(hr_path: Path, lr_dir: Path, scale: int) -> Path:
    expected = lr_dir / f"{hr_path.stem}x{scale}{hr_path.suffix}"
    if expected.exists():
        return expected

    # Fallbacks for extension variations
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = lr_dir / f"{hr_path.stem}x{scale}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No LR pair found for HR image: {hr_path.name} in {lr_dir}")


@torch.no_grad()
def _eval_dataset(
    model: DWNOS,
    dataset_dir: Path,
    scale: int,
    device: torch.device,
    amp: bool,
) -> Dict[str, object]:
    hr_dir = dataset_dir / "HR"
    lr_dir = dataset_dir / "LR_bicubic" / f"X{scale}"

    if not hr_dir.exists() or not lr_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset layout missing under {dataset_dir} (need HR/ and LR_bicubic/X{scale}/)"
        )

    hr_paths = sorted(
        [p for p in hr_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    )
    if len(hr_paths) == 0:
        raise RuntimeError(f"No HR images found in {hr_dir}")

    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    infer_ms_vals: List[float] = []
    lr_heights: List[int] = []
    lr_widths: List[int] = []
    hr_heights: List[int] = []
    hr_widths: List[int] = []
    dataset_gflops: Optional[float] = None
    dataset_gflops_input_lr: Optional[Tuple[int, int]] = None

    for hr_path in hr_paths:
        lr_path = _match_lr_for_hr(hr_path, lr_dir, scale)

        lr = _load_rgb_tensor(lr_path, device)
        hr = _load_rgb_tensor(hr_path, device)

        lr_heights.append(int(lr.shape[-2]))
        lr_widths.append(int(lr.shape[-1]))
        hr_heights.append(int(hr.shape[-2]))
        hr_widths.append(int(hr.shape[-1]))

        if dataset_gflops is None:
            dataset_gflops = _estimate_gflops(model, lr, device, amp)
            dataset_gflops_input_lr = (int(lr.shape[-2]), int(lr.shape[-1]))

        _sync_if_cuda(device)
        start = time.perf_counter()
        with autocast("cuda", enabled=amp and device.type == "cuda"):
            sr = model(lr)
        _sync_if_cuda(device)
        infer_ms_vals.append((time.perf_counter() - start) * 1000.0)

        sr = sr.clamp(0, 1).float()
        hr = hr.float()

        # Safety crop in case of any 1px mismatch from odd sizes
        h = min(sr.shape[-2], hr.shape[-2])
        w = min(sr.shape[-1], hr.shape[-1])
        sr = sr[..., :h, :w]
        hr = hr[..., :h, :w]

        psnr_vals.append(psnr(sr, hr))
        ssim_vals.append(ssim(sr, hr))

    return {
        "num_images": len(hr_paths),
        "psnr": float(sum(psnr_vals) / len(psnr_vals)),
        "ssim": float(sum(ssim_vals) / len(ssim_vals)),
        "inference_ms": {
            "avg": float(sum(infer_ms_vals) / len(infer_ms_vals)),
            "min": float(min(infer_ms_vals)),
            "max": float(max(infer_ms_vals)),
        },
        "image_size": {
            "lr": _size_stats(lr_heights, lr_widths),
            "hr": _size_stats(hr_heights, hr_widths),
        },
        "gflops": {
            "value": dataset_gflops,
            "input_lr_h": dataset_gflops_input_lr[0] if dataset_gflops_input_lr else None,
            "input_lr_w": dataset_gflops_input_lr[1] if dataset_gflops_input_lr else None,
        },
    }


def _weighted_global_average(results: Dict[str, Dict[str, object]]) -> Tuple[float, float, int]:
    total_n = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    for stats in results.values():
        n = int(stats["num_images"])
        total_n += n
        psnr_sum += float(stats["psnr"]) * n
        ssim_sum += float(stats["ssim"]) * n

    if total_n == 0:
        return 0.0, 0.0, 0

    return psnr_sum / total_n, ssim_sum / total_n, total_n


def main():
    parser = argparse.ArgumentParser(description="Benchmark DWNO-SR on Set5/Set14/B100/Urban100")
    parser.add_argument("--config", required=True, help="Path to YAML config used for model definition")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (.pth)")
    parser.add_argument("--bench-root", default="../data/benchmark", help="Benchmark root folder")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, help="Datasets to evaluate")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--amp", action="store_true", help="Enable AMP during inference on CUDA")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint)

    print(f"[Benchmark] Device: {device}")
    print(f"[Benchmark] Config: {cfg_path}")
    print(f"[Benchmark] Checkpoint: {checkpoint_path}")

    model = _load_model(cfg, checkpoint_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Benchmark] Params: total={total_params:,} ({total_params / 1e6:.3f}M), "
        f"trainable={trainable_params:,} ({trainable_params / 1e6:.3f}M)"
    )
    scale = int(cfg["scale"])

    bench_root = Path(args.bench_root)
    if not bench_root.exists():
        raise FileNotFoundError(f"Benchmark root not found: {bench_root}")

    results: Dict[str, Dict[str, object]] = {}

    print("\nDataset results")
    print("-" * 90)
    print(f"{'Dataset':<10} {'Images':>8} {'PSNR(dB)':>10} {'SSIM':>10} {'Inf(ms)':>10} {'GFLOPs':>10} {'LR size (avg)':>18}")
    print("-" * 90)

    for name in args.datasets:
        ds_dir = bench_root / name
        stats = _eval_dataset(model, ds_dir, scale, device, args.amp)
        results[name] = stats
        lr_stats = stats["image_size"]["lr"]
        avg_lr_h = lr_stats["avg_h"]
        avg_lr_w = lr_stats["avg_w"]
        gflops_value = stats["gflops"]["value"]
        gflops_text = f"{gflops_value:.3f}" if gflops_value is not None else "n/a"
        print(
            f"{name:<10} {int(stats['num_images']):>8d} {float(stats['psnr']):>10.4f} {float(stats['ssim']):>10.6f} "
            f"{float(stats['inference_ms']['avg']):>10.3f} {gflops_text:>10} "
            f"{avg_lr_h:>8.1f}x{avg_lr_w:<8.1f}"
        )

    global_psnr, global_ssim, total_images = _weighted_global_average(results)
    all_infer_ms = [float(stats["inference_ms"]["avg"]) for stats in results.values()]
    avg_infer_ms = float(sum(all_infer_ms) / len(all_infer_ms)) if all_infer_ms else 0.0
    print("-" * 90)
    print(f"{'ALL(weighted)':<10} {total_images:>8d} {global_psnr:>10.4f} {global_ssim:>10.6f} {avg_infer_ms:>10.3f} {'-':>10} {'-':>18}")

    out = {
        "config": str(cfg_path),
        "checkpoint": str(checkpoint_path),
        "scale": scale,
        "device": str(device),
        "model_params": {
            "total": int(total_params),
            "trainable": int(trainable_params),
        },
        "datasets": results,
        "all_weighted": {
            "num_images": total_images,
            "psnr": global_psnr,
            "ssim": global_ssim,
            "inference_ms_dataset_avg": avg_infer_ms,
        },
    }

    ckpt_dir = checkpoint_path.parent
    out_path = ckpt_dir / f"benchmark_x{scale}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n[Benchmark] Saved summary to: {out_path}")


if __name__ == "__main__":
    main()

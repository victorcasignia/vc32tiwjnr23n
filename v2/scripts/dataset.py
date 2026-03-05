"""
DIV2K / DF2K Dataset for Single-Image Super-Resolution.

Supports:
  - DIV2K train/val splits
  - Random crop augmentation (training)
  - Bicycle degradation (LR images pre-generated)
  - On-the-fly LR generation via bicubic downsampling
  - Data caching to RAM for fast iteration
"""

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image → float32 tensor in [0, 1], shape (3, H, W)."""
    return TF.to_tensor(img)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Float32 tensor in [0, 1] → PIL image."""
    return TF.to_pil_image(t.clamp(0, 1))


# ---------------------------------------------------------------------------
# Paired SR Dataset
# ---------------------------------------------------------------------------

class SRDataset(Dataset):
    """
    Paired LR/HR super-resolution dataset.

    Parameters
    ----------
    hr_dir      : directory of high-resolution images
    lr_dir      : directory of low-resolution bicubic images (may be None)
    scale       : upscale factor
    patch_size  : HR patch size for training crops (None = full image)
    augment     : random flip + rotation augmentation
    cache       : load all images into RAM (fast but memory-heavy)
    pregenerate : if True and lr_dir is None, downscale HR bicubically on the fly
    """

    def __init__(
        self,
        hr_dir: str,
        lr_dir: Optional[str] = None,
        scale: int = 4,
        patch_size: Optional[int] = 256,
        augment: bool = True,
        cache: bool = False,
        pregenerate: bool = True,
    ):
        super().__init__()
        self.scale      = scale
        self.patch_size = patch_size
        self.augment    = augment
        self.cache      = cache
        self.pregenerate = pregenerate

        self.hr_paths: List[Path] = sorted(
            [p for p in Path(hr_dir).glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        if len(self.hr_paths) == 0:
            raise FileNotFoundError(f"No images found in {hr_dir}")

        self.lr_dir = Path(lr_dir) if lr_dir else None
        if self.lr_dir and not self.lr_dir.exists():
            self.lr_dir = None

        # Optional RAM cache
        self._hr_cache: dict = {}
        self._lr_cache: dict = {}
        if cache:
            print(f"[Dataset] Caching {len(self.hr_paths)} images to RAM …")
            for i, p in enumerate(self.hr_paths):
                self._hr_cache[i] = pil_to_tensor(load_image(str(p)))
                if self.lr_dir:
                    lp = self._lr_path(p)
                    if lp.exists():
                        self._lr_cache[i] = pil_to_tensor(load_image(str(lp)))

    def _lr_path(self, hr_path: Path) -> Path:
        """Infer LR path from HR path and lr_dir."""
        stem = hr_path.stem
        # DIV2K naming: 0001x4.png
        lr_name = f"{stem}x{self.scale}{hr_path.suffix}"
        return self.lr_dir / lr_name

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load HR
        if idx in self._hr_cache:
            hr = self._hr_cache[idx].clone()
        else:
            hr = pil_to_tensor(load_image(str(self.hr_paths[idx])))

        # Load or generate LR
        if idx in self._lr_cache:
            lr = self._lr_cache[idx].clone()
        elif self.lr_dir is not None:
            lp = self._lr_path(self.hr_paths[idx])
            if lp.exists():
                lr = pil_to_tensor(load_image(str(lp)))
            else:
                lr = self._bicubic_lr(hr)
        else:
            lr = self._bicubic_lr(hr)

        # Random crop
        if self.patch_size is not None:
            hr, lr = self._random_crop(hr, lr)

        # Augmentation
        if self.augment:
            hr, lr = self._augment(hr, lr)

        return lr, hr

    def _bicubic_lr(self, hr: torch.Tensor) -> torch.Tensor:
        """Generate LR from HR via bicubic downsampling."""
        C, H, W = hr.shape
        lr_h, lr_w = H // self.scale, W // self.scale
        return F.interpolate(
            hr.unsqueeze(0),
            size=(lr_h, lr_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp(0, 1)

    def _random_crop(
        self, hr: torch.Tensor, lr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, H, W = hr.shape
        ps = self.patch_size
        if H < ps or W < ps:
            # Pad if image is smaller than patch size
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            hr = F.pad(hr, (0, pad_w, 0, pad_h), mode="reflect")
            lr_ps = ps // self.scale
            lr_pad_h = max(0, lr_ps - lr.shape[1])
            lr_pad_w = max(0, lr_ps - lr.shape[2])
            lr = F.pad(lr, (0, lr_pad_w, 0, lr_pad_h), mode="reflect")
            _, H, W = hr.shape

        top  = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        hr_crop = hr[:, top:top + ps, left:left + ps]

        # Corresponding LR crop
        lr_top  = top  // self.scale
        lr_left = left // self.scale
        lr_ps   = ps   // self.scale
        lr_crop = lr[:, lr_top:lr_top + lr_ps, lr_left:lr_left + lr_ps]
        return hr_crop, lr_crop

    @staticmethod
    def _augment(
        hr: torch.Tensor, lr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random horizontal flip
        if random.random() > 0.5:
            hr = TF.hflip(hr)
            lr = TF.hflip(lr)
        # Random vertical flip
        if random.random() > 0.5:
            hr = TF.vflip(hr)
            lr = TF.vflip(lr)
        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            hr = torch.rot90(hr, k, dims=[1, 2])
            lr = torch.rot90(lr, k, dims=[1, 2])
        return hr, lr


# ---------------------------------------------------------------------------
# Validation Dataset (full images, no aug)
# ---------------------------------------------------------------------------

class SRValDataset(SRDataset):
    """Validation variant: full images, no augmentation, no crop."""

    def __init__(self, hr_dir: str, lr_dir: Optional[str] = None, scale: int = 4):
        super().__init__(
            hr_dir, lr_dir, scale,
            patch_size=None, augment=False, cache=False,
        )


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def make_train_loader(cfg: dict) -> DataLoader:
    ds = SRDataset(
        hr_dir     = cfg["train_hr_dir"],
        lr_dir     = cfg.get("train_lr_dir"),
        scale      = cfg["scale"],
        patch_size = cfg.get("patch_size", 256),
        augment    = cfg.get("augment", True),
        cache      = cfg.get("cache", False),
    )
    return DataLoader(
        ds,
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        num_workers = cfg.get("num_workers", 4),
        pin_memory  = True,
        drop_last   = True,
        persistent_workers = cfg.get("num_workers", 4) > 0,
    )


def make_val_loader(cfg: dict) -> DataLoader:
    ds = SRValDataset(
        hr_dir  = cfg["val_hr_dir"],
        lr_dir  = cfg.get("val_lr_dir"),
        scale   = cfg["scale"],
    )
    return DataLoader(
        ds,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )

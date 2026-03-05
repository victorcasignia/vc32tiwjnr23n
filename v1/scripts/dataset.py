"""
Dataset loaders for super-resolution training and evaluation.

Supports DIV2K, Flickr2K, DF2K, and benchmark test sets.
"""

import logging
import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

log = logging.getLogger(__name__)


def _find_images(directory: str, extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif")) -> List[str]:
    """Recursively find all image files in a directory."""
    images = []
    directory = Path(directory)
    if not directory.exists():
        return images
    for ext in extensions:
        images.extend(sorted(str(p) for p in directory.rglob(f"*{ext}")))
    return images


class SRPairedDataset(Dataset):
    """
    Paired super-resolution dataset (HR + LR directories).
    
    For training: random crop + augmentation.
    For validation/test: center crop or full image.
    """

    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        patch_size: int = 256,
        scale_factor: int = 4,
        augment: bool = True,
        training: bool = True,
    ):
        self.hr_files = _find_images(hr_dir)
        self.lr_files = _find_images(lr_dir)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment and training
        self.training = training

        assert len(self.hr_files) == len(self.lr_files), (
            f"HR ({len(self.hr_files)}) and LR ({len(self.lr_files)}) count mismatch.\n"
            f"HR dir: {hr_dir}\nLR dir: {lr_dir}"
        )
        assert len(self.hr_files) > 0, f"No images found in {hr_dir}"

        log.info("SRPairedDataset: %d images, scale x%d, patch %d, augment=%s",
                 len(self.hr_files), scale_factor, patch_size, augment and training)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        lr = Image.open(self.lr_files[idx]).convert("RGB")

        hr = self.to_tensor(hr)  # (3, H, W) in [0, 1]
        lr = self.to_tensor(lr)

        if self.training:
            hr, lr = self._random_crop(hr, lr)
            if self.augment:
                hr, lr = self._augment(hr, lr)
        else:
            hr, lr = self._center_crop(hr, lr)

        return hr, lr

    def _random_crop(self, hr: torch.Tensor, lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random crop at HR resolution, corresponding crop at LR."""
        _, h, w = hr.shape
        ps = self.patch_size
        s = self.scale_factor

        lr_ps = ps // s

        # Ensure we can crop
        _, lh, lw = lr.shape
        if lh < lr_ps or lw < lr_ps:
            # Resize if too small
            lr = torch.nn.functional.interpolate(
                lr.unsqueeze(0), size=(lr_ps, lr_ps), mode="bicubic", align_corners=False
            ).squeeze(0).clamp(0, 1)
            hr = torch.nn.functional.interpolate(
                hr.unsqueeze(0), size=(ps, ps), mode="bicubic", align_corners=False
            ).squeeze(0).clamp(0, 1)
            return hr, lr

        # Random position
        top_lr = random.randint(0, lh - lr_ps)
        left_lr = random.randint(0, lw - lr_ps)
        top_hr = top_lr * s
        left_hr = left_lr * s

        lr = lr[:, top_lr : top_lr + lr_ps, left_lr : left_lr + lr_ps]
        hr = hr[:, top_hr : top_hr + ps, left_hr : left_hr + ps]

        return hr, lr

    def _center_crop(self, hr: torch.Tensor, lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Center crop at HR resolution, corresponding crop at LR."""
        _, h, w = hr.shape
        ps = self.patch_size
        s = self.scale_factor
        lr_ps = ps // s
        _, lh, lw = lr.shape
        if lh < lr_ps or lw < lr_ps:
            lr = torch.nn.functional.interpolate(
                lr.unsqueeze(0), size=(lr_ps, lr_ps), mode="bicubic", align_corners=False
            ).squeeze(0).clamp(0, 1)
            hr = torch.nn.functional.interpolate(
                hr.unsqueeze(0), size=(ps, ps), mode="bicubic", align_corners=False
            ).squeeze(0).clamp(0, 1)
            return hr, lr
        top_lr = (lh - lr_ps) // 2
        left_lr = (lw - lr_ps) // 2
        top_hr = top_lr * s
        left_hr = left_lr * s
        lr = lr[:, top_lr : top_lr + lr_ps, left_lr : left_lr + lr_ps]
        hr = hr[:, top_hr : top_hr + ps, left_hr : left_hr + ps]
        return hr, lr

    def _augment(self, hr: torch.Tensor, lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random horizontal flip + 90° rotations."""
        # Horizontal flip
        if random.random() > 0.5:
            hr = torch.flip(hr, [-1])
            lr = torch.flip(lr, [-1])
        # Vertical flip
        if random.random() > 0.5:
            hr = torch.flip(hr, [-2])
            lr = torch.flip(lr, [-2])
        # 90° rotation
        if random.random() > 0.5:
            hr = torch.rot90(hr, 1, [-2, -1])
            lr = torch.rot90(lr, 1, [-2, -1])
        return hr, lr


class SRSyntheticDataset(Dataset):
    """
    Dataset that creates LR images on-the-fly from HR via bicubic downsampling.
    Useful when only HR images are available (e.g., Flickr2K).
    """

    def __init__(
        self,
        hr_dir: str,
        patch_size: int = 256,
        scale_factor: int = 4,
        augment: bool = True,
        training: bool = True,
    ):
        self.hr_files = _find_images(hr_dir)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment and training
        self.training = training

        assert len(self.hr_files) > 0, f"No images found in {hr_dir}"
        log.info("SRSyntheticDataset: %d images, scale x%d, patch %d, augment=%s",
                 len(self.hr_files), scale_factor, patch_size, augment and training)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        hr = self.to_tensor(hr)

        if self.training:
            hr = self._random_crop_hr(hr)
            if self.augment:
                hr = self._augment_single(hr)
        else:
            # Center crop for validation (enables batching)
            hr = self._center_crop_hr(hr)

        # Create LR by bicubic downsampling
        s = self.scale_factor
        _, h, w = hr.shape
        lr = torch.nn.functional.interpolate(
            hr.unsqueeze(0),
            size=(h // s, w // s),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0, 1)

        return hr, lr

    def _random_crop_hr(self, hr: torch.Tensor) -> torch.Tensor:
        _, h, w = hr.shape
        ps = self.patch_size
        if h < ps or w < ps:
            hr = torch.nn.functional.interpolate(
                hr.unsqueeze(0), size=(ps, ps), mode="bicubic", align_corners=False
            ).squeeze(0).clamp(0, 1)
            return hr
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return hr[:, top : top + ps, left : left + ps]

    def _center_crop_hr(self, hr: torch.Tensor) -> torch.Tensor:
        """Center crop HR for deterministic validation patches."""
        _, h, w = hr.shape
        ps = self.patch_size
        if h < ps or w < ps:
            hr = torch.nn.functional.interpolate(
                hr.unsqueeze(0), size=(ps, ps), mode="bicubic", align_corners=False
            ).squeeze(0).clamp(0, 1)
            return hr
        top = (h - ps) // 2
        left = (w - ps) // 2
        return hr[:, top : top + ps, left : left + ps]

    def _augment_single(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            x = torch.flip(x, [-1])
        if random.random() > 0.5:
            x = torch.flip(x, [-2])
        if random.random() > 0.5:
            x = torch.rot90(x, 1, [-2, -1])
        return x


def build_dataset(
    dataset_name: str,
    data_dir: str,
    scale_factor: int = 4,
    patch_size: int = 256,
    augment: bool = True,
    split: str = "train",
    subset: float = None,
) -> Dataset:
    """
    Build a dataset by name.
    
    Args:
        dataset_name: One of 'div2k', 'flickr2k', 'df2k', 'set5', 'set14', 'bsd100', 'urban100'
        data_dir:     Root data directory
        scale_factor: SR scale
        patch_size:   HR patch size for training crops
        augment:      Enable augmentation
        split:        'train' or 'val' or 'test'
        subset:       If set (0.0-1.0), use only this fraction of the dataset
    """
    training = split == "train"
    name = dataset_name.lower()
    ds = None

    if name == "div2k":
        if split == "train":
            hr_dir = os.path.join(data_dir, "DIV2K", "DIV2K_train_HR")
            lr_dir = os.path.join(data_dir, "DIV2K", f"DIV2K_train_LR_bicubic", f"X{scale_factor}")
        else:
            hr_dir = os.path.join(data_dir, "DIV2K", "DIV2K_valid_HR")
            lr_dir = os.path.join(data_dir, "DIV2K", f"DIV2K_valid_LR_bicubic", f"X{scale_factor}")

        if os.path.exists(hr_dir) and os.path.exists(lr_dir):
            ds = SRPairedDataset(hr_dir, lr_dir, patch_size, scale_factor, augment, training)
        elif os.path.exists(hr_dir):
            ds = SRSyntheticDataset(hr_dir, patch_size, scale_factor, augment, training)
        else:
            raise FileNotFoundError(f"DIV2K not found. Run: python -m scripts.download_data --datasets div2k")

    elif name == "flickr2k":
        hr_dir = os.path.join(data_dir, "Flickr2K", "Flickr2K_HR")
        ds = SRSyntheticDataset(hr_dir, patch_size, scale_factor, augment, training)

    elif name == "df2k":
        # Concatenate DIV2K + Flickr2K
        div2k = build_dataset("div2k", data_dir, scale_factor, patch_size, augment, split, subset=subset)
        if split == "train":
            flickr2k = build_dataset("flickr2k", data_dir, scale_factor, patch_size, augment, split, subset=subset)
            return torch.utils.data.ConcatDataset([div2k, flickr2k])  # subset already applied per-component
        return div2k

    elif name in ["set5", "set14", "bsd100", "urban100", "manga109"]:
        # Benchmark test sets
        name_map = {
            "set5": "Set5",
            "set14": "Set14",
            "bsd100": "B100",
            "urban100": "Urban100",
            "manga109": "Manga109",
        }
        bench_name = name_map[name]
        hr_dir = os.path.join(data_dir, "benchmark", bench_name, "HR")
        lr_dir = os.path.join(data_dir, "benchmark", bench_name, "LR_bicubic", f"X{scale_factor}")

        if os.path.exists(hr_dir) and os.path.exists(lr_dir):
            ds = SRPairedDataset(hr_dir, lr_dir, patch_size, scale_factor, False, False)
        elif os.path.exists(hr_dir):
            ds = SRSyntheticDataset(hr_dir, patch_size, scale_factor, False, False)
        else:
            raise FileNotFoundError(
                f"{bench_name} not found. Run: python -m scripts.download_data --datasets {name}"
            )

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return _maybe_subset(ds, subset, f"{name}/{split}")


def _maybe_subset(ds: Dataset, subset: float, name: str) -> Dataset:
    """Return a random subset of the dataset if subset is specified."""
    if subset is None or subset >= 1.0:
        return ds
    n = max(1, int(len(ds) * subset))
    indices = torch.randperm(len(ds))[:n].tolist()
    log.info("Subset %s: using %d / %d images (%.0f%%)", name, n, len(ds), subset * 100)
    return torch.utils.data.Subset(ds, indices)

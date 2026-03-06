"""
DWNO-SR: Deep Wavelet Neural Operator for Image Super-Resolution.

Full model architecture:

    LR input (B, 3, H, W)
          │
          ▼
    ShallowFeatureExtractor   →  feat_shallow  (B, C, H, W)
          │
          ▼
    DeepFeatureExtractor
     ├─ Stage 0: WNOStage (depth d0, levels L, window M)
     ├─ Stage 1: WNOStage (depth d1, levels L, window M)
     └─ Stage 2: WNOStage (depth d2, levels L, window M)
          │
          ▼
    Channel-mix conv  →  residual add feat_shallow  →  feat_deep  (B, C, H, W)
          │
          ▼
    PixelShuffleUpsample (×scale)
          │
          ▼
    Refinement conv (3×3)
          │
          ▼
    HR output (B, 3, H*scale, W*scale)

The long skip connection from shallow features to the upsample head
(like RRDB / SwinIR) ensures gradients reach the front of the network.

Wavelet-domain Loss (WaveletLoss)
-----------------------------------
In addition to pixel-domain L1, we compute a multi-scale loss in the
wavelet domain:

    L_wave = Σ_{j,d} w_{j,d} · || DWT_{j,d}[SR] - DWT_{j,d}[HR] ||_1

The weights w_{j,d} are set so that high-frequency subbands (LH, HL, HH)
are penalised more than LL, and finer levels (smaller j) are penalised more:

    w_{j,d} = α^{level-j} · β_d    where β_LL < β_LH = β_HL < β_HH

This encourages the model to faithfully reconstruct edges and textures,
which are represented by non-LL subbands.  The pixel-domain L1 alone tends
to produce over-smoothed outputs precisely because these HF coefficients have
small amplitude but contribute strongly to perceptual quality.

SSIM Loss component
--------------------
We also include a differentiable SSIM term:
    L = (1-λ_ssim) · L_l1 + λ_ssim · (1 - SSIM(SR, HR)) + λ_wave · L_wave

Total parameters: approximately 6-15M depending on config.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ShallowFeatureExtractor,
    WNOStage,
    PixelShuffleUpsample,
)
from .dwt import DWTForward2D, wavelet_orthogonality_loss


# ---------------------------------------------------------------------------
# Deep Feature Extractor (stacked WNO stages)
# ---------------------------------------------------------------------------

class DeepFeatureExtractor(nn.Module):
    """
    Stack of WNO stages with a final blending conv.
    """

    def __init__(
        self,
        channels: int,
        stage_depths: List[int],
        levels: int = 2,
        kernel_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        filter_len: int = 3,
    ):
        super().__init__()
        self.stages = nn.ModuleList([
            WNOStage(
                channels, depth, levels, kernel_size,
                mlp_ratio, drop_path, filter_len,
            )
            for depth in stage_depths
        ])
        self.blend = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return self.blend(x)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class DWNOS(nn.Module):
    """
    Deep Wavelet Neural Operator Super-Resolution (DWNO-SR).

    Parameters
    ----------
    scale         : upscale factor (2, 3, 4)
    in_channels   : RGB = 3
    channels      : internal feature channels (default 64)
    stage_depths  : list of block depths per stage (default [4,4,4])
    levels        : DWT levels per WNO block (default 2)
    kernel_size   : depthwise conv kernel size in SubbandConvBlock (default 7)
    mlp_ratio     : FFN hidden expansion factor (default 4.0)
    drop_path     : stochastic depth rate (default 0.05)
    filter_len    : learnable lifting filter length (default 3)
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        channels: int = 64,
        stage_depths: List[int] = None,
        levels: int = 2,
        kernel_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.05,
        filter_len: int = 3,
    ):
        super().__init__()
        if stage_depths is None:
            stage_depths = [4, 4, 4]

        self.scale = scale

        # ── Shallow features ─────────────────────────────────────────────
        self.shallow = ShallowFeatureExtractor(in_channels, channels)

        # ── Deep features ────────────────────────────────────────────────
        self.deep = DeepFeatureExtractor(
            channels, stage_depths, levels, kernel_size,
            mlp_ratio, drop_path, filter_len,
        )

        # ── Upsample head ────────────────────────────────────────────────
        self.upsample = PixelShuffleUpsample(channels, scale)

        # ── Output refinement ────────────────────────────────────────────
        self.refine_body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.refine_out = nn.Conv2d(channels, in_channels, 3, padding=1)

        # Init only normalization layers; all other modules keep their
        # per-module defaults (eye_ for SubbandOp, ICNR for upsample, etc.).
        self.apply(self._init_norms)

    @staticmethod
    def _init_norms(m: nn.Module):
        """Initialize only normalization layers; leave Conv2d/Linear to
        their per-module inits (eye_, ICNR, kaiming, etc.)."""
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : LR image  (B, 3, H, W)
        out : HR image  (B, 3, H*scale, W*scale)
        """
        # Bicubic upsampled LR as global skip (stabilizes luminance & dark details)
        lr_bicubic = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        
        feat_s  = self.shallow(x)           # (B, C, H, W)
        feat_d  = self.deep(feat_s)         # (B, C, H, W)
        feat    = feat_s + feat_d           # long skip
        feat_up = self.upsample(feat)       # (B, C, H*scale, W*scale)
        feat_refined = feat_up + self.refine_body(feat_up)
        residual = self.refine_out(feat_refined)  # (B, 3, H*scale, W*scale)
        return residual + lr_bicubic        # global image skip

    def orthogonality_loss(self) -> torch.Tensor:
        """Auxiliary loss to keep learned lifting filters near bi-orthogonal."""
        return wavelet_orthogonality_loss(self)


# ---------------------------------------------------------------------------
# Wavelet-Domain Loss
# ---------------------------------------------------------------------------

class WaveletLoss(nn.Module):
    """
    Multi-scale wavelet-domain L1 loss.

    Computes DWT of both SR and HR images, then penalises the coefficient
    differences at each (level, direction) with direction-specific weights.

    HH (diagonal) > LH/HL (edge) > LL (smooth), and finer levels weighted more.
    """

    def __init__(
        self,
        levels: int = 3,
        weight_ll: float = 0.1,
        weight_edge: float = 1.0,
        weight_diag: float = 2.0,   # Increased from 1.5 for stronger diagonal emphasis
        level_decay: float = 0.7,   # weight multiplier per coarser level
    ):
        super().__init__()
        self.levels      = levels
        self.w_ll    = weight_ll
        self.w_edge  = weight_edge
        self.w_diag  = weight_diag
        self.decay   = level_decay
        # Fixed Haar DWT for loss computation (not learnable)
        self.dwt = _HaarDWT2D()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=sr.device)
        x, y = sr, hr
        for lvl in range(self.levels):
            scale = self.decay ** lvl
            x_LL, x_LH, x_HL, x_HH = self.dwt(x)
            y_LL, y_LH, y_HL, y_HH = self.dwt(y)
            loss = loss + (
                self.w_ll   * F.l1_loss(x_LL, y_LL) * scale * 0.1 +
                self.w_edge * F.l1_loss(x_LH, y_LH) * scale +
                self.w_edge * F.l1_loss(x_HL, y_HL) * scale +
                self.w_diag * F.l1_loss(x_HH, y_HH) * scale
            )
            x, y = x_LL, y_LL   # recurse into approximation
        return loss


class _HaarDWT2D(nn.Module):
    """Fixed (non-learnable) Haar DWT used only for loss computation."""

    def __init__(self):
        super().__init__()
        # Haar analysis filters
        h_lo = torch.tensor([[1,  1]], dtype=torch.float32) / math.sqrt(2)
        h_hi = torch.tensor([[1, -1]], dtype=torch.float32) / math.sqrt(2)
        # 2D separable Haar filters as fixed conv kernels
        ll = (h_lo.T @ h_lo).unsqueeze(0).unsqueeze(0)  # (1,1,2,2)
        lh = (h_hi.T @ h_lo).unsqueeze(0).unsqueeze(0)
        hl = (h_lo.T @ h_hi).unsqueeze(0).unsqueeze(0)
        hh = (h_hi.T @ h_hi).unsqueeze(0).unsqueeze(0)
        self.register_buffer("ll", ll)
        self.register_buffer("lh", lh)
        self.register_buffer("hl", hl)
        self.register_buffer("hh", hh)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        # Apply per channel
        x_flat = x.reshape(B * C, 1, H, W)
        # Pad to even size
        if H % 2 != 0:
            x_flat = F.pad(x_flat, (0, 0, 0, 1), mode="reflect")
        if W % 2 != 0:
            x_flat = F.pad(x_flat, (0, 1, 0, 0), mode="reflect")

        def _apply(filt):
            return F.conv2d(x_flat, filt, stride=2, padding=0).reshape(B, C, -1, x_flat.shape[-1] // 2)

        return _apply(self.ll), _apply(self.lh), _apply(self.hl), _apply(self.hh)


# ---------------------------------------------------------------------------
# SSIM Loss (differentiable)
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss.
    L_ssim = 1 - SSIM(x, y)
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.ws = window_size
        kernel = self._gaussian_kernel(window_size, sigma)
        self.register_buffer("kernel", kernel)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        return kernel

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad    = self.ws // 2
        B, C, H, W = x.shape
        kernel = self.kernel.expand(C, 1, self.ws, self.ws)

        mu_x  = F.conv2d(x, kernel, padding=pad, groups=C)
        mu_y  = F.conv2d(y, kernel, padding=pad, groups=C)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sig_x2  = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x2
        sig_y2  = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y2
        sig_xy  = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

        numer = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        denom = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
        return (numer / denom).mean()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(x, y)


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class DWNOLoss(nn.Module):
    """
    Combined loss: L1 + λ_ssim·SSIM + λ_wave·WaveletLoss + λ_orth·Orth

    λ_wave penalises wavelet-domain errors (especially HF subbands).
    λ_orth regularises learnable lifting filters toward bi-orthogonal frames.
    """

    def __init__(
        self,
        lambda_ssim: float = 0.1,
        lambda_wave: float = 0.15,   # Increased from 0.05 for stronger HF supervision
        lambda_orth: float = 1e-4,
        wave_levels: int = 3,
        wave_weight_edge: float = 1.0,
        wave_weight_diag: float = 2.0,  # Exposed for config control
    ):
        super().__init__()
        self.lam_ssim = lambda_ssim
        self.lam_wave = lambda_wave
        self.lam_orth = lambda_orth
        self.ssim_fn  = SSIMLoss()
        self.wave_fn  = WaveletLoss(
            levels=wave_levels,
            weight_edge=wave_weight_edge,
            weight_diag=wave_weight_diag,
        )

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, dict]:
        l1   = F.l1_loss(sr, hr)
        ssim = self.ssim_fn(sr, hr)
        wave = self.wave_fn(sr, hr)
        orth = (
            wavelet_orthogonality_loss(model)
            if model is not None and self.lam_orth > 0
            else torch.tensor(0.0, device=sr.device)
        )

        total = l1 + self.lam_ssim * ssim + self.lam_wave * wave + self.lam_orth * orth

        return total, {
            "loss_total": total.item(),
            "loss_l1":    l1.item(),
            "loss_ssim":  ssim.item(),
            "loss_wave":  wave.item(),
            "loss_orth":  orth.item() if torch.is_tensor(orth) else 0.0,
        }

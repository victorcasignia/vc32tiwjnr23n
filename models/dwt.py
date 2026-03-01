"""Discrete Wavelet Transform (DWT) for super-resolution.

Implements a multi-level 2D Haar wavelet packet transform that matches
the interface of ``BlockDCT2d`` / ``BlockIDCT2d``, so the DCNO model
can switch between DCT and DWT transforms seamlessly.

In *wavelet packet* mode every subband is recursively decomposed
(not just LL), so after ``num_levels`` levels the channel / spatial
relationship is identical to blockwise DCT with
``block_size = 2^num_levels``::

    (B, C, H, W) → (B, C · 4^L, H / 2^L, W / 2^L)

Default ``block_size=8`` (``num_levels=3``):

    (B, 3, H, W) → (B, 192, H/8, W/8)   — same as BlockDCT2d(8)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["BlockDWT2d", "BlockIDWT2d", "DWTUpsample"]


# ---------------------------------------------------------------------------
# Primitives – single-level 2-D Haar DWT / IDWT
# ---------------------------------------------------------------------------

def _dwt2d_haar(x: torch.Tensor) -> torch.Tensor:
    """Single-level 2-D Haar wavelet packet transform.

    ``(B, C, H, W) → (B, 4C, H/2, W/2)``

    The four subbands per input channel are stacked as
    ``[LL, LH, HL, HH]`` along the channel axis.  Orthonormal scaling
    (multiply by 0.5 in 2-D) is used so that the transform preserves
    energy.
    """
    a = x[:, :, 0::2, 0::2]  # top-left
    b = x[:, :, 0::2, 1::2]  # top-right
    c = x[:, :, 1::2, 0::2]  # bottom-left
    d = x[:, :, 1::2, 1::2]  # bottom-right

    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5   # horizontal detail
    hl = (a + b - c - d) * 0.5   # vertical detail
    hh = (a - b - c + d) * 0.5   # diagonal detail

    return torch.cat([ll, lh, hl, hh], dim=1)


def _idwt2d_haar(x: torch.Tensor) -> torch.Tensor:
    """Single-level inverse 2-D Haar wavelet packet transform.

    ``(B, 4C, H_h, W_h) → (B, C, 2·H_h, 2·W_h)``
    """
    B, C4, Hh, Wh = x.shape
    C = C4 // 4

    ll = x[:, 0 * C : 1 * C]
    lh = x[:, 1 * C : 2 * C]
    hl = x[:, 2 * C : 3 * C]
    hh = x[:, 3 * C : 4 * C]

    a = (ll + lh + hl + hh) * 0.5   # top-left
    b = (ll - lh + hl - hh) * 0.5   # top-right
    c = (ll + lh - hl - hh) * 0.5   # bottom-left
    d = (ll - lh - hl + hh) * 0.5   # bottom-right

    out = x.new_zeros(B, C, 2 * Hh, 2 * Wh)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d

    return out


# ---------------------------------------------------------------------------
# BlockDWT2d / BlockIDWT2d  — multi-level wavelet packet
# ---------------------------------------------------------------------------

class BlockDWT2d(nn.Module):
    """Multi-level 2-D Haar wavelet packet transform.

    Applies ``num_levels`` successive single-level 2-D Haar DWT to
    **all** channels at each level (wavelet packet decomposition).
    This produces an output whose channel count and spatial reduction
    match ``BlockDCT2d`` with the same ``block_size``::

        block_size=8  →  num_levels=3
        (B, C, H, W) → (B, C·64, H/8, W/8)

    Args:
        block_size: Spatial reduction factor per dimension.
            Must be a power of two.  ``num_levels = log₂(block_size)``.
    """

    def __init__(self, block_size: int = 8):
        super().__init__()
        if block_size < 2 or (block_size & (block_size - 1)) != 0:
            raise ValueError(
                f"block_size must be a power of 2 ≥ 2, got {block_size}"
            )
        self.block_size = block_size
        self.num_levels: int = int(math.log2(block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_levels):
            x = _dwt2d_haar(x)
        return x

    def extra_repr(self) -> str:
        return f"block_size={self.block_size}, num_levels={self.num_levels}"


class BlockIDWT2d(nn.Module):
    """Multi-level inverse 2-D Haar wavelet packet transform.

    Reconstructs a pixel-domain image from wavelet packet coefficients::

        (B, out_channels·block_size², H_s, W_s)
        → (B, out_channels, H_s·block_size, W_s·block_size)

    Args:
        block_size: Must match the forward transform.
        out_channels: Number of output image channels (e.g. 3 for RGB).
    """

    def __init__(self, block_size: int = 8, out_channels: int = 3):
        super().__init__()
        if block_size < 2 or (block_size & (block_size - 1)) != 0:
            raise ValueError(
                f"block_size must be a power of 2 ≥ 2, got {block_size}"
            )
        self.block_size = block_size
        self.out_channels = out_channels
        self.num_levels: int = int(math.log2(block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_levels):
            x = _idwt2d_haar(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"block_size={self.block_size}, "
            f"out_channels={self.out_channels}, "
            f"num_levels={self.num_levels}"
        )


# ---------------------------------------------------------------------------
# DWTUpsample – spectral zero-padding analogue for wavelets
# ---------------------------------------------------------------------------

class DWTUpsample(nn.Module):
    """Upsample in wavelet packet domain by zero-padding detail subbands.

    Adds ``extra_levels = log₂(scale_factor)`` new wavelet levels where
    detail subbands (LH, HL, HH) are zero, analogous to ``DCTUpsample``
    which zero-pads high-frequency DCT coefficients.

    Args:
        scale_factor: Integer upsampling factor (must be power of 2).
        block_size: Current wavelet block size.
        channels: Image channels (unused, kept for API parity).
    """

    def __init__(
        self, scale_factor: int, block_size: int = 8, channels: int = 3
    ):
        super().__init__()
        if scale_factor < 2 or (scale_factor & (scale_factor - 1)) != 0:
            raise ValueError(
                f"scale_factor must be a power of 2 ≥ 2, got {scale_factor}"
            )
        self.scale_factor = scale_factor
        self.block_size = block_size
        self.channels = channels
        self.extra_levels: int = int(math.log2(scale_factor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand by inserting zero-detail levels.

        Each extra level: C_in channels → 4·C_in channels where the
        first C_in slots keep the values (LL) and the remaining
        3·C_in slots are zeros (LH, HL, HH).  Energy is scaled by 2
        per level (= ``scale_factor`` total) to match the orthonormal
        Haar convention.
        """
        B = x.shape[0]
        Hb, Wb = x.shape[2], x.shape[3]
        for _ in range(self.extra_levels):
            C_in = x.shape[1]
            out = x.new_zeros(B, 4 * C_in, Hb, Wb)
            out[:, :C_in] = x * 2.0  # energy scaling per Haar level
            x = out
        return x

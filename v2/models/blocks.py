"""
Wavelet Neural Operator (WNO) Blocks for Super-Resolution.

Mathematical Formulation
------------------------
A single WNO layer computes:

    v(x) = σ( W_local u(x)  +  IDWT( Σ_{j,d} R_{j,d} · DWT_{j,d}[u](x) ) )

where:
  u        : input feature map  ∈ R^{B × C × H × W}
  DWT_{j,d}: wavelet transform at level j, subband d ∈ {LL,LH,HL,HH}
  R_{j,d}  : per-(level,direction) channel-mixing operator
             → implemented as SubbandOperator (dw3×3 + pw1×1)
  W_local  : local spatial mixing within each subband's spatial grid
             → implemented as SubbandConvBlock (ConvNeXt-V2, dw7×7 + MLP)
  IDWT     : inverse wavelet transform (learnable lifting, perfect reconstruction)

Why ConvNeXt-V2 instead of window attention
--------------------------------------------
Wavelet subbands are already spatially downsampled by 2^j.  A 7×7 depthwise
conv on a level-1 subband covers the same receptive field as attending over
an ~56×56 region in the original image — sufficient for local mixing without
materialising any attention matrix.

Complexity per WNO block:
  Previous (window MHSA): O( HW/4 · M²·C + 3·HW/4·M²·C ) per level, ×heads
  Now (LK-DWConv + MLP) : O( 49·C·HW/4 ) per subband — ~5-10× cheaper

References
----------
- Li et al. (2021) "Fourier Neural Operator"  arXiv:2010.08895
- Tripura & Chakraborty (2022) "Wavelet Neural Operator"  arXiv:2205.02191
- Liu et al. (2022) "ConvNeXt V2"  arXiv:2301.00808
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dwt import DWTForward2D
from .attention import SubbandConvBlock


# ---------------------------------------------------------------------------
# Per-Subband Integral Operator  R_{j,d}
# ---------------------------------------------------------------------------

class SubbandOperator(nn.Module):
    """
    Learned integral operator kernel R_{j,d} applied to one wavelet subband.

    Performs depthwise separable convolution:
      dw3×3 (spatial correlation within subband) +
      pw1×1 (channel mixing / operator weight matrix)

    Both steps together approximate a low-rank factorisation of
    the full C×C×3×3 kernel while sharing parameters across spatial
    positions (translation equivariance).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=True)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act  = nn.GELU()
        nn.init.eye_(self.pw.weight.view(channels, channels))
        nn.init.zeros_(self.pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x)) + x))


# ---------------------------------------------------------------------------
# WNO Block
# ---------------------------------------------------------------------------

class WNOBlock(nn.Module):
    """
    Wavelet Neural Operator Block.

    Dataflow:
      x ──┬──────────────────────────────────────────────────┐
          │  DWT (J levels)                                   │ residual
          │   LL  → SubbandOp → SubbandConvBlock             │
          │   LH_j → SubbandOp → SubbandConvBlock (per lvl)  │
          │   HL_j → SubbandOp → SubbandConvBlock (per lvl)  │
          │   HH_j → SubbandOp → SubbandConvBlock (per lvl)  │
          │  IDWT ────────────────────────────────────────────┤
          └──────────────────────────────────────────────────▶┴ GroupNorm → out

    LH/HL share a SubbandConvBlock per level, while HH gets its own block
    (diagonal statistics differ from horizontal/vertical edges). LL gets its
    own block (distinct coarse-scale statistics).
    """

    def __init__(
        self,
        channels: int,
        levels: int = 2,
        kernel_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        filter_len: int = 3,
    ):
        super().__init__()
        self.levels = levels

        self.dwt = DWTForward2D(channels, levels, filter_len)

        # Integral operators R_{j,d} — separate per subband type and level
        self.ll_op  = SubbandOperator(channels)
        self.lh_ops = nn.ModuleList([SubbandOperator(channels) for _ in range(levels)])
        self.hl_ops = nn.ModuleList([SubbandOperator(channels) for _ in range(levels)])
        self.hh_ops = nn.ModuleList([SubbandOperator(channels) for _ in range(levels)])

        # Local mixing blocks W_local — LL gets its own.
        # LH/HL share a block per level; HH gets a dedicated block per level.
        self.ll_block  = SubbandConvBlock(channels, kernel_size, mlp_ratio, drop_path)
        self.hf_blocks = nn.ModuleList([
            SubbandConvBlock(channels, kernel_size, mlp_ratio, drop_path)
            for _ in range(levels)
        ])
        self.hh_blocks = nn.ModuleList([
            SubbandConvBlock(channels, kernel_size, mlp_ratio, drop_path)
            for _ in range(levels)
        ])

        self.out_norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # ── DWT ──────────────────────────────────────────────────────────
        LL, highs = self.dwt(x)   # highs[i] = (LH, HL, HH) finest-first

        # ── Process LL ───────────────────────────────────────────────────
        LL = self.ll_block(self.ll_op(LL))

        # ── Process HF subbands ──────────────────────────────────────────
        new_highs = []
        for i, (LH, HL, HH) in enumerate(highs):
            blk = self.hf_blocks[i]
            hh_blk = self.hh_blocks[i]
            LH  = blk(self.lh_ops[i](LH))
            HL  = blk(self.hl_ops[i](HL))
            HH  = hh_blk(self.hh_ops[i](HH))
            new_highs.append((LH, HL, HH))

        # ── IDWT + residual + norm ────────────────────────────────────────
        out = self.dwt.inverse(LL, new_highs)
        return self.out_norm(residual + out)


# ---------------------------------------------------------------------------
# WNO Stage
# ---------------------------------------------------------------------------

class WNOStage(nn.Module):
    """Stack of WNO blocks."""

    def __init__(
        self,
        channels: int,
        depth: int = 4,
        levels: int = 2,
        kernel_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        filter_len: int = 3,
        stage_res_init: float = 0.1,
    ):
        super().__init__()
        # Linearly increasing drop_path rate across blocks in stage
        dp_rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            WNOBlock(channels, levels, kernel_size, mlp_ratio, dp_rates[i], filter_len)
            for i in range(depth)
        ])
        self.stage_scale = nn.Parameter(torch.tensor(float(stage_res_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stage_in = x
        for blk in self.blocks:
            x = blk(x)
        stage_delta = x - stage_in
        return stage_in + self.stage_scale * stage_delta


# ---------------------------------------------------------------------------
# Pixel-Shuffle Upsampler (ICNR init)
# ---------------------------------------------------------------------------

class PixelShuffleUpsample(nn.Module):
    """
    Sub-pixel convolution with ICNR initialisation to avoid checkerboard artefacts.
    ICNR init fills each sub-pixel group with the same value → equivalent to
    nearest-neighbour upsampling at initialisation (smooth baseline).
    """

    def __init__(self, channels: int, scale_factor: int):
        super().__init__()
        self.scale = scale_factor
        out_ch = channels * (scale_factor ** 2)
        self.conv = nn.Conv2d(channels, out_ch, 3, padding=1)
        self._icnr_init()
        self.ps   = nn.PixelShuffle(scale_factor)

    def _icnr_init(self):
        C_in = self.conv.weight.shape[1]
        C_out = self.conv.weight.shape[0]
        s = self.scale
        small_w = nn.init.kaiming_normal_(
            torch.empty(C_out // (s * s), C_in, *self.conv.weight.shape[2:])
        )
        self.conv.weight.data.copy_(small_w.repeat_interleave(s * s, dim=0))
        # Zero bias for true nearest-neighbor-like behavior at init
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


# ---------------------------------------------------------------------------
# Shallow Feature Extractor
# ---------------------------------------------------------------------------

class ShallowFeatureExtractor(nn.Module):
    """3×3 conv → GELU → 3×3 conv, lifts input channels to model dim."""

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm  = nn.GroupNorm(min(8, dim), dim)
        self.act   = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.act(self.norm(self.conv1(x)))
        return feat + self.conv2(feat)

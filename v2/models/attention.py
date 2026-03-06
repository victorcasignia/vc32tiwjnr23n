"""
Efficient per-subband local mixing block for Wavelet Neural Operators.

Replaces window self-attention with a ConvNeXt-V2 style block:

    x_out = x + γ · FFN( LayerNorm( DWConv(x) ) )

where DWConv is a large-kernel (7×7) depthwise convolution.

Why this is appropriate for WNO
--------------------------------
In the WNO formulation:

    v(x) = σ( W_local u(x)  +  IDWT( Σ_{j,d} R_{j,d} · DWT_{j,d}[u] ) )

The term R_{j,d} (integral operator) is already handled by SubbandOperator
(depthwise 3×3 + pointwise 1×1).  W_local only needs to capture *local*
spatial structure within each subband's spatial grid, which is already
downsampled by factor 2^j — so a 7×7 depthwise conv at level j covers the
same receptive field as a ~56×56 spatial window in the original image.

Complexity comparison
---------------------
Window MHSA (M=8, H'×W' subband tokens):
  O(H'W' · M² · C/h)  — materialises (M²)×(M²) attention matrices

LK-DWConv + MLP:
  O(k² · C · H'W')  — no N² term, no QKV materialisation, no masking

Memory: at C=64, k=7, H'=W'=32 (level-1 subband from 64×64 LR):
  Window attn : ~3× more activation memory than LK-DWConv for the same RF

GRN (Global Response Normalisation, ConvNeXt-V2)
-------------------------------------------------
After the pointwise expansion we add:
    GRN(x) = x · (‖x‖₂ / mean(‖x‖₂)) · γ_grn + β_grn
This prevents feature collapse in pure-conv architectures (analogous to
batch normalisation but operates in channel-feature space).

References
----------
- Liu et al. (2022) "ConvNeXt V2"  arXiv:2301.00808
- Liu et al. (2021) "ConvNeXt V1"  arXiv:2201.03545
- Ding et al. (2022) "Scaling Up Your Kernels to 31×31"  arXiv:2203.06717

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Global Response Normalization (ConvNeXt-V2)
# ---------------------------------------------------------------------------

class GRN(nn.Module):
    """
    Global Response Normalization.
    Applied in channel-last layout (B, H, W, C).

    GRN(x) = x · ( ‖x‖₂ / E[‖x‖₂] ) · γ + β + x
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)       # (B,1,1,C)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)        # normalised
        return x * nx * self.gamma + self.beta + x


# ---------------------------------------------------------------------------
# Stochastic Depth (Drop Path)
# ---------------------------------------------------------------------------

class StochasticDepth(nn.Module):
    """Randomly drops the entire residual branch during training."""

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep  = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.bernoulli(torch.full(shape, keep, dtype=x.dtype, device=x.device))
        return x * mask / keep


# ---------------------------------------------------------------------------
# Subband Convolution Block (ConvNeXt-V2 style)
# ---------------------------------------------------------------------------

class SubbandConvBlock(nn.Module):
    """
    Large-kernel depthwise + inverted-bottleneck MLP block for wavelet subbands.

    Dataflow (channel-first input):
        x  ──┐
             ▼
        DWConv(kernel × kernel, groups=C)     ← local spatial mixing
             ↓  permute to channel-last
        LayerNorm
             ↓
        Linear(C → mlp_ratio·C) → GELU → GRN  ← inverted bottleneck
             ↓
        Linear(mlp_ratio·C → C)
             ↓
        × γ (layer-scale, init=1e-6)
             ↓  permute to channel-first
        + x  (residual)

    Parameters
    ----------
    dim         : number of channels
    kernel_size : depthwise conv kernel (7 by default)
    mlp_ratio   : hidden dim expansion (4 by default)
    drop_path   : stochastic depth rate
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        pad    = kernel_size // 2

        self.dwconv    = nn.Conv2d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.norm      = nn.LayerNorm(dim, eps=1e-6)
        self.fc1       = nn.Linear(dim, hidden)
        self.act       = nn.GELU()
        self.grn       = GRN(hidden)
        self.fc2       = nn.Linear(hidden, dim)
        self.gamma     = nn.Parameter(torch.ones(dim) * 1e-6)  # layer-scale init near zero
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        shortcut = x
        x = self.dwconv(x)                   # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)           # (B, H, W, C)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = self.gamma * x                   # layer-scale
        x = x.permute(0, 3, 1, 2)           # (B, C, H, W)
        return shortcut + self.drop_path(x)



# ---------------------------------------------------------------------------

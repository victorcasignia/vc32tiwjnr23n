"""
DCT Neural Operator (DCNO) — U-Net backbone for diffusion-based super-resolution.

All core operator layers work on blockwise DCT coefficients. The architecture
follows a U-Net with encoder → bottleneck → decoder + skip connections.
Time conditioning is injected via Adaptive Group Normalization (AdaGN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple

from .dct import BlockDCT2d, BlockIDCT2d
from .dwt import BlockDWT2d, BlockIDWT2d


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) timestep values in [0, 1]
        Returns:
            (B, dim) embedding
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepMLP(nn.Module):
    """Projects sinusoidal time embedding to per-layer scale/shift."""

    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, out_dim * 4),
            nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim * 2),  # scale + shift
        )

    def forward(self, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(t_emb)  # (B, 2 * out_dim)
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift


class AdaptiveGroupNorm(nn.Module):
    """Group Normalization with adaptive scale/shift from time embedding."""

    def __init__(self, num_channels: int, num_groups: int = 32):
        super().__init__()
        self.gn = nn.GroupNorm(min(num_groups, num_channels), num_channels)

    def forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        x = self.gn(x)
        # scale, shift: (B, C) → (B, C, 1, 1)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x


class DCTSpectralConv(nn.Module):
    """
    Spectral convolution in DCT coefficient space.
    
    Learns per-mode weights that rebalance the importance of each DCT
    frequency, then applies a pointwise 1×1 convolution for channel mixing.
    This is the DCT analog of Fourier Neural Operator's spectral convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, mode_weighting: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Pointwise convolution for channel mixing
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

        # Adaptive mode weighting: learnable per-channel importance
        if mode_weighting:
            self.mode_weight = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        else:
            self.mode_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode_weight is not None:
            # Soft gating per channel (each channel = one DCT mode × original channel)
            x = x * torch.sigmoid(self.mode_weight)
        x = self.conv(x)
        return x


class ChannelMLP(nn.Module):
    """Two-layer MLP applied pointwise across channels."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * expansion, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * expansion, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DCNO Block — single operator layer
# ---------------------------------------------------------------------------

class DCNOBlock(nn.Module):
    """
    One DCT Neural Operator block:
      AdaGN → DCTSpectralConv → SiLU → AdaGN → ChannelMLP → residual
    """

    def __init__(
        self,
        dim: int,
        time_dim: int,
        num_heads: int = 8,
        mode_weighting: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Time projection
        self.time_mlp = TimestepMLP(time_dim, dim)

        # Spectral path
        self.norm1 = AdaptiveGroupNorm(dim)
        self.spectral_conv = DCTSpectralConv(dim, dim, mode_weighting)
        self.act = nn.SiLU()

        # Channel MLP path
        self.norm2 = AdaptiveGroupNorm(dim)
        self.channel_mlp = ChannelMLP(dim, expansion=4, dropout=dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_mlp(t_emb)

        # Spectral convolution
        h = self.norm1(x, scale, shift)
        h = self.spectral_conv(h)
        h = self.act(h)
        x = x + h

        # Channel MLP 
        h = self.norm2(x, scale, shift)
        h = self.channel_mlp(h)
        x = x + h

        return x


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or dim
        self.conv = nn.Conv2d(dim, out_dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or dim
        self.conv = nn.Conv2d(dim, out_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Conditioning encoder — encodes LR DCT coefficients
# ---------------------------------------------------------------------------

class ConditionEncoder(nn.Module):
    """
    Encode the LR input (in DCT space) into multi-scale condition features
    that are injected into the U-Net decoder via addition.
    """

    def __init__(self, in_channels: int, hidden_dims: List[int]):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)

        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.GroupNorm(min(32, hidden_dims[i]), hidden_dims[i]),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], 3, stride=2, padding=1),
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.stem(x)
        feats.append(x)
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats  # [finest, ..., coarsest]


# ---------------------------------------------------------------------------
# DCNO U-Net
# ---------------------------------------------------------------------------

class DCNO(nn.Module):
    """
    DCT/DWT Neural Operator — full U-Net for diffusion-based super-resolution.
    
    Operates entirely in spectral coefficient space:
      1. Forward transform (DCT or DWT) converts input to coefficients
      2. U-Net of DCNOBlocks processes coefficients with time conditioning
      3. LR conditioning via ConditionEncoder (added to decoder features)
      4. Inverse transform reconstructs pixel-domain output
    
    Args:
        in_channels:   Input image channels (3 for RGB)
        out_channels:  Output image channels
        hidden_dims:   Channel dimensions for each U-Net stage
        depths:        Number of DCNOBlocks per stage
        dct_block_size: Block size / spatial reduction (8 for JPEG-standard DCT,
                        must be power-of-2 for DWT)
        num_heads:     Number of heads (used in future attention extension)
        mode_weighting: Enable adaptive spectral mode rebalancing
        scale_factor:  Super-resolution scale (2, 3, 4, etc.)
        dropout:       Dropout rate
        transform_type: Spectral transform — ``"dct"`` (default) or ``"dwt"``
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dims: List[int] = [64, 128, 256, 512],
        depths: List[int] = [2, 2, 4, 2],
        dct_block_size: int = 8,
        num_heads: int = 8,
        mode_weighting: bool = True,
        scale_factor: int = 4,
        dropout: float = 0.0,
        transform_type: str = "dct",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.dct_block_size = dct_block_size
        self.transform_type = transform_type

        n = dct_block_size
        num_down = len(hidden_dims) - 1  # number of spatial downsamples
        # Minimum divisor: block_size * 2^num_down ensures clean down/up
        self._min_div = n * (2 ** num_down)
        spec_channels = in_channels * n * n  # 3 * 64 = 192 for block_size=8

        # Spectral transforms (DCT or DWT — same shapes)
        if transform_type == "dct":
            self.fwd_transform = BlockDCT2d(n)
            self.inv_transform = BlockIDCT2d(n, out_channels)
        elif transform_type == "dwt":
            self.fwd_transform = BlockDWT2d(n)
            self.inv_transform = BlockIDWT2d(n, out_channels)
        else:
            raise ValueError(
                f"Unknown transform_type '{transform_type}'. "
                "Choose 'dct' or 'dwt'."
            )

        # Time embedding
        time_dim = hidden_dims[0] * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(hidden_dims[0]),
            nn.Linear(hidden_dims[0], time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Stem: spectral coefficients → hidden
        self.stem = nn.Conv2d(spec_channels, hidden_dims[0], 3, padding=1)

        # LR condition encoder (operates on LR spectral coefficients)
        self.cond_encoder = ConditionEncoder(spec_channels, hidden_dims)

        # ---- Encoder ----
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(hidden_dims, depths)):
            stage_blocks = nn.ModuleList()
            for _ in range(depth):
                stage_blocks.append(
                    DCNOBlock(dim, time_dim, num_heads, mode_weighting, dropout)
                )
            self.encoder_blocks.append(stage_blocks)
            
            if i < len(hidden_dims) - 1:
                self.downsamples.append(Downsample(dim, hidden_dims[i + 1]))

        # ---- Bottleneck ----
        self.bottleneck = nn.ModuleList([
            DCNOBlock(hidden_dims[-1], time_dim, num_heads, mode_weighting, dropout),
            DCNOBlock(hidden_dims[-1], time_dim, num_heads, mode_weighting, dropout),
        ])

        # ---- Decoder ----
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        rev_dims = list(reversed(hidden_dims))
        rev_depths = list(reversed(depths))

        for i, (dim, depth) in enumerate(zip(rev_dims, rev_depths)):
            # Skip connection projection (encoder_dim + decoder_dim → decoder_dim)
            if i > 0:
                self.skip_convs.append(nn.Conv2d(dim * 2, dim, 1))
            else:
                self.skip_convs.append(nn.Conv2d(dim * 2, dim, 1))

            stage_blocks = nn.ModuleList()
            for _ in range(depth):
                stage_blocks.append(
                    DCNOBlock(dim, time_dim, num_heads, mode_weighting, dropout)
                )
            self.decoder_blocks.append(stage_blocks)

            if i < len(rev_dims) - 1:
                self.upsamples.append(Upsample(dim, rev_dims[i + 1]))

        # Head: hidden → DCT coefficients (same spatial resolution as input)
        self.head = nn.Sequential(
            nn.GroupNorm(min(32, hidden_dims[0]), hidden_dims[0]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[0], spec_channels, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_lr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:  Noisy HR image at time t, (B, C, H, W)  — pixel domain
            t:    Diffusion timestep, (B,) in [0, 1]
            x_lr: Low-resolution conditioning image, (B, C, H_lr, W_lr)
        
        Returns:
            Predicted velocity (for rectified flow) in pixel domain, (B, C, H, W)
        """
        n = self.dct_block_size
        B = x_t.shape[0]

        # --- Pad inputs to be divisible by min_div (block_size × 2^num_down) ---
        x_t, pad_h, pad_w = self._pad(x_t, self._min_div)
        x_lr_padded, _, _ = self._pad(x_lr, self._min_div)

        # --- Transform to spectral space (DCT or DWT) ---
        x_spec = self.fwd_transform(x_t)           # (B, C*n^2, H/n, W/n)
        lr_spec = self.fwd_transform(x_lr_padded)  # (B, C*n^2, H_lr/n, W_lr/n)

        # --- Time embedding ---
        t_emb = self.time_embed(t)      # (B, time_dim)

        # --- LR condition features ---
        cond_feats = self.cond_encoder(lr_spec)  # multi-scale list

        # --- Stem ---
        h = self.stem(x_spec)            # (B, hidden_dims[0], H/n, W/n)

        # --- Encoder ---
        skip_connections = []
        for i, stage_blocks in enumerate(self.encoder_blocks):
            # Add LR conditioning at matching scale
            if i < len(cond_feats):
                cond = cond_feats[i]
                # Resize cond to match h if needed
                if cond.shape[-2:] != h.shape[-2:]:
                    cond = F.interpolate(cond, size=h.shape[-2:], mode="bilinear", align_corners=False)
                h = h + cond

            for block in stage_blocks:
                h = block(h, t_emb)
            skip_connections.append(h)

            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        # --- Bottleneck ---
        for block in self.bottleneck:
            h = block(h, t_emb)

        # --- Decoder ---
        for i, stage_blocks in enumerate(self.decoder_blocks):
            if i > 0:
                h = self.upsamples[i - 1](h)

            # Skip connection
            skip = skip_connections[-(i + 1)]
            if skip.shape[-2:] != h.shape[-2:]:
                skip = F.interpolate(skip, size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = self.skip_convs[i](h)

            for block in stage_blocks:
                h = block(h, t_emb)

        # --- Head: back to spectral coefficients ---
        h = self.head(h)               # (B, C*n^2, H/n, W/n)

        # --- Inverse transform back to pixel domain ---
        out = self.inv_transform(h)    # (B, C, H, W)

        # --- Remove padding ---
        out = self._unpad(out, pad_h, pad_w)

        return out

    def _pad(self, x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int, int]:
        """Pad spatial dims to be divisible by block_size."""
        _, _, H, W = x.shape
        pad_h = (block_size - H % block_size) % block_size
        pad_w = (block_size - W % block_size) % block_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def _unpad(self, x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        """Remove padding."""
        if pad_h > 0:
            x = x[:, :, :-pad_h, :]
        if pad_w > 0:
            x = x[:, :, :, :-pad_w]
        return x

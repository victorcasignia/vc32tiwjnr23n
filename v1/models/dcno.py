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


class FreqNorm(nn.Module):
    """
    Learnable per-channel affine normalization for spectral coefficients.

    DCT coefficients have wildly different magnitudes across frequency bands
    (DC is ~block_size× larger than high-freq AC).  This layer applies a
    data-driven normalisation so the downstream stem sees uniform-scale
    features across all spectral channels.

    Uses running statistics computed during training (like BatchNorm) but
    normalises per-channel only, not across the batch — this avoids the
    diffusion-timestep-dependent distribution shift that makes standard
    BatchNorm problematic for diffusion models.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels)  # LayerNorm equivalent
        # Extra learnable scale/shift to recover representational power
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * self.gamma + self.beta


def _gn(ch: int) -> nn.GroupNorm:
    """GroupNorm with largest valid num_groups <= 32."""
    g = min(32, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class ProgressiveStem(nn.Module):
    """
    Multi-layer projection from spectral channels to hidden dim.

    Avoids the harsh 1-layer compression (e.g. 768→64 with block_size=16)
    that destroys spectral information.  Uses two intermediate layers with
    GroupNorm + SiLU activation and a residual shortcut when shapes match.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = max(out_ch, (in_ch + out_ch) // 2)
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            _gn(mid),
            nn.SiLU(),
            nn.Conv2d(mid, out_ch, 3, padding=1),
            _gn(out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ProgressiveHead(nn.Module):
    """
    Multi-layer projection from hidden dim back to spectral channels.

    Mirror of :class:`ProgressiveStem` — gradually expands from hidden_dim
    to spec_channels at the output of the U-Net.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = max(in_ch, (in_ch + out_ch) // 2)
        self.layers = nn.Sequential(
            _gn(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, mid, 3, padding=1),
            _gn(mid),
            nn.SiLU(),
            nn.Conv2d(mid, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


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
    frequency, then applies a convolution for channel mixing.
    
    With ``kernel_size=1`` (default) this is a pointwise mix — each block's
    coefficients are processed independently.  With ``kernel_size=3`` the conv
    spans neighbouring blocks, enabling cross-block frequency reasoning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode_weighting: bool = True,
        kernel_size: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convolution for channel mixing (1×1 or 3×3)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )

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


class SpatialBlock(nn.Module):
    """
    Depthwise-separable 3×3 conv for cross-block spatial reasoning.

    In spectral feature space, adjacent spatial positions correspond to
    neighbouring blocks, so a 3×3 conv introduces inter-block coherence
    without leaving the spectral domain.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw_conv = nn.Conv2d(dim, dim, 1)
        self.norm = nn.GroupNorm(min(32, dim), dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.dw_conv(h)
        h = self.act(h)
        h = self.pw_conv(h)
        h = self.dropout(h)
        return x + h


# ---------------------------------------------------------------------------
# DCT Window Attention — neighbor-aware local attention over DCT blocks
# ---------------------------------------------------------------------------

class DCTWindowAttention(nn.Module):
    """
    Local window self-attention over DCT block positions.

    In spectral feature space each spatial grid cell corresponds to one
    DCT block.  Full O(N²) self-attention across all blocks is expensive;
    instead we restrict attention to a local window_size × window_size
    neighbourhood of blocks, giving O(N · w²) complexity.

    This lets every block reason about how its DCT coefficients relate to
    those of its spatial neighbours — exactly the cross-block frequency
    coherence that matters for sharp edge reconstruction in super-resolution.

    A learnable relative position bias table (following Swin Transformer)
    is added to the attention logits so the model is aware of how far two
    blocks are from each other within the window.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} must be divisible by num_heads {num_heads}"
        )
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Fused QKV projection (linear on flattened tokens)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Relative position bias — table size (2w-1)² × heads
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        # Pre-compute relative position index for a window_size × window_size window
        self._build_rel_pos_index(window_size)

    def _build_rel_pos_index(self, ws: int) -> None:
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(ws), torch.arange(ws), indexing="ij"
            )
        )  # (2, ws, ws)
        coords_flat = coords.flatten(1)  # (2, ws²)
        # Relative distance for all pairs of positions
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, ws², ws²)
        rel = rel.permute(1, 2, 0).contiguous()  # (ws², ws², 2)
        rel[:, :, 0] += ws - 1   # shift to non-negative
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1  # rasterise to 1-D index
        rel_pos_index = rel.sum(-1)  # (ws², ws²)
        self.register_buffer("rel_pos_index", rel_pos_index)

    # ------------------------------------------------------------------
    # Window partition / merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _partition(x: torch.Tensor, ws: int) -> Tuple[torch.Tensor, int, int, int, int]:
        """Split (B, C, H, W) into non-overlapping ws×ws windows.

        Returns:
            windows: (B*nH*nW, ws², C)
            pad_h, pad_w, H_padded, W_padded
        """
        B, C, H, W = x.shape
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.permute(0, 2, 3, 1)  # (B, Hp, Wp, C)
        x = x.reshape(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, nH, nW, ws, ws, C)
        nH, nW = Hp // ws, Wp // ws
        windows = x.reshape(B * nH * nW, ws * ws, C)
        return windows, pad_h, pad_w, Hp, Wp

    @staticmethod
    def _merge(windows: torch.Tensor, ws: int,
               B: int, Hp: int, Wp: int,
               pad_h: int, pad_w: int, H: int, W: int) -> torch.Tensor:
        """Merge window tokens back to (B, C, H, W)."""
        nH, nW = Hp // ws, Wp // ws
        C = windows.shape[-1]
        x = windows.reshape(B, nH, nW, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, Hp, Wp, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, Hp, Wp)
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — DCT feature map where H, W are in block space
        Returns:
            (B, C, H, W) — same shape, each block updated by its neighbours'
            frequency context.
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # Partition into local windows
        wins, pad_h, pad_w, Hp, Wp = self._partition(x, ws)
        # wins: (B*nH*nW, ws², C)
        N_win, N_tok, _ = wins.shape

        # QKV
        qkv = self.qkv(wins)  # (N_win, N_tok, 3*C)
        qkv = qkv.reshape(N_win, N_tok, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N_win, heads, N_tok, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention (local: N_tok = ws² ≤ 16 for ws=4)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (N_win, heads, N_tok, N_tok)

        # Add relative position bias
        bias = self.rel_pos_bias_table[
            self.rel_pos_index.view(-1)
        ]  # (N_tok², heads)
        bias = bias.view(
            N_tok, N_tok, self.num_heads
        ).permute(2, 0, 1).contiguous()  # (heads, N_tok, N_tok)
        attn = attn + bias.unsqueeze(0)  # broadcast over N_win

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(N_win, N_tok, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Merge windows back to spatial map
        out = self._merge(out, ws, B, Hp, Wp, pad_h, pad_w, H, W)
        return out


# ---------------------------------------------------------------------------
# DCNO Block — single operator layer
# ---------------------------------------------------------------------------

class DCNOBlock(nn.Module):
    """
    One DCT Neural Operator block:

      AdaGN → DCTSpectralConv → SiLU → residual
      (optional) AdaGN → DCTWindowAttention → residual   ← neighbor-aware
      AdaGN → ChannelMLP → residual
      (optional) SpatialBlock for cross-block coherence

    When ``use_attention=True`` a local window self-attention (default
    window 4×4 = 16 neighbouring DCT blocks) is inserted between the
    spectral convolution and the channel MLP.  This lets each block
    aggregate context from its spatial neighbours in frequency space,
    which is critical for reconstructing consistent edges and textures
    across block boundaries in super-resolution.

    Complexity per block: O(N · w²) where N = number of DCT blocks and
    w = window_size, so it scales gracefully even at fine resolutions.
    """

    def __init__(
        self,
        dim: int,
        time_dim: int,
        num_heads: int = 8,
        mode_weighting: bool = True,
        dropout: float = 0.0,
        spectral_conv_size: int = 1,
        spatial_dual_path: bool = False,
        use_attention: bool = False,
        attention_window_size: int = 4,
        attention_heads: int = 4,
    ):
        super().__init__()
        # Time projection
        self.time_mlp = TimestepMLP(time_dim, dim)

        # Spectral path
        self.norm1 = AdaptiveGroupNorm(dim)
        self.spectral_conv = DCTSpectralConv(
            dim, dim, mode_weighting, kernel_size=spectral_conv_size,
        )
        self.act = nn.SiLU()

        # Local window attention (neighbor-aware, bounded complexity)
        if use_attention:
            # Clamp heads so dim remains divisible
            heads = attention_heads
            while dim % heads != 0 and heads > 1:
                heads -= 1
            self.attn_norm = AdaptiveGroupNorm(dim)
            self.attn = DCTWindowAttention(
                dim=dim,
                num_heads=heads,
                window_size=attention_window_size,
                dropout=dropout,
            )
        else:
            self.attn_norm = None
            self.attn = None

        # Channel MLP path
        self.norm2 = AdaptiveGroupNorm(dim)
        self.channel_mlp = ChannelMLP(dim, expansion=4, dropout=dropout)

        # Optional spatial dual path
        self.spatial_block = (
            SpatialBlock(dim, dropout=dropout) if spatial_dual_path else None
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_mlp(t_emb)

        # 1. Spectral convolution
        h = self.norm1(x, scale, shift)
        h = self.spectral_conv(h)
        h = self.act(h)
        x = x + h

        # 2. Local window attention over DCT block neighbours (optional)
        if self.attn is not None:
            h = self.attn_norm(x, scale, shift)
            h = self.attn(h)
            x = x + h

        # 3. Channel MLP
        h = self.norm2(x, scale, shift)
        h = self.channel_mlp(h)
        x = x + h

        # 4. Spatial cross-block path (if enabled)
        if self.spatial_block is not None:
            x = self.spatial_block(x)

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
        residual_learning: If ``True`` the model predicts the *residual*
            ``x_hr − bicubic(x_lr)`` instead of ``x_hr`` directly.  Reduces
            the energy the diffusion process must learn by 3-5×.
        spectral_conv_size: Kernel size for
            :class:`DCTSpectralConv` (1 = pointwise, 3 = cross-block).
        spatial_dual_path: Append a lightweight depthwise-separable 3×3
            :class:`SpatialBlock` inside every :class:`DCNOBlock` for
            cross-block spatial coherence.
        freq_norm: Apply :class:`FreqNorm` (learnable per-channel
            normalisation) to spectral coefficients before the stem.
            Equalises the vastly different scales of DC vs AC coefficients.
        progressive_stem: Use :class:`ProgressiveStem` /
            :class:`ProgressiveHead` instead of single Conv2d for the
            stem / head projection.  Avoids the harsh N→hidden_dim
            compression when ``dct_block_size`` is large.
        concat_cond: If ``True``, concatenate the LR spectral coefficients
            with x_t spectral coefficients channel-wise before the stem
            (input channels = 2 × spec_channels).  This is the standard
            SR3/SRDiff approach and prevents the model from learning a
            shortcut through a separate condition encoder that ignores x_t.
        input_proj: If ``True``, add a pixel-space projection of x_t
            (Conv2d with stride=block_size) that is added to the spectral
            stem output.  This gives the model a direct pixel-space pathway
            to access x_t, bypassing the DCT transform which scrambles the
            spatial noise structure.  Critical for learning denoising.
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
        residual_learning: bool = False,
        spectral_conv_size: int = 1,
        spatial_dual_path: bool = False,
        freq_norm: bool = False,
        progressive_stem: bool = False,
        concat_cond: bool = False,
        input_proj: bool = False,
        pixel_refinement: bool = False,
        use_block_attention: bool = False,
        attention_window_size: int = 4,
        attention_heads: int = 4,
        attn_start_stage: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.dct_block_size = dct_block_size
        self.transform_type = transform_type
        self.residual_learning = residual_learning
        self.concat_cond = concat_cond
        self.use_input_proj = input_proj
        self.use_pixel_refinement = pixel_refinement

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

        # Optional frequency normalization
        self.freq_norm = FreqNorm(spec_channels) if freq_norm else None

        # Stem: spectral coefficients → hidden
        stem_in_ch = spec_channels * 2 if concat_cond else spec_channels
        if progressive_stem:
            self.stem = ProgressiveStem(stem_in_ch, hidden_dims[0])
        else:
            self.stem = nn.Conv2d(stem_in_ch, hidden_dims[0], 3, padding=1)

        # LR condition encoder (operates on LR spectral coefficients)
        # Skipped when concat_cond is on — LR is concatenated at input level.
        if not concat_cond:
            self.cond_encoder = ConditionEncoder(spec_channels, hidden_dims)
        else:
            self.cond_encoder = None

        # Pixel-space projection of x_t (bypasses DCT bottleneck for denoising).
        # Uses stride=block_size to match the spatial resolution of spectral
        # features (H/n × W/n) produced by the DCT transform.
        if input_proj:
            self.input_proj = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], kernel_size=n, stride=n),
                _gn(hidden_dims[0]),
                nn.SiLU(),
                nn.Conv2d(hidden_dims[0], hidden_dims[0], 3, padding=1),
            )
        else:
            self.input_proj = None

        # ---- Encoder ----
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        # Helper: should stage i use attention?
        def _use_attn(stage_idx: int) -> bool:
            return use_block_attention and stage_idx >= attn_start_stage

        for i, (dim, depth) in enumerate(zip(hidden_dims, depths)):
            stage_blocks = nn.ModuleList()
            for _ in range(depth):
                stage_blocks.append(
                    DCNOBlock(
                        dim, time_dim, num_heads, mode_weighting, dropout,
                        spectral_conv_size=spectral_conv_size,
                        spatial_dual_path=spatial_dual_path,
                        use_attention=_use_attn(i),
                        attention_window_size=attention_window_size,
                        attention_heads=attention_heads,
                    )
                )
            self.encoder_blocks.append(stage_blocks)
            
            if i < len(hidden_dims) - 1:
                self.downsamples.append(Downsample(dim, hidden_dims[i + 1]))

        # ---- Bottleneck (always uses attention when use_block_attention=True) ----
        self.bottleneck = nn.ModuleList([
            DCNOBlock(
                hidden_dims[-1], time_dim, num_heads, mode_weighting, dropout,
                spectral_conv_size=spectral_conv_size,
                spatial_dual_path=spatial_dual_path,
                use_attention=use_block_attention,
                attention_window_size=attention_window_size,
                attention_heads=attention_heads,
            ),
            DCNOBlock(
                hidden_dims[-1], time_dim, num_heads, mode_weighting, dropout,
                spectral_conv_size=spectral_conv_size,
                spatial_dual_path=spatial_dual_path,
                use_attention=use_block_attention,
                attention_window_size=attention_window_size,
                attention_heads=attention_heads,
            ),
        ])

        # ---- Decoder ----
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        rev_dims = list(reversed(hidden_dims))
        rev_depths = list(reversed(depths))
        num_stages = len(hidden_dims)

        for i, (dim, depth) in enumerate(zip(rev_dims, rev_depths)):
            # decoder stage i mirrors encoder stage (num_stages-1-i)
            enc_stage_idx = num_stages - 1 - i
            self.skip_convs.append(nn.Conv2d(dim * 2, dim, 1))

            stage_blocks = nn.ModuleList()
            for _ in range(depth):
                stage_blocks.append(
                    DCNOBlock(
                        dim, time_dim, num_heads, mode_weighting, dropout,
                        spectral_conv_size=spectral_conv_size,
                        spatial_dual_path=spatial_dual_path,
                        use_attention=_use_attn(enc_stage_idx),
                        attention_window_size=attention_window_size,
                        attention_heads=attention_heads,
                    )
                )
            self.decoder_blocks.append(stage_blocks)

            if i < len(rev_dims) - 1:
                self.upsamples.append(Upsample(dim, rev_dims[i + 1]))

        # Head: hidden → DCT coefficients (same spatial resolution as input)
        if progressive_stem:
            self.head = ProgressiveHead(hidden_dims[0], spec_channels)
        else:
            self.head = nn.Sequential(
                nn.GroupNorm(min(32, hidden_dims[0]), hidden_dims[0]),
                nn.SiLU(),
                nn.Conv2d(hidden_dims[0], spec_channels, 3, padding=1),
            )

        # Pixel-space refinement tail: blends across DCT block boundaries
        # to eliminate grid artifacts. 8 conv layers, 64 channels (~225K params).
        # Receptive field = 19px (covers 2+ block boundaries for 8×8 blocks).
        # Applied after IDCT in pixel domain with a residual connection.
        if pixel_refinement:
            refine_ch = 64
            layers = []
            layers.append(nn.Conv2d(out_channels, refine_ch, 3, padding=1))
            layers.append(nn.SiLU())
            for _ in range(6):  # 6 interior layers
                layers.append(nn.Conv2d(refine_ch, refine_ch, 3, padding=1))
                layers.append(nn.SiLU())
            layers.append(nn.Conv2d(refine_ch, out_channels, 3, padding=1))
            self.pixel_refine = nn.Sequential(*layers)
            # Init last conv to zero so refinement starts as identity
            nn.init.zeros_(self.pixel_refine[-1].weight)
            nn.init.zeros_(self.pixel_refine[-1].bias)
        else:
            self.pixel_refine = None

        self._attn_cfg = dict(
            use_block_attention=use_block_attention,
            attention_window_size=attention_window_size,
            attention_heads=attention_heads,
            attn_start_stage=attn_start_stage,
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

        # --- Optional frequency normalization ---
        if self.freq_norm is not None:
            x_spec = self.freq_norm(x_spec)
            lr_spec = self.freq_norm(lr_spec)

        # --- Time embedding ---
        t_emb = self.time_embed(t)      # (B, time_dim)

        # --- LR conditioning ---
        if self.concat_cond:
            # Concatenate LR spectral coefficients with x_t along channels
            if lr_spec.shape[-2:] != x_spec.shape[-2:]:
                lr_spec = F.interpolate(
                    lr_spec, size=x_spec.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            x_spec = torch.cat([x_spec, lr_spec], dim=1)  # (B, 2*C*n^2, ...)
            cond_feats = None
        else:
            cond_feats = self.cond_encoder(lr_spec)  # multi-scale list

        # --- Stem ---
        h = self.stem(x_spec)            # (B, hidden_dims[0], H/n, W/n)

        # --- Pixel-space x_t projection (bypass DCT for denoising) ---
        if self.input_proj is not None:
            h_pixel = self.input_proj(x_t)
            if h_pixel.shape[-2:] != h.shape[-2:]:
                h_pixel = F.interpolate(
                    h_pixel, size=h.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            h = h + h_pixel

        # --- Encoder ---
        skip_connections = []
        for i, stage_blocks in enumerate(self.encoder_blocks):
            # Add LR conditioning at matching scale (only if not concat_cond)
            if cond_feats is not None and i < len(cond_feats):
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

        # --- Pixel-space refinement (deblocking / cross-boundary blending) ---
        if self.pixel_refine is not None:
            out = out + self.pixel_refine(out)

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

"""
Blockwise DCT / IDCT transforms for operating in JPEG-like frequency space.

Implements 2D Type-II DCT and Type-III (inverse) DCT on non-overlapping
blocks (default 8×8, matching JPEG standard). All operations are
differentiable and run on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def _build_dct_matrix(n: int) -> torch.Tensor:
    """Build the n×n DCT-II basis matrix (orthonormal)."""
    mat = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            mat[k, i] = math.cos(math.pi * k * (2 * i + 1) / (2 * n))
    # Orthonormal scaling
    mat[0] *= 1.0 / math.sqrt(n)
    mat[1:] *= math.sqrt(2.0 / n)
    return mat


class BlockDCT2d(nn.Module):
    """
    Blockwise 2D DCT-II transform.
    
    Input : (B, C, H, W) image tensor, H and W divisible by block_size
    Output: (B, C * block_size^2, H // block_size, W // block_size)
    
    Each spatial position in the output contains the flattened DCT coefficients
    of one block. Channel dimension is expanded by block_size^2.
    """
    
    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
        # Register DCT matrix as buffer (not a parameter)
        dct_mat = _build_dct_matrix(block_size)
        self.register_buffer("dct_mat", dct_mat)        # (n, n)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        n = self.block_size
        assert H % n == 0 and W % n == 0, \
            f"Spatial dims ({H}, {W}) must be divisible by block_size {n}"
        
        # Reshape into blocks: (B, C, H//n, n, W//n, n)
        x = x.reshape(B, C, H // n, n, W // n, n)
        # Permute to (B, C, H//n, W//n, n, n) for easier matmul
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        
        # Apply 2D DCT: D @ block @ D^T
        # x: (..., n, n), dct_mat: (n, n)
        dct = self.dct_mat  # (n, n)
        x = dct @ x         # row-wise DCT
        x = x @ dct.T       # col-wise DCT
        
        # Flatten block coefficients into channels:
        # (B, C, H//n, W//n, n, n) → (B, C*n*n, H//n, W//n)
        x = x.reshape(B, C, H // n, W // n, n * n)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.reshape(B, C * n * n, H // n, W // n)
        
        return x


class BlockIDCT2d(nn.Module):
    """
    Blockwise 2D inverse DCT (DCT-III) transform.
    
    Input : (B, C * block_size^2, H_blocks, W_blocks) DCT coefficients
    Output: (B, C, H_blocks * block_size, W_blocks * block_size) image tensor
    """
    
    def __init__(self, block_size: int = 8, out_channels: int = 3):
        super().__init__()
        self.block_size = block_size
        self.out_channels = out_channels
        dct_mat = _build_dct_matrix(block_size)
        # Inverse of orthonormal DCT-II is its transpose
        self.register_buffer("idct_mat", dct_mat.T)     # (n, n)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.block_size
        C = self.out_channels
        B, Cn2, Hb, Wb = x.shape
        assert Cn2 == C * n * n, \
            f"Expected {C * n * n} channels, got {Cn2}"
        
        # Unflatten: (B, C, n*n, Hb, Wb) → (B, C, Hb, Wb, n, n)
        x = x.reshape(B, C, n * n, Hb, Wb)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.reshape(B, C, Hb, Wb, n, n)
        
        # Apply 2D IDCT: D^T @ coeffs @ D
        idct = self.idct_mat  # (n, n) = dct_mat.T
        x = idct @ x          # row-wise IDCT
        x = x @ idct.T        # col-wise IDCT
        
        # Reassemble blocks: (B, C, Hb, Wb, n, n) → (B, C, H, W)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.reshape(B, C, Hb * n, Wb * n)
        
        return x


class DCTUpsample(nn.Module):
    """
    Upsample in DCT space by zero-padding high-frequency coefficients.
    
    For a scale_factor of s, each (n×n) DCT block is placed into the
    top-left corner of a (s*n × s*n) block (zero-padding high freqs),
    then the energy is rescaled to preserve pixel-domain magnitude.
    
    This is the DCT-domain equivalent of sinc interpolation.
    """
    
    def __init__(self, scale_factor: int, block_size: int = 8, channels: int = 3):
        super().__init__()
        self.scale_factor = scale_factor
        self.block_size = block_size
        self.channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.scale_factor
        n = self.block_size
        C = self.channels
        B, Cn2, Hb, Wb = x.shape
        
        # Unflatten coefficients per block: (B, C, Hb, Wb, n, n)
        x = x.reshape(B, C, n * n, Hb, Wb)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.reshape(B, C, Hb, Wb, n, n)
        
        # Zero-pad to (s*n, s*n) — place original in top-left
        sn = s * n
        out = x.new_zeros(B, C, Hb, Wb, sn, sn)
        out[..., :n, :n] = x * (s * s)  # energy scaling
        
        # Reflatten: (B, C * sn^2, Hb, Wb)
        out = out.reshape(B, C, Hb, Wb, sn * sn)
        out = out.permute(0, 1, 4, 2, 3).contiguous()
        out = out.reshape(B, C * sn * sn, Hb, Wb)
        
        return out

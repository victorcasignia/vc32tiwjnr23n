# DWNO-SR: Deep Wavelet Neural Operator for Image Super-Resolution

A novel super-resolution model that combines **Wavelet Neural Operator** theory
with **learnable lifting-scheme wavelets** and **sparse local window attention**.

---

## Architecture Overview

```
LR (B,3,H,W)
     │
     ▼
ShallowFeatureExtractor (3×3 conv × 2)
     │
     ├───────────────────────────── long skip ──────────────────────────┐
     ▼                                                                   │
DeepFeatureExtractor (3 × WNOStage)                                     │
  └── WNOStage (4 WNOBlocks, alternating window shifts)                 │
        └── WNOBlock                                                     │
              ├── DWT (J levels, learnable lifting)                      │
              │    ├── LL  → SubbandOp → WindowMSA                      │
              │    ├── LH_j → SubbandOp → WindowMSA (+ cross-scale←LL) │
              │    ├── HL_j → SubbandOp → WindowMSA (+ cross-scale←LL) │
              │    └── HH_j → SubbandOp → WindowMSA (+ cross-scale←LL) │
              └── IDWT                                                   │
     │                                                                   │
     ▼                                                                   │
   + ←──────────────────────────────────────────────────────────────────┘
     │
     ▼
PixelShuffleUpsample (ICNR-init, ×scale)
     │
     ▼
Refinement conv (3×3 → 3×3 → output)
     │
     ▼
HR (B,3,H·scale,W·scale)
```

---

## Novel Contributions

### 1. Learnable Lifting-Scheme DWT
Based on Sweldens' lifting framework, we replace fixed filter banks with small
CNN `Predict` and `Update` steps:

```
d[n] = x_odd[n]  − P(x_even)[n]    (predict step)
s[n] = x_even[n] + U(d)[n]         (update step)
```

**Perfect reconstruction is guaranteed by construction** regardless of the learned
filter values.  The lifting structure also provides bounded Jacobians, preventing
gradient explosion/vanishing through the wavelet layers.

### 2. Wavelet Neural Operator (WNO) Layer
Formulated as an integral operator in the wavelet domain:

$$v(x) = \sigma\!\left(W_\text{local}\,u(x) + \mathcal{W}^{-1}\!\left(\sum_{j,d} R_{j,d}\cdot\mathcal{W}_{j,d}[u]\right)\right)$$

- `R_{j,d}`: learnable channel-mixing operator per *(level j, direction d)*
- `W_local`: local window attention (sparse, O(HW·M²))
- The DWT basis provides **both frequency AND spatial localisation**, unlike FFT-based FNO

### 3. Sparse Local Window Attention on Subbands
Applied inside each wavelet subband (already spatially downsampled by 2^j),
with Swin-style shifted windows for cross-window connectivity.

- Non-shifted and shifted blocks alternate (depth must be even per stage)
- Wavelet-aware relative position bias (scale-modulated)

### 4. Cross-Scale Attention
HF subbands (LH, HL, HH) attend to the coarse LL approximation:

$$A_\text{cross} = \text{softmax}\!\left(\frac{Q_j^\text{HF}(K_{j+1}^{LL})^\top}{\sqrt{d}}\right) V_{j+1}^{LL}$$

This lets edge/texture detail be guided by global amplitude structure.

### 5. Wavelet-Domain Loss
Penalises coefficient errors at each scale, weighting HF subbands more:

$$\mathcal{L}_\text{wave} = \sum_{j,d} w_{j,d}\,\|\mathcal{W}_{j,d}[\text{SR}] - \mathcal{W}_{j,d}[\text{HR}]\|_1$$

Combined with L1 + differentiable SSIM + lifting orthogonality regulariser.

---

## Known Pitfalls & Mitigations

| Pitfall | Mitigation |
|---------|-----------|
| Boundary artifacts from DWT padding | Reflect/circular padding; always trim to original size after IDWT |
| Aliasing from unconstrained lifting filters | Orthogonality regularisation loss `λ_orth · ‖WW⊤ − I‖_F²` |
| Gradient explosion through multi-level DWT | Learnable lifting Jacobian bounded; gradient clipping (norm=1.0) |
| Window size must divide subband dimensions | Automatic padding before windowing, stripped after |
| Cross-scale attention cost at finest level | Full attention only on downsampled subbands (2^J smaller) |
| Checkerboard artifacts from PixelShuffle | ICNR initialisation → smooth nearest-neighbour baseline at init |
| Over-smoothing from pixel-domain L1 alone | Wavelet-domain loss emphasises HF coefficients |

---

## Setup

```bash
# Activate your dcno conda environment
conda activate dcno

# Install dependencies
pip install -r requirements.txt
```

---

## Training

```bash
# From v2/
# Quick sanity check (small model, 5 epochs)
python -m scripts.train --config configs/dwno_debug.yaml

# Full ×4 training on DIV2K
python -m scripts.train --config configs/dwno_div2k_x4.yaml

# Full ×2 training
python -m scripts.train --config configs/dwno_div2k_x2.yaml

# Resume from checkpoint
python -m scripts.train --config configs/dwno_div2k_x4.yaml --resume checkpoints/div2k_x4/latest.pth
```

---

## Smoke Test

```bash
python -m scripts.smoke_test
```

Expected output:
```
── DWT perfect-reconstruction ──
  levels=1  MSE=<1e-8  PASS ✓
  levels=2  MSE=<1e-8  PASS ✓
  levels=3  MSE=<1e-8  PASS ✓
  odd input (65×65)  MSE=<1e-8  PASS ✓
── Model forward pass ──
  Parameters: ~X.XXM
  Input:  (2, 3, 64, 64)
  Output: (2, 3, 256, 256)  PASS ✓
── Loss backward pass ──
  Loss = X.XXXX  Gradients finite: True  PASS ✓
```

---

## Data Layout

Expected structure (already present):
```
data/
  DIV2K/
    DIV2K_train_HR/
    DIV2K_train_LR_bicubic/
      X2/
      X4/
    DIV2K_valid_HR/
    DIV2K_valid_LR_bicubic/
      X2/
      X4/
```

If `train_lr_dir` / `val_lr_dir` are unset in the config, LR images are
generated on-the-fly via bicubic downsampling.

---

## Configuration Reference

Key parameters in the YAML configs:

| Key | Description | Default |
|-----|-------------|---------|
| `scale` | Upscale factor | 4 |
| `channels` | Feature channels | 96 |
| `stage_depths` | WNO blocks per stage | [4,4,4] |
| `levels` | DWT levels per WNO block | 2 |
| `window_size` | Local attention window M | 8 |
| `num_heads` | Attention heads | 4 |
| `filter_len` | Lifting filter taps | 3 |
| `cross_scale` | Cross-scale HF↔LL attention | true |
| `lambda_wave` | Wavelet loss weight | 0.05 |
| `lambda_orth` | Lifting orthogonality weight | 1e-4 |
| `ema` | Use EMA weights for validation | true |

---

## References

- Sweldens (1997) *The Lifting Scheme: A Construction of Second Generation Wavelets*
- Li et al. (2021) *Fourier Neural Operator* — arXiv:2010.08895
- Tripura & Chakraborty (2022) *Wavelet Neural Operator* — arXiv:2205.02191
- Liu et al. (2021) *Swin Transformer* — arXiv:2103.14030
- Lim et al. (2017) *EDSR* — arXiv:1707.02921
- Chen et al. (2021) *SwinIR* — arXiv:2108.10257

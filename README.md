# DCNO — DCT Neural Operator for Super-Resolution

A diffusion-based neural operator that operates entirely in the **Discrete Cosine Transform (DCT)** coefficient space for image super-resolution. Because DCT is the backbone of JPEG compression, DCNO is naturally suited for learning the compression→super-resolution mapping.

## Key Ideas

| Component | Description |
|---|---|
| **DCT Neural Operator** | All operator layers work on blockwise DCT coefficients (8×8, matching JPEG). Learnable spectral convolutions weight each DCT mode adaptively. |
| **Rectified Flow Matching** | Modern diffusion formulation: learn a velocity field $v(x_t, t)$ where $x_t = (1-t) x_0 + t \epsilon$. Straight ODE paths for fast sampling. |
| **Consistency Distillation** | Optional one/few-step inference via self-consistency training, eliminating the need for a teacher model (inspired by CTMSR, FlowSR). |
| **Adaptive Time-Step ODE** | Deterministic ODE solver with adaptive step sizes for quality-speed trade-off at inference. |

## Architecture Overview

```
Input LR Image
     │
     ▼
┌─────────────┐
│  Blockwise   │   8×8 block DCT (like JPEG)
│  DCT Transform│
└──────┬──────┘
       │  [B, 64, H/8, W/8]  (64 = 8×8 DCT modes per block × C channels)
       ▼
┌─────────────┐
│  Stem Conv   │   Project DCT coefficients → hidden dim
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│     DCT Neural Operator U-Net       │
│                                     │
│  ┌───────────┐  Encoder blocks:     │
│  │ DCTNOBlock│  Spectral Conv +     │
│  │ DCTNOBlock│  AdaGN (time cond.)  │
│  └─────┬─────┘  + Channel MLP       │
│        │ ↓ downsample                │
│  ┌─────┴─────┐                      │
│  │ DCTNOBlock│  (deeper features)   │
│  └─────┬─────┘                      │
│        │ bottleneck                  │
│  ┌─────┴─────┐                      │
│  │ DCTNOBlock│  Decoder + skip conn │
│  │ DCTNOBlock│  + upsample          │
│  └─────┬─────┘                      │
│        │                            │
└────────┼────────────────────────────┘
         │
         ▼
┌─────────────┐
│  Head Conv   │   Project back → DCT coefficient space
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Blockwise   │   Inverse DCT → pixel space
│  IDCT        │
└──────┬──────┘
       │
       ▼
  Output HR Image
```

### DCTNOBlock Detail

Each operator block contains:

1. **Adaptive Group Norm (AdaGN)** — conditioned on diffusion timestep $t$
2. **DCT Spectral Convolution** — learnable complex weights per DCT frequency mode with adaptive mode rebalancing
3. **SiLU activation**
4. **Pointwise Channel MLP** — two-layer MLP for channel mixing
5. **Residual connection**

### Diffusion: Rectified Flow Matching

Training objective (velocity prediction):

$$\mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}(0,1),\, x_0,\, \epsilon} \left\| v_\theta(x_t, t, c) - (\epsilon - x_0) \right\|^2$$

where $x_t = (1-t) x_0 + t \epsilon$ and $c$ is the LR conditioning signal.

Inference uses an ODE solver: $x_0 = x_1 - \int_1^0 v_\theta(x_t, t, c)\, dt$

For low-step inference, consistency training enforces:

$$f_\theta(x_t, t) = f_\theta(x_{t'}, t') \quad \forall\, t, t' \text{ on the same ODE trajectory}$$

## Project Structure

```
dcno/
├── README.md
├── requirements.txt
├── configs/
│   ├── dcno_div2k_x2.yaml      # ×2 SR on DIV2K
│   ├── dcno_div2k_x4.yaml      # ×4 SR on DIV2K
│   └── dcno_df2k_x4.yaml       # ×4 SR on DF2K (DIV2K + Flickr2K)
├── models/
│   ├── __init__.py
│   ├── dct.py                   # Blockwise DCT / IDCT transforms
│   ├── dcno.py                  # DCT Neural Operator architecture
│   ├── diffusion.py             # Rectified flow + consistency distillation
│   └── ema.py                   # Exponential moving average
├── optim/
│   ├── __init__.py
│   └── musgd.py                 # Muon-SGD hybrid optimizer
├── scripts/
│   ├── download_data.py         # Download DIV2K, Flickr2K, Set5, Set14, BSD100, Urban100
│   ├── train.py                 # Training with wandb logging
│   ├── test.py                  # Evaluation with PSNR/SSIM/LPIPS
│   └── visualize.py             # Generate SR comparison grids
└── data/                        # Downloaded datasets go here
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download DIV2K (default training set) + benchmark test sets
python -m scripts.download_data --datasets div2k set5 set14 bsd100 urban100
```

### 3. Train

```bash
# Train ×4 SR on DIV2K
python -m scripts.train --config configs/dcno_div2k_x4.yaml

# Train with specific optimizer
python -m scripts.train --config configs/dcno_div2k_x4.yaml --optimizer adamw

# Resume training
python -m scripts.train --config configs/dcno_div2k_x4.yaml --resume checkpoints/latest.pth
```

### 4. Test

```bash
# Evaluate on benchmark datasets
python -m scripts.test --config configs/dcno_div2k_x4.yaml --checkpoint checkpoints/best.pth --datasets set5 set14 bsd100 urban100

# Single image SR
python -m scripts.test --checkpoint checkpoints/best.pth --input path/to/image.jpg --scale 4
```

## Configuration

All hyperparameters are controlled via YAML configs. Key options:

```yaml
model:
  hidden_dims: [64, 128, 256, 512]   # U-Net channel dimensions
  depths: [2, 2, 4, 2]               # Blocks per stage
  dct_block_size: 8                   # DCT block size (8 = JPEG standard)
  num_heads: 8                        # Attention heads (channel MLP)
  mode_weighting: true                # Adaptive DCT mode rebalancing

diffusion:
  type: "rectified_flow"              # rectified_flow | ddpm
  num_train_timesteps: 1000
  num_inference_steps: 10             # ODE steps at inference
  consistency_training: false         # Enable consistency distillation
  consistency_steps: 1                # Target inference steps after distillation

training:
  optimizer: "adamw"                  # adam | adamw | musgd
  learning_rate: 2.0e-4
  scheduler:
    type: "cosine"                    # cosine | linear | constant_with_warmup
    warmup_steps: 1000
    min_lr: 1.0e-6
```

## Optimizer Options

| Optimizer | Best For |
|---|---|
| `adam` | General purpose, stable training |
| `adamw` | Better generalization with weight decay (default) |
| `musgd` | Muon-SGD hybrid — Muon for 2D+ params (spectral orthogonalization), SGD for 1D params |

## Datasets

| Dataset | Images | Usage |
|---|---|---|
| DIV2K | 800 train / 100 val | Primary training set |
| Flickr2K | 2650 | Extended training (DF2K = DIV2K + Flickr2K) |
| Set5 | 5 | Benchmark test |
| Set14 | 14 | Benchmark test |
| BSD100 | 100 | Benchmark test |
| Urban100 | 100 | Benchmark test (challenging urban structures) |

## Metrics

- **PSNR** (Peak Signal-to-Noise Ratio) — pixel fidelity
- **SSIM** (Structural Similarity) — perceptual structure
- **LPIPS** (Learned Perceptual Image Patch Similarity) — deep perceptual quality

## References

- DiffFNO: Diffusion Fourier Neural Operator (CVPR 2025)
- JPNeO: JPEG Processing Neural Operator (ICCV 2025)
- AJQE: Uncover Treasures in DCT (ICCV 2025)
- FlowSR: Fast SR via Consistency Rectified Flow (ICCV 2025)
- CTMSR: Consistency Trajectory Matching for SR (ICCV 2025)
- FluxSR: One Diffusion Step to Real-World SR (ICML 2025)

## License

MIT

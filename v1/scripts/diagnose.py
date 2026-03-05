"""Quick diagnostic: loss breakdown by timestep bin + gradient analysis."""
import torch
import torch.nn.functional as F
import yaml
import sys
sys.path.insert(0, ".")

from models.dcno import DCNO
from models.diffusion import RectifiedFlow
from scripts.dataset import build_dataset

device = "mps" if torch.backends.mps.is_available() else "cpu"

with open("configs/dcno_div2k_x4.yaml") as f:
    cfg = yaml.safe_load(f)

# Build model and load latest checkpoint
mcfg = cfg["model"]
model = DCNO(
    in_channels=mcfg.get("in_channels", 3),
    out_channels=mcfg.get("out_channels", 3),
    hidden_dims=mcfg.get("hidden_dims", [64, 128, 256, 512]),
    depths=mcfg.get("depths", [2, 2, 4, 2]),
    dct_block_size=mcfg.get("dct_block_size", 8),
    num_heads=mcfg.get("num_heads", 8),
    mode_weighting=mcfg.get("mode_weighting", True),
    scale_factor=mcfg.get("scale_factor", 4),
    dropout=0.0,
    transform_type=mcfg.get("transform_type", "dct"),
    freq_norm=mcfg.get("freq_norm", False),
    progressive_stem=mcfg.get("progressive_stem", False),
).to(device)

nparams = sum(p.numel() for p in model.parameters())
print(f"Model params: {nparams:,}")

# Load latest checkpoint if exists
import os
ckpt_path = "checkpoints/latest.pth"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
else:
    print("No checkpoint found, using random weights")

model.eval()

# Load a few images
train_ds = build_dataset("div2k", "./data", scale_factor=4, patch_size=192, augment=False, split="train", subset=0.125)
loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
hr, lr = next(iter(loader))
hr, lr = hr.to(device), lr.to(device)
lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)

print(f"\nData stats:")
print(f"  HR: {hr.shape}, range [{hr.min():.3f}, {hr.max():.3f}], mean={hr.mean():.3f}, std={hr.std():.3f}")
print(f"  LR: {lr.shape}, range [{lr.min():.3f}, {lr.max():.3f}]")

# Compute v_target stats
noise = torch.randn_like(hr)
v_target = noise - hr
print(f"  v_target: mean={v_target.mean():.3f}, std={v_target.std():.3f}, norm={v_target.norm():.1f}")
print(f"  Per-element E[v²] = {(v_target**2).mean():.4f}")
print(f"  → Trivial loss (predict 0): {(v_target**2).mean():.4f}")

# Loss per timestep bin
print(f"\n--- Loss per timestep bin ---")
bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

with torch.no_grad():
    for lo, hi in bins:
        losses = []
        trivial_losses = []
        vp_norms = []
        vt_norms = []
        for _ in range(20):  # average over multiple noise samples
            t = torch.full((hr.shape[0],), (lo + hi) / 2, device=device)
            noise = torch.randn_like(hr)
            x_t = (1 - t[:, None, None, None]) * hr + t[:, None, None, None] * noise
            v_target = noise - hr
            v_pred = model(x_t, t, lr_up)
            
            loss = F.mse_loss(v_pred, v_target).item()
            trivial = (v_target ** 2).mean().item()
            losses.append(loss)
            trivial_losses.append(trivial)
            vp_norms.append(v_pred.norm().item())
            vt_norms.append(v_target.norm().item())
        
        avg_loss = sum(losses) / len(losses)
        avg_trivial = sum(trivial_losses) / len(trivial_losses)
        avg_vp = sum(vp_norms) / len(vp_norms)
        avg_vt = sum(vt_norms) / len(vt_norms)
        explained = (avg_trivial - avg_loss) / avg_trivial * 100
        print(f"  t=[{lo:.1f},{hi:.1f}]: loss={avg_loss:.4f}, trivial={avg_trivial:.4f}, "
              f"explained={explained:+.1f}%, vp/vt={avg_vp/avg_vt:.3f}")

# Gradient norm diagnostic (which layers have biggest gradients?)
print(f"\n--- Gradient magnitudes for one training step ---")
model.train()
t = torch.rand(hr.shape[0], device=device)
noise = torch.randn_like(hr)
x_t = (1 - t[:, None, None, None]) * hr + t[:, None, None, None] * noise
v_target = noise - hr
v_pred = model(x_t, t, lr_up)
loss = F.mse_loss(v_pred, v_target)
loss.backward()

layer_grads = {}
for name, p in model.named_parameters():
    if p.grad is not None:
        layer_grads[name] = p.grad.norm().item()

# Print top 10 and bottom 10 by gradient norm
sorted_grads = sorted(layer_grads.items(), key=lambda x: x[1], reverse=True)
print("  Top 10 gradient norms:")
for name, gn in sorted_grads[:10]:
    print(f"    {gn:.6f}  {name}")
print("  Bottom 10 gradient norms:")
for name, gn in sorted_grads[-10:]:
    print(f"    {gn:.6f}  {name}")

# Check what fraction of total gradient each part of the network gets
stem_grad = sum(v for k, v in layer_grads.items() if 'stem' in k)
head_grad = sum(v for k, v in layer_grads.items() if 'head' in k)
encoder_grad = sum(v for k, v in layer_grads.items() if 'encoder' in k or 'down' in k)
decoder_grad = sum(v for k, v in layer_grads.items() if 'decoder' in k or 'up' in k)
total = sum(layer_grads.values())
print(f"\n  Gradient distribution:")
print(f"    stem:    {stem_grad/total*100:.1f}%")
print(f"    head:    {head_grad/total*100:.1f}%")
print(f"    encoder: {encoder_grad/total*100:.1f}%")
print(f"    decoder: {decoder_grad/total*100:.1f}%")
print(f"    total norm: {total:.4f}")

"""Diagnose why PSNR is ~6 dB despite training loss decreasing."""
import sys; sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
from scripts.dataset import build_dataset
from torch.utils.data import DataLoader

ds = build_dataset('div2k', './data', scale_factor=4, patch_size=192, augment=False, split='val')
dl = DataLoader(ds, batch_size=8, shuffle=False)
hr, lr = next(iter(dl))

def psnr(a, b):
    mse = (a - b).pow(2).mean(dim=[1,2,3])
    return (-10 * torch.log10(mse)).mean().item()

lr_up = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)

print("=== PSNR Baselines ===")
print(f"Bicubic (lr_up):              {psnr(lr_up, hr):.2f} dB")

# Simulate what happens when model output ≈ 0 (eps-skip does nothing)
# With residual + lr_init: x_init = 0.8*noise, after ODE with pred≈0 → x stays ≈ 0.8*noise
# Final: (lr_up + 0.8*noise).clamp(0,1)
noise = torch.randn_like(hr)
for scale in [0.8, 0.4, 0.2, 0.1, 0.08, 0.01]:
    sr = (lr_up + scale * noise).clamp(0, 1)
    print(f"lr_up + {scale:.2f}*noise:          {psnr(sr, hr):.2f} dB")

# What if model correctly removes ALL noise (x0 = residual)?
residual = hr - lr_up
print(f"\nResidual stats: mean={residual.mean():.4f}, std={residual.std():.4f}, range=[{residual.min():.3f}, {residual.max():.3f}]")
print(f"Perfect residual:             {psnr(lr_up + residual, hr):.2f} dB (=inf, same as HR)")

# Now test: if an x0-prediction model outputs zeros (=predict zero residual):
# ODE steps: at each step, x_0_pred=0, eps=(x_t - 0)/t = x_t/t
# x_next = (1-s)*0 + s*(x_t/t) = s/t * x_t
# Starting from x=0.8*noise, after 10 steps: x = (0.08/0.8)*0.8*noise = 0.08*noise
print("\n=== Simulated ODE with x0-prediction (model outputs 0) ===")
x = 0.8 * noise
t_max = 0.8
num_steps = 10
dt = t_max / num_steps
for step in range(num_steps):
    t_val = t_max - step * dt
    s = t_val - dt
    x_0_pred = torch.zeros_like(x)  # model predicts zero
    if t_val > 1e-3:
        eps_inferred = (x - (1 - t_val) * x_0_pred) / t_val
    else:
        eps_inferred = x
    if s <= 1e-6:
        x = x_0_pred
    else:
        x = (1 - s) * x_0_pred + s * eps_inferred
    if step % 3 == 0 or step == num_steps - 1:
        sr = (lr_up + x).clamp(0, 1)
        print(f"  Step {step+1}, t={t_val:.2f}→{s:.2f}: ODE x norm={x.norm():.3f}, PSNR={psnr(sr, hr):.2f} dB")

print("\n=== Simulated ODE with eps-skip (model outputs 0) ===")
x = 0.8 * noise
for step in range(num_steps):
    t_val = t_max - step * dt
    s = t_val - dt
    pred = torch.zeros_like(x)  # model predicts zero correction
    eps_pred = pred + x  # eps = x_t (skip only)
    one_minus_t = 1.0 - t_val
    if one_minus_t < 1e-3:
        x_0_pred = torch.zeros_like(x)
    else:
        x_0_pred = (x - t_val * eps_pred) / one_minus_t
    if s <= 1e-6:
        x = x_0_pred
    else:
        x = (1 - s) * x_0_pred + s * eps_pred
    if step % 3 == 0 or step == num_steps - 1:
        sr = (lr_up + x).clamp(0, 1)
        print(f"  Step {step+1}, t={t_val:.2f}→{s:.2f}: ODE x norm={x.norm():.3f}, PSNR={psnr(sr, hr):.2f} dB")

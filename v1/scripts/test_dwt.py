"""Smoke test: verify DWT transform and DCNO with DWT work end-to-end."""
import torch
from models import DCNO, RectifiedFlow

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# --- DWT roundtrip ---
from models.dwt import BlockDWT2d, BlockIDWT2d

x = torch.randn(2, 3, 64, 64)
dwt = BlockDWT2d(8)
idwt = BlockIDWT2d(8, 3)
coeffs = dwt(x)
recon = idwt(coeffs)
err = (recon - x).abs().max().item()
print(f"DWT roundtrip: {x.shape} -> {coeffs.shape} -> {recon.shape}, err={err:.2e}")
assert err < 1e-5, f"Roundtrip error too large: {err}"

# --- DCNO with DWT ---
model = DCNO(
    in_channels=3, out_channels=3,
    hidden_dims=[32, 64], depths=[1, 1],
    dct_block_size=8, transform_type="dwt",
    scale_factor=4, dropout=0.0,
).to(device)
print(f"Transform: {model.transform_type}")
print(f"fwd: {model.fwd_transform}")
print(f"inv: {model.inv_transform}")

B, C, H, W = 1, 3, 64, 64
x_t = torch.randn(B, C, H, W, device=device)
t = torch.rand(B, device=device)
x_lr = torch.randn(B, C, H // 4, W // 4, device=device)

v = model(x_t, t, x_lr)
print(f"DWT forward: {x_t.shape} + LR {x_lr.shape} -> v {v.shape}")

flow = RectifiedFlow()
x_hr = torch.randn(B, C, H, W, device=device)
result = flow.training_step(model, x_hr, x_lr)
loss = result["loss"]
loss.backward()
print(f"DWT training loss: {loss.item():.4f}  (backward OK)")

# --- DCNO with DCT still works ---
model_dct = DCNO(
    in_channels=3, out_channels=3,
    hidden_dims=[32, 64], depths=[1, 1],
    dct_block_size=8, transform_type="dct",
    scale_factor=4,
).to(device)
v2 = model_dct(x_t, t, x_lr)
print(f"DCT forward: v {v2.shape}  (still works)")

print("\nAll tests passed!")

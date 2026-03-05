"""
Smoke test: verifies the full model forward/backward pass and
DWT perfect-reconstruction property.

Run from v2/:
    python -m scripts.smoke_test
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dwt import DWTForward2D, LearnableWavelet2D
from models.dwno import DWNOS, DWNOLoss


# ---------------------------------------------------------------------------
# 1. DWT perfect-reconstruction test
# ---------------------------------------------------------------------------

def test_dwt_reconstruction():
    print("── DWT perfect-reconstruction ──────────────────────────────────")
    torch.manual_seed(0)
    B, C, H, W = 2, 8, 64, 64

    for levels in (1, 2, 3):
        dwt = DWTForward2D(C, levels=levels, filter_len=3)
        x   = torch.randn(B, C, H, W)
        LL, highs = dwt(x)
        x_rec     = dwt.inverse(LL, highs)
        err = F.mse_loss(x_rec, x).item()
        status = "PASS ✓" if err < 1e-8 else f"FAIL (mse={err:.2e})"
        print(f"  levels={levels}  MSE={err:.2e}  {status}")

    # Test with odd-sized input
    x_odd = torch.randn(B, C, 65, 65)
    dwt2  = DWTForward2D(C, levels=2, filter_len=3)
    LL, highs = dwt2(x_odd)
    x_rec     = dwt2.inverse(LL, highs)
    err = F.mse_loss(x_rec, x_odd).item()
    status = "PASS ✓" if err < 1e-8 else f"FAIL (mse={err:.2e})"
    print(f"  odd input (65×65)  MSE={err:.2e}  {status}")


# ---------------------------------------------------------------------------
# 2. Model forward pass
# ---------------------------------------------------------------------------

def test_model_forward():
    print("\n── Model forward pass ──────────────────────────────────────────")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 3, 64, 64
    scale = 4

    model = DWNOS(
        scale=scale,
        channels=32,
        stage_depths=[2, 2],
        levels=1,
        kernel_size=7,
        mlp_ratio=4.0,
        drop_path=0.0,
        filter_len=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.2f}M")

    lr = torch.randn(B, C, H, W, device=device)
    with torch.no_grad():
        sr = model(lr)

    expected = (B, C, H * scale, W * scale)
    status = "PASS ✓" if tuple(sr.shape) == expected else f"FAIL shape={sr.shape}"
    print(f"  Input:  {tuple(lr.shape)}")
    print(f"  Output: {tuple(sr.shape)}  expected {expected}  {status}")


# ---------------------------------------------------------------------------
# 3. Loss backward pass
# ---------------------------------------------------------------------------

def test_loss_backward():
    print("\n── Loss backward pass ──────────────────────────────────────────")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 1, 3, 64, 64
    scale = 2

    model = DWNOS(
        scale=scale,
        channels=16,
        stage_depths=[2],
        levels=1,
        kernel_size=7,
        mlp_ratio=4.0,
        drop_path=0.0,
        filter_len=3,
    ).to(device)

    criterion = DWNOLoss(lambda_orth=1e-4).to(device)

    lr = torch.rand(B, C, H, W, device=device)
    hr = torch.rand(B, C, H * scale, W * scale, device=device)

    sr = model(lr)
    loss, info = criterion(sr, hr, model)
    loss.backward()

    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    status = "PASS ✓" if grad_ok and torch.isfinite(loss) else "FAIL"
    print(f"  Loss = {loss.item():.4f}  Gradients finite: {grad_ok}  {status}")
    for k, v in info.items():
        print(f"    {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DWNO-SR smoke test")
    print("=" * 60)
    test_dwt_reconstruction()
    test_model_forward()
    test_loss_backward()
    print("\nAll smoke tests complete.")

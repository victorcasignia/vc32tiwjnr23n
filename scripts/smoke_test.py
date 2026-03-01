"""Smoke test for all ablation flag combinations."""
import torch
import torch.nn.functional as F
from models.dcno import DCNO
from models.diffusion import RectifiedFlow

device = "mps" if torch.backends.mps.is_available() else "cpu"


def test(label, model_kwargs, test_residual=False, hr_size=64):
    print(f"=== {label} ===")
    m = DCNO(**model_kwargs).to(device)
    nparams = sum(p.numel() for p in m.parameters())
    print(f"  params: {nparams:,}")

    sf = model_kwargs.get("scale_factor", 4)
    lr_size = hr_size // sf
    x = torch.randn(1, 3, hr_size, hr_size, device=device)
    t = torch.rand(1, device=device)
    xlr = torch.randn(1, 3, lr_size, lr_size, device=device)
    out = m(x, t, xlr)
    print(f"  fwd: {x.shape} -> {out.shape}")
    out.sum().backward()
    print("  bwd: OK")

    # Also test through the diffusion wrapper
    flow = RectifiedFlow()
    hr = torch.randn(1, 3, hr_size, hr_size, device=device)
    lr = torch.randn(1, 3, lr_size, lr_size, device=device)
    lr_up = F.interpolate(lr, size=(hr_size, hr_size), mode="bicubic", align_corners=False)

    x_bic = lr_up if test_residual else None
    m.zero_grad()
    result = flow.training_step(m, hr, lr_up, x_bicubic=x_bic)
    print(f"  loss: {result['loss'].item():.4f}")
    result["loss"].backward()
    print("  training_step bwd: OK")

    m.eval()
    sr = flow.sample(m, lr, shape=(1, 3, hr_size, hr_size), device=device, x_bicubic=x_bic)
    print(f"  sample: {sr.shape}, range: [{sr.min():.2f}, {sr.max():.2f}]")
    print()


base = dict(hidden_dims=[32, 64], depths=[1, 1], dct_block_size=8, scale_factor=4)

test("1. Baseline", {**base})
test("2. residual_learning", {**base, "residual_learning": True}, test_residual=True)
test("3. spectral_conv_size=3", {**base, "spectral_conv_size": 3})
test("4. spatial_dual_path", {**base, "spatial_dual_path": True})
test("5. All three ON", {**base, "residual_learning": True, "spectral_conv_size": 3, "spatial_dual_path": True}, test_residual=True)
test("6. DWT + all", {**base, "transform_type": "dwt", "residual_learning": True, "spectral_conv_size": 3, "spatial_dual_path": True}, test_residual=True)
test("7. freq_norm only", {**base, "freq_norm": True})
test("8. progressive_stem only", {**base, "progressive_stem": True})
test("9. freq_norm + progressive_stem", {**base, "freq_norm": True, "progressive_stem": True})
test("10. All flags ON", {**base, "residual_learning": True, "spectral_conv_size": 3, "spatial_dual_path": True, "freq_norm": True, "progressive_stem": True}, test_residual=True)
test("11. block_size=16 + progressive_stem", {**base, "dct_block_size": 16, "progressive_stem": True, "freq_norm": True, "hidden_dims": [64, 128], "depths": [1, 1]}, hr_size=128)
test("12. concat_cond only", {**base, "concat_cond": True})
test("13. concat_cond + freq_norm + progressive_stem", {**base, "concat_cond": True, "freq_norm": True, "progressive_stem": True})
test("14. All flags + concat_cond", {**base, "residual_learning": True, "spectral_conv_size": 3, "spatial_dual_path": True, "freq_norm": True, "progressive_stem": True, "concat_cond": True}, test_residual=True)
test("15. input_proj only", {**base, "input_proj": True})
test("16. input_proj + concat_cond + progressive_stem", {**base, "input_proj": True, "concat_cond": True, "progressive_stem": True, "freq_norm": True})

print("ALL TESTS PASSED")

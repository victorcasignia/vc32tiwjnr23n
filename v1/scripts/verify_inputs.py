"""Verify: does the trained model actually use x_t and timestep t?"""
import torch
import torch.nn.functional as F
import yaml, sys
sys.path.insert(0, ".")
from models.dcno import DCNO
from scripts.dataset import build_dataset

device = "mps" if torch.backends.mps.is_available() else "cpu"
with open("configs/dcno_div2k_x4.yaml") as f:
    cfg = yaml.safe_load(f)

mcfg = cfg["model"]
model = DCNO(
    in_channels=mcfg.get("in_channels", 3), out_channels=mcfg.get("out_channels", 3),
    hidden_dims=mcfg.get("hidden_dims", [64, 128, 256, 512]),
    depths=mcfg.get("depths", [2, 2, 4, 2]),
    dct_block_size=mcfg.get("dct_block_size", 8), num_heads=mcfg.get("num_heads", 8),
    mode_weighting=mcfg.get("mode_weighting", True), scale_factor=mcfg.get("scale_factor", 4),
    dropout=0.0, transform_type=mcfg.get("transform_type", "dct"),
    freq_norm=mcfg.get("freq_norm", False), progressive_stem=mcfg.get("progressive_stem", False),
    concat_cond=mcfg.get("concat_cond", False),
    input_proj=mcfg.get("input_proj", False),
).to(device)
ckpt = torch.load("checkpoints/latest.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

ds = build_dataset("div2k", "./data", scale_factor=4, patch_size=192, augment=False, split="train", subset=0.125)
hr, lr = ds[0]
hr, lr = hr.unsqueeze(0).to(device), lr.unsqueeze(0).to(device)
lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)

with torch.no_grad():
    t = torch.tensor([0.5], device=device)
    noise = torch.randn_like(hr)
    x_t = 0.5 * hr + 0.5 * noise

    v1 = model(x_t, t, lr_up)

    # Different noise → different x_t, same image
    noise2 = torch.randn_like(hr)
    x_t2 = 0.5 * hr + 0.5 * noise2
    v2 = model(x_t2, t, lr_up)

    # Completely random x_t (unrelated to this image)
    x_t_rand = torch.randn_like(hr)
    v3 = model(x_t_rand, t, lr_up)

    # v1 norm as reference
    v_norm = v1.abs().mean().item()

    print("=== Does model use x_t? ===")
    print(f"  v(x_t) mean abs:             {v_norm:.6f}")
    print(f"  |v(x_t) - v(x_t2)|  mean:    {(v1 - v2).abs().mean():.6f}  ({(v1-v2).abs().mean()/v_norm*100:.1f}% of signal)")
    print(f"  |v(x_t) - v(random)| mean:   {(v1 - v3).abs().mean():.6f}  ({(v1-v3).abs().mean()/v_norm*100:.1f}% of signal)")

    print("\n=== Does model use timestep? ===")
    v_t01 = model(x_t, torch.tensor([0.01], device=device), lr_up)
    v_t05 = model(x_t, torch.tensor([0.5], device=device), lr_up)
    v_t09 = model(x_t, torch.tensor([0.99], device=device), lr_up)
    print(f"  |v(t=0.01) - v(t=0.50)| mean: {(v_t01 - v_t05).abs().mean():.6f}  ({(v_t01-v_t05).abs().mean()/v_norm*100:.1f}% of signal)")
    print(f"  |v(t=0.01) - v(t=0.99)| mean: {(v_t01 - v_t09).abs().mean():.6f}  ({(v_t01-v_t09).abs().mean()/v_norm*100:.1f}% of signal)")

    print("\n=== Does model use LR? ===")
    lr_rand = torch.randn_like(lr_up)
    v_no_lr = model(x_t, t, lr_rand)
    print(f"  |v(real_lr) - v(rand_lr)| mean: {(v1 - v_no_lr).abs().mean():.6f}  ({(v1-v_no_lr).abs().mean()/v_norm*100:.1f}% of signal)")

    # For eps-skip: eps_pred = model_output + x_t, target = noise
    pred_type = cfg.get("diffusion", {}).get("prediction_type", "velocity")
    if pred_type == "epsilon":
        print("\n=== Eps-skip analysis (eps_pred = output + x_t) ===")
        eps1 = v1 + x_t
        eps2 = v2 + x_t2
        eps3 = v3 + x_t_rand
        eps_rand_lr = v_no_lr + x_t
        # How close is eps_pred to actual noise?
        print(f"  MSE(eps_pred, noise):         {F.mse_loss(eps1, noise).item():.4f}  (1.0 = random, 0 = perfect)")
        print(f"  MSE(x_t, noise) [no model]:   {F.mse_loss(x_t, noise).item():.4f}  (skip-only baseline)")
        # Does eps_pred depend on which noise was used?
        eps_norm = eps1.abs().mean().item()
        print(f"  |eps(x_t) - eps(x_t2)| mean:  {(eps1 - eps2).abs().mean():.6f}  ({(eps1-eps2).abs().mean()/eps_norm*100:.1f}% of eps signal)")
        print(f"  |eps(x_t) - eps(rand)| mean:   {(eps1 - eps3).abs().mean():.6f}  ({(eps1-eps3).abs().mean()/eps_norm*100:.1f}% of eps signal)")

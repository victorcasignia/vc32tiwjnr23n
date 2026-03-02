"""
Rectified Flow Matching + Consistency Training for DCNO.

Rectified Flow:
  - Interpolation: x_t = (1 - t) * x_0 + t * epsilon
  - Velocity:      v   = epsilon - x_0
  - Model learns v_theta(x_t, t, cond) ≈ v
  - Sampling: solve ODE from t=1 (noise) to t=0 (clean) with Euler/midpoint

Consistency Training (optional, for low-step inference):
  - Enforces f_theta(x_t, t) = f_theta(x_{t'}, t') on same ODE trajectory
  - Allows 1-4 step inference without a teacher model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Optional, Tuple, Dict

from tqdm import tqdm

from models.dct import BlockDCT2d

log = logging.getLogger(__name__)


class RectifiedFlow:
    """
    Rectified Flow for conditional super-resolution.
    
    Training: add noise via linear interpolation, predict velocity.
    Inference: ODE integration from noise to clean image.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 10,
        prediction_type: str = "velocity",
        ode_solver: str = "euler",
        # --- LR-initialized sampling ---
        lr_init: bool = False,
        t_max: float = 0.8,
        # --- Frequency-weighted loss ---
        freq_weighted_loss: bool = False,
        freq_loss_alpha: float = 2.0,
        dct_block_size: int = 8,
        # --- Logit-normal timestep sampling ---
        timestep_sampling: str = "uniform",  # "uniform" or "logit_normal"
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        # --- Single-step inference (bypass ODE for x0 prediction) ---
        single_step_inference: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.prediction_type = prediction_type  # "velocity", "epsilon", or "x0"
        self.ode_solver = ode_solver

        # LR-init: start sampling from noised LR instead of pure noise
        self.lr_init = lr_init
        self.t_max = t_max

        # Single-step: for x0 prediction, one forward pass at t_max instead of ODE
        self.single_step_inference = single_step_inference

        # Logit-normal timestep sampling (SD3 style)
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std

        # Frequency-weighted loss: weight low-freq DCT errors higher
        self.freq_weighted_loss = freq_weighted_loss
        self._freq_loss_weights = None  # lazy-init on first call
        if freq_weighted_loss:
            n = dct_block_size
            self._dct = BlockDCT2d(n)
            # Precompute per-mode weights: w_ij = exp(-alpha * dist / max_dist)
            ii, jj = torch.meshgrid(
                torch.arange(n, dtype=torch.float32),
                torch.arange(n, dtype=torch.float32),
                indexing="ij",
            )
            dist = (ii.pow(2) + jj.pow(2)).sqrt()
            max_dist = math.sqrt(2) * (n - 1)
            raw_w = torch.exp(-freq_loss_alpha * dist / max_dist)  # DC≈1, high-freq≈0.2
            raw_w = raw_w / raw_w.mean()  # normalize so mean=1
            # Tile for 3 channels: (1, 3*n*n, 1, 1)
            self._freq_weights_1ch = raw_w.reshape(1, n * n, 1, 1)  # stored for lazy tiling

    def _get_freq_weights(self, C: int, device: torch.device) -> torch.Tensor:
        """Get frequency weights tiled for C channels, on the right device."""
        if self._freq_loss_weights is None or self._freq_loss_weights.shape[1] != C * self._freq_weights_1ch.shape[1]:
            self._freq_loss_weights = self._freq_weights_1ch.repeat(1, C, 1, 1).to(device)
        if self._freq_loss_weights.device != device:
            self._freq_loss_weights = self._freq_loss_weights.to(device)
        return self._freq_loss_weights

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps for training (uniform or logit-normal)."""
        if self.timestep_sampling == "logit_normal":
            # Logit-normal: t = sigmoid(mu + sigma * N(0,1))
            # Concentrates samples around t=0.5, avoids extremes
            u = torch.randn(batch_size, device=device)
            t = torch.sigmoid(self.logit_mean + self.logit_std * u)
            # Clamp to avoid exact 0/1
            t = t.clamp(1e-4, 1 - 1e-4)
            return t
        else:
            return torch.rand(batch_size, device=device)

    def add_noise(
        self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Linear interpolation: x_t = (1 - t) * x_0 + t * noise
        
        Args:
            x_0:   Clean HR image, (B, C, H, W)
            noise: Gaussian noise, same shape
            t:     Timesteps, (B,)
        
        Returns:
            x_t: Noisy image at time t
        """
        t = t[:, None, None, None]  # (B, 1, 1, 1)
        return (1 - t) * x_0 + t * noise

    def get_velocity(self, x_0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Ground-truth velocity: v = noise - x_0"""
        return noise - x_0

    def training_step(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_lr: torch.Tensor,
        x_bicubic: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        One training step:
          1. Sample t ~ U(0, 1)
          2. Sample noise ~ N(0, I)
          3. Compute x_t = (1-t)*x_0 + t*noise
          4. Predict v_theta = model(x_t, t, x_lr)
          5. Loss = ||v_theta - (noise - x_0)||^2

        If *x_bicubic* is provided (residual learning), the clean target
        becomes ``x_0 - x_bicubic`` so the diffusion process only needs to
        learn the residual detail.

        Returns dict with 'loss' and diagnostics.
        """
        # When residual learning is active, diffuse the residual instead
        target = x_0 if x_bicubic is None else (x_0 - x_bicubic)

        B = target.shape[0]
        device = target.device

        # Sample timesteps and noise
        t = self.sample_timesteps(B, device)
        noise = torch.randn_like(target)

        # Forward process
        x_t = self.add_noise(target, noise, t)

        # Model prediction
        pred = model(x_t, t, x_lr)

        # Compute loss and diagnostics based on prediction type
        if self.prediction_type == "x0":
            # x0 prediction: model directly predicts x_0 (or residual)
            pixel_loss = F.mse_loss(pred, target)

            with torch.no_grad():
                # Per-timestep-bin loss
                t_flat = t.detach()
                loss_per_sample = (pred - target).pow(2).mean(dim=[1, 2, 3])
                lo = (t_flat < 0.33)
                hi = (t_flat >= 0.67)
                loss_lo = loss_per_sample[lo].mean().item() if lo.any() else float('nan')
                loss_hi = loss_per_sample[hi].mean().item() if hi.any() else float('nan')

                # x0 improvement: how much better is pred than the trivial zero prediction?
                # For residual learning, zero is the baseline (=bicubic only).
                # For non-residual, baseline is the mean image.
                target_var = target.pow(2).mean(dim=[1, 2, 3]).clamp(min=1e-8)
                pred_mse = (pred - target).pow(2).mean(dim=[1, 2, 3])
                improvement = (1 - pred_mse / target_var).mean().item()
        else:
            # Velocity or epsilon (eps-skip) prediction
            v_target = self.get_velocity(target, noise)  # eps - x_0
            eps_skip_target = noise - x_t                  # (1-t) * v
            loss_target = eps_skip_target if self.prediction_type == "epsilon" else v_target
            pixel_loss = F.mse_loss(pred, loss_target)

            with torch.no_grad():
                t_flat = t.detach()
                loss_per_sample = (pred - loss_target).pow(2).mean(dim=[1, 2, 3])
                lo = (t_flat < 0.33)
                hi = (t_flat >= 0.67)
                loss_lo = loss_per_sample[lo].mean().item() if lo.any() else float('nan')
                loss_hi = loss_per_sample[hi].mean().item() if hi.any() else float('nan')

                if self.prediction_type == "epsilon":
                    eps_pred = pred + x_t
                    noise_mse = (eps_pred - noise).pow(2).mean(dim=[1, 2, 3])
                    baseline_mse = (x_t - noise).pow(2).mean(dim=[1, 2, 3]).clamp(min=1e-8)
                    improvement = (1 - noise_mse / baseline_mse).mean().item()
                else:
                    improvement = 0.0

        # Frequency-weighted loss (optional, works with any prediction type)
        loss_target_for_freq = target if self.prediction_type == "x0" else loss_target
        if self.freq_weighted_loss:
            C = target.shape[1]
            self._dct.dct_mat = self._dct.dct_mat.to(device)
            pred_spec = self._dct(pred)
            tgt_spec = self._dct(loss_target_for_freq)
            w = self._get_freq_weights(C, device)
            freq_loss = (w * (pred_spec - tgt_spec).pow(2)).mean()
            loss = 0.5 * pixel_loss + 0.5 * freq_loss
        else:
            loss = pixel_loss

        return {
            "loss": loss,
            "noise_improve": improvement,
            "loss_lo_t": loss_lo,
            "loss_hi_t": loss_hi,
            "v_pred_norm": pred.detach().norm().item(),
            "v_target_norm": target.detach().norm().item() if self.prediction_type == "x0" else loss_target.detach().norm().item(),
        }

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x_lr: torch.Tensor,
        shape: Tuple[int, ...],
        num_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
        show_progress: bool = False,
        x_bicubic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate HR image from noise via ODE integration.
        
        Args:
            model:         DCNO model
            x_lr:          LR conditioning image
            shape:         Target HR shape (B, C, H, W)
            num_steps:     Override number of ODE steps
            device:        Device
            show_progress: Show tqdm bar for ODE steps
            x_bicubic:     Bicubic-upsampled LR (required when using
                           residual learning — added to ODE output)
        
        Returns:
            x_0: Generated HR image
        """
        num_steps = num_steps or self.num_inference_steps
        device = device or x_lr.device

        # Starting point for ODE
        if self.lr_init:
            # LR-initialized: start from noised LR upsampled, covering [t_max → 0]
            lr_up = F.interpolate(
                x_lr, size=shape[-2:], mode="bilinear", align_corners=False
            )
            noise = torch.randn(shape, device=device)
            if x_bicubic is not None:
                # Residual mode: target is near-zero, so init = t_max * noise
                x = self.t_max * noise
            else:
                # Full-image mode: x = (1 - t_max) * lr_up + t_max * noise
                x = (1 - self.t_max) * lr_up + self.t_max * noise
            t_start = self.t_max
        else:
            # Standard: start from pure noise at t=1
            x = torch.randn(shape, device=device)
            t_start = 1.0

        # --- Single-step direct prediction (bypass ODE for x0-prediction) ---
        if self.prediction_type == "x0" and self.single_step_inference:
            t = torch.full((shape[0],), t_start, device=device)
            x = model(x, t, x_lr)  # model directly predicts x_0
            if x_bicubic is not None:
                x = x + x_bicubic
            return x.clamp(0, 1)

        # Time steps from t_start → 0
        dt = t_start / num_steps
        timesteps = torch.linspace(t_start, dt, num_steps, device=device)

        if self.ode_solver == "euler":
            x = self._euler_solve(model, x, x_lr, timesteps, dt, show_progress)
        elif self.ode_solver == "midpoint":
            x = self._midpoint_solve(model, x, x_lr, timesteps, dt, show_progress)
        elif self.ode_solver == "adaptive":
            x = self._adaptive_solve(model, x, x_lr, num_steps, show_progress)
        else:
            raise ValueError(f"Unknown ODE solver: {self.ode_solver}")

        # Residual learning: ODE produced the residual, add bicubic back
        if x_bicubic is not None:
            x = x + x_bicubic

        return x.clamp(0, 1)

    def _output_to_step(self, output, x, t_val, dt):
        """
        Convert model output to next x based on prediction_type.
        
        For velocity: standard Euler step x -= dt * v
        For epsilon: DDIM-style step via x_0/eps reparameterization
        For x0: model directly predicts x_0, step via DDIM
        """
        s = t_val - dt  # next timestep

        if self.prediction_type == "x0":
            # Model directly predicts x_0
            x_0_pred = output
            if s <= 1e-6:
                return x_0_pred
            # Infer noise from x_0_pred: eps = (x_t - (1-t)*x_0) / t
            if t_val > 1e-3:
                eps_inferred = (x - (1.0 - t_val) * x_0_pred) / t_val
            else:
                eps_inferred = torch.zeros_like(x)
            return (1 - s) * x_0_pred + s * eps_inferred

        elif self.prediction_type == "epsilon":
            # Model predicted correction; eps_pred = output + x_t
            eps_pred = output + x
            one_minus_t = 1.0 - t_val
            if one_minus_t < 1e-3:
                x_0_pred = torch.zeros_like(x)
            else:
                x_0_pred = (x - t_val * eps_pred) / one_minus_t
            if s <= 1e-6:
                return x_0_pred
            return (1 - s) * x_0_pred + s * eps_pred

        else:
            # Velocity prediction: Euler step
            return x - dt * output

    def _euler_solve(
        self, model, x, x_lr, timesteps, dt, show_progress=False,
    ) -> torch.Tensor:
        """Euler method: x_{t-dt} = x_t - dt * v(x_t, t)"""
        steps = tqdm(timesteps, desc="ODE (euler)", leave=False) if show_progress else timesteps
        for t_val in steps:
            t = torch.full((x.shape[0],), t_val.item(), device=x.device)
            output = model(x, t, x_lr)
            x = self._output_to_step(output, x, t_val.item(), dt)
        return x

    def _midpoint_solve(
        self, model, x, x_lr, timesteps, dt, show_progress=False,
    ) -> torch.Tensor:
        """Midpoint method: higher-order ODE solve."""
        steps = tqdm(timesteps, desc="ODE (midpoint)", leave=False) if show_progress else timesteps
        for t_val in steps:
            t = torch.full((x.shape[0],), t_val.item(), device=x.device)
            # Evaluate at current point
            output1 = model(x, t, x_lr)
            # Midpoint estimate
            x_mid = self._output_to_step(output1, x, t_val.item(), 0.5 * dt)
            t_mid = torch.full_like(t, max(t_val.item() - 0.5 * dt, 0))
            output2 = model(x_mid, t_mid, x_lr)
            # Full step with midpoint velocity
            x = self._output_to_step(output2, x, t_val.item(), dt)
        return x

    def _adaptive_solve(
        self, model, x, x_lr, num_steps, show_progress=False,
    ) -> torch.Tensor:
        """
        Adaptive time-step ODE solver (inspired by DiffFNO's ATS).
        Uses larger steps where velocity is smooth, smaller steps near t=0.
        """
        # Cosine schedule: more steps near t=0 where detail emerges
        indices = torch.arange(num_steps + 1, device=x.device, dtype=torch.float32)
        schedule = torch.cos(indices / num_steps * math.pi / 2)  # 1 → 0
        timesteps = schedule[:-1]
        next_timesteps = schedule[1:]

        pairs = list(zip(timesteps, next_timesteps))
        steps = tqdm(pairs, desc="ODE (adaptive)", leave=False) if show_progress else pairs
        for t_val, t_next in steps:
            step_dt = (t_val - t_next).item()
            t = torch.full((x.shape[0],), t_val.item(), device=x.device)
            output = model(x, t, x_lr)
            x = self._output_to_step(output, x, t_val.item(), step_dt)
        return x


class ConsistencyTrainer:
    """
    Self-consistency training for few/one-step inference.
    
    Core idea: for any two points x_t and x_{t'} on the same ODE trajectory,
    the mapping f_theta should produce the same output:
        f_theta(x_t, t) ≈ f_theta(x_{t'}, t')
    
    This is trained without a teacher model by using the EMA of the model
    itself as the target network.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        target_steps: int = 1,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.target_steps = target_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _get_adjacent_timesteps(
        self, batch_size: int, device: torch.device, num_boundaries: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample adjacent timestep pairs (t, t') for consistency loss."""
        # Discretize [0, 1] into num_boundaries segments
        boundaries = torch.linspace(0, 1, num_boundaries + 1, device=device)
        
        # Sample a random boundary index for each sample
        idx = torch.randint(0, num_boundaries, (batch_size,), device=device)
        t = boundaries[idx + 1]  # larger t
        t_next = boundaries[idx]  # smaller t (closer to clean)
        
        return t, t_next

    def training_step(
        self,
        model: nn.Module,
        ema_model: nn.Module,
        x_0: torch.Tensor,
        x_lr: torch.Tensor,
        num_boundaries: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Consistency training step.
        
        1. Sample (t, t') adjacent pair
        2. Create x_t from x_0
        3. One-step ODE from x_t → x_{t'} using EMA model
        4. Loss = ||f_theta(x_t, t) - f_ema(x_{t'}, t')||^2
        """
        B = x_0.shape[0]
        device = x_0.device

        t, t_next = self._get_adjacent_timesteps(B, device, num_boundaries)
        noise = torch.randn_like(x_0)

        # Create noisy sample at t
        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * x_0 + t_expand * noise

        # One-step ODE to get x_{t'} using EMA model (no grad)
        dt = t - t_next
        with torch.no_grad():
            v_ema = ema_model(x_t, t, x_lr)
            x_t_next = x_t - dt[:, None, None, None] * v_ema

        # Student prediction at x_t
        pred_at_t = model(x_t, t, x_lr)

        # Target: EMA prediction at x_{t'} (detached)
        with torch.no_grad():
            pred_at_t_next = ema_model(x_t_next, t_next, x_lr)

        # Pseudo-Huber loss (smoother than L2 for consistency)
        c = 0.00054 * math.sqrt(x_0.numel() / B)  # from consistency models paper
        diff = pred_at_t - pred_at_t_next.detach()
        loss = torch.sqrt(diff.pow(2) + c * c).mean() - c

        return {
            "loss": loss,
            "consistency_gap": diff.detach().abs().mean().item(),
        }

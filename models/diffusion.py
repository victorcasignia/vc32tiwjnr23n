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
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.prediction_type = prediction_type
        self.ode_solver = ode_solver

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample uniform timesteps in [0, 1] for training."""
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
    ) -> Dict[str, torch.Tensor]:
        """
        One training step:
          1. Sample t ~ U(0, 1)
          2. Sample noise ~ N(0, I)
          3. Compute x_t = (1-t)*x_0 + t*noise
          4. Predict v_theta = model(x_t, t, x_lr)
          5. Loss = ||v_theta - (noise - x_0)||^2
        
        Returns dict with 'loss' and diagnostics.
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample timesteps and noise
        t = self.sample_timesteps(B, device)
        noise = torch.randn_like(x_0)

        # Forward process
        x_t = self.add_noise(x_0, noise, t)
        v_target = self.get_velocity(x_0, noise)

        # Model prediction
        v_pred = model(x_t, t, x_lr)

        # Velocity loss (MSE)
        loss = F.mse_loss(v_pred, v_target)

        return {
            "loss": loss,
            "v_pred_norm": v_pred.detach().norm().item(),
            "v_target_norm": v_target.detach().norm().item(),
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
        
        Returns:
            x_0: Generated HR image
        """
        num_steps = num_steps or self.num_inference_steps
        device = device or x_lr.device

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Time steps from 1 → 0
        dt = 1.0 / num_steps
        timesteps = torch.linspace(1.0, dt, num_steps, device=device)

        if self.ode_solver == "euler":
            x = self._euler_solve(model, x, x_lr, timesteps, dt, show_progress)
        elif self.ode_solver == "midpoint":
            x = self._midpoint_solve(model, x, x_lr, timesteps, dt, show_progress)
        elif self.ode_solver == "adaptive":
            x = self._adaptive_solve(model, x, x_lr, num_steps, show_progress)
        else:
            raise ValueError(f"Unknown ODE solver: {self.ode_solver}")

        return x.clamp(0, 1)

    def _euler_solve(
        self, model, x, x_lr, timesteps, dt, show_progress=False,
    ) -> torch.Tensor:
        """Euler method: x_{t-dt} = x_t - dt * v(x_t, t)"""
        steps = tqdm(timesteps, desc="ODE (euler)", leave=False) if show_progress else timesteps
        for t_val in steps:
            t = torch.full((x.shape[0],), t_val.item(), device=x.device)
            v = model(x, t, x_lr)
            x = x - dt * v
        return x

    def _midpoint_solve(
        self, model, x, x_lr, timesteps, dt, show_progress=False,
    ) -> torch.Tensor:
        """Midpoint method: higher-order ODE solve."""
        steps = tqdm(timesteps, desc="ODE (midpoint)", leave=False) if show_progress else timesteps
        for t_val in steps:
            t = torch.full((x.shape[0],), t_val.item(), device=x.device)
            # Evaluate at current point
            v1 = model(x, t, x_lr)
            # Midpoint estimate
            x_mid = x - 0.5 * dt * v1
            t_mid = torch.full_like(t, t_val.item() - 0.5 * dt)
            t_mid = t_mid.clamp(min=0)
            v2 = model(x_mid, t_mid, x_lr)
            # Full step with midpoint velocity
            x = x - dt * v2
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
            dt = t_val - t_next
            t = torch.full((x.shape[0],), t_val.item(), device=x.device)
            v = model(x, t, x_lr)
            x = x - dt * v
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

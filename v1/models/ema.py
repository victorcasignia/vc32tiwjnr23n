"""
Exponential Moving Average (EMA) for model parameters.

Used both for stable evaluation and as the target network
in consistency training.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class EMA:
    """
    Maintains an exponential moving average of model parameters.
    
    Usage:
        ema = EMA(model, decay=0.9999)
        # After each optimizer step:
        ema.update()
        # For evaluation:
        with ema.average_parameters():
            evaluate(model)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0,
        update_after_step: int = 0,
    ):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.update_after_step = update_after_step
        self.step_count = 0

        # Shadow copy of parameters
        self.shadow = {}
        self.backup = {}
        self._build_shadow()

    def _build_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _get_decay(self) -> float:
        step = self.step_count - self.update_after_step
        if step <= 0:
            return 0.0
        if step < self.warmup_steps:
            # Linear warmup of decay
            return min(self.decay, 1 - 1 / (step + 1))
        return self.decay

    @torch.no_grad()
    def update(self):
        """Update shadow parameters with current model parameters."""
        self.step_count += 1
        decay = self._get_decay()

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def apply_shadow(self):
        """Replace model parameters with shadow (EMA) parameters."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model parameters from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    class _AverageContext:
        def __init__(self, ema):
            self.ema = ema

        def __enter__(self):
            self.ema.apply_shadow()
            return self.ema.model

        def __exit__(self, *args):
            self.ema.restore()

    def average_parameters(self):
        """Context manager that temporarily applies EMA parameters."""
        return self._AverageContext(self)

    def state_dict(self):
        return {
            "shadow": self.shadow,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.step_count = state_dict["step_count"]

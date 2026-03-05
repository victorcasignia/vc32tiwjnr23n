"""
MuSGD (Muon-SGD) hybrid optimizer.

- 2D+ parameters (conv weights, linear projections): Muon optimizer with
  Newton-Schulz orthogonalization of the gradient direction.
- 1D parameters (biases, norms): standard SGD with momentum.

Adapted from the Muon optimizer (Jordan et al.) with Nesterov momentum.
"""

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Maps G → U where U^T U ≈ I, preserving the "direction" of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if X.size(0) > X.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class MuSGD(Optimizer):
    """
    Muon-SGD hybrid optimizer.

    Args:
        params:          Iterable of parameters
        lr:              Learning rate
        momentum:        SGD momentum (for 1D params)
        muon_momentum:   Muon momentum (for 2D+ params)
        weight_decay:    Weight decay coefficient
        nesterov:        Use Nesterov momentum
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        muon_momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            muon_momentum=muon_momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            momentum     = group["momentum"]
            muon_momentum = group["muon_momentum"]
            weight_decay = group["weight_decay"]
            nesterov     = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad  = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # Decoupled weight decay
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # 2D+ parameters → Muon (orthogonalized gradient)
                if len(p.shape) >= 2:
                    orig_shape = p.shape
                    if len(p.shape) > 2:
                        grad    = grad.view(grad.size(0), -1)
                        buf_view = buf.view(buf.size(0), -1)
                    else:
                        buf_view = buf

                    buf_view.mul_(muon_momentum).add_(grad)
                    g = (grad + muon_momentum * buf_view) if nesterov else buf_view

                    update = zeropower_via_newtonschulz5(g)
                    scale  = max(update.size(0), update.size(1)) ** 0.5
                    update.mul_(scale * 0.2)

                    if len(orig_shape) > 2:
                        update = update.view(orig_shape)

                    p.data.add_(update, alpha=-lr)

                # 1D parameters → SGD with momentum
                else:
                    buf.mul_(momentum).add_(grad)
                    update = (grad + momentum * buf) if nesterov else buf
                    p.data.add_(update, alpha=-lr)

        return loss

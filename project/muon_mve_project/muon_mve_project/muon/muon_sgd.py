
import math
from typing import Iterable, Optional, Callable, Any, Dict

import torch
from torch.optim import Optimizer

from .inner_solvers import BaseInnerSolver, SpectralClipSolver


class MuonSGD(Optimizer):
    """
    A very simple SGD-like optimizer that uses an inner solver to enforce
    a spectral-norm budget on matrix-shaped parameters.

    This is *not* a faithful reimplementation of manifold Muon, but it gives you
    a clean place to plug in more sophisticated inner solvers later.

    Design:
      - We maintain momentum and weight decay like standard SGD.
      - For 2D tensors (e.g. Linear weights, Conv kernels flattened), we:
          1. Form the effective gradient (including weight decay, momentum).
          2. Take a raw step: delta = -lr * grad_eff.
          3. Pass (W, delta) through an inner solver that enforces ||delta||_2 <= spectral_budget.
          4. Apply the modified delta to W.
      - For non-2D tensors (biases, LayerNorm scales, etc.) we just do vanilla SGD.

    Arguments:
        params: iterable of parameters to optimize.
        lr: learning rate.
        momentum: momentum factor (0 for no momentum).
        weight_decay: L2 weight decay.
        spectral_budget: maximum allowed spectral norm of delta for 2D params.
        inner_solver: an instance of BaseInnerSolver. If None, SpectralClipSolver is used.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        spectral_budget: Optional[float] = None,
        inner_solver: Optional[BaseInnerSolver] = None,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults: Dict[str, Any] = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            spectral_budget=spectral_budget,
        )
        super().__init__(params, defaults)

        if inner_solver is None:
            inner_solver = SpectralClipSolver()
        self.inner_solver = inner_solver

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            loss if a closure was provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            spectral_budget = group.get("spectral_budget", None)

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                if weight_decay != 0.0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                if momentum > 0.0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                # Proposed raw update
                delta = -lr * d_p

                # If it's a matrix-shaped parameter and we have a budget, call the inner solver.
                if p.data.ndim == 2 and spectral_budget is not None:
                    delta = self.inner_solver(p.data, delta, spectral_budget)

                # Apply update
                p.add_(delta)

        return loss

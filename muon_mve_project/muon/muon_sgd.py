"""
MuonSGD: SGD-like optimizer with spectral-norm constraints on matrix updates.

This optimizer maintains the semantics of SGD (momentum, weight decay) while
enforcing a spectral-norm budget on updates to 2D parameters. The key property
is that the step size η directly bounds how much ||Wx|| can change for any unit x.

Design:
    1. Compute effective gradient (including weight decay, momentum).
    2. Form raw update: Δ = -lr * g_eff.
    3. For 2D params, pass (W, Δ) to an inner solver that enforces ||Δ||_2 <= budget.
    4. Apply the modified Δ to W.

This is a simplified version of manifold Muon. For full manifold constraints
(Stiefel, tangent space projections), use the TangentSpaceProjector from
inner_solvers.py and apply retractions.

Reference:
    - Bernstein's "Modular Manifolds" blog post
    - Jordan's original Muon implementation
"""

import math
from typing import Iterable, Optional, Callable, Any, Dict, List

import torch
from torch.optim import Optimizer

from .inner_solvers import BaseInnerSolver, SpectralClipSolver, get_inner_solver


class MuonSGD(Optimizer):
    """
    SGD optimizer with spectral-norm-constrained updates for matrix parameters.

    For 2D parameters (e.g., Linear weights, reshaped Conv kernels):
        - Computes raw update Δ = -lr * (grad + weight_decay * W)
        - Applies momentum: Δ = momentum * Δ_prev + Δ
        - Passes through inner solver: Δ' = solver(W, Δ, budget)
        - Updates: W <- W + Δ'

    For 1D parameters (biases, norms): vanilla SGD.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 0.1).
        momentum: Momentum factor (default: 0.9).
        weight_decay: L2 weight decay (default: 0.0).
        spectral_budget: Max spectral norm of update for 2D params (default: None).
        inner_solver: Instance of BaseInnerSolver, or None for SpectralClipSolver.
        nesterov: Use Nesterov momentum (default: False).
        dampening: Dampening for momentum (default: 0.0).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        spectral_budget: Optional[float] = None,
        inner_solver: Optional[BaseInnerSolver] = None,
        nesterov: bool = False,
        dampening: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening = 0")

        defaults: Dict[str, Any] = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            spectral_budget=spectral_budget,
            nesterov=nesterov,
            dampening=dampening,
        )
        super().__init__(params, defaults)

        self.inner_solver = inner_solver if inner_solver is not None else SpectralClipSolver()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            spectral_budget = group.get("spectral_budget")
            nesterov = group.get("nesterov", False)
            dampening = group.get("dampening", 0.0)

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                # Weight decay (L2 regularization)
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                # Momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # Raw update
                delta = -lr * d_p

                # Apply inner solver to 2D parameters
                if p.data.ndim == 2 and spectral_budget is not None:
                    delta = self.inner_solver(p.data, delta, spectral_budget)

                # Apply update
                p.add_(delta)

        return loss

    def reset_inner_solver(self) -> None:
        """Reset the inner solver's state (e.g., warm-start caches)."""
        self.inner_solver.reset()


class MuonAdamW(Optimizer):
    """
    AdamW optimizer with spectral-norm-constrained updates for matrix parameters.

    Combines Adam's adaptive learning rates with spectral-norm constraints.
    Useful for transformers where AdamW is the standard baseline.

    Args:
        params: Iterable of parameters.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages (default: (0.9, 0.999)).
        eps: Term for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay (default: 0.01).
        spectral_budget: Max spectral norm of update for 2D params.
        inner_solver: Instance of BaseInnerSolver.
        amsgrad: Use AMSGrad variant (default: False).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        spectral_budget: Optional[float] = None,
        inner_solver: Optional[BaseInnerSolver] = None,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            spectral_budget=spectral_budget,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

        self.inner_solver = inner_solver if inner_solver is not None else SpectralClipSolver()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            spectral_budget = group.get("spectral_budget")
            amsgrad = group.get("amsgrad", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("MuonAdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                state["step"] += 1

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Raw update direction
                delta = -step_size * exp_avg / denom

                # Apply inner solver to 2D parameters
                if p.data.ndim == 2 and spectral_budget is not None:
                    delta = self.inner_solver(p.data, delta, spectral_budget)

                p.add_(delta)

        return loss

    def reset_inner_solver(self) -> None:
        """Reset inner solver state."""
        self.inner_solver.reset()


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "muon_sgd",
    inner_solver_type: str = "spectral_clip",
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    spectral_budget: Optional[float] = 0.1,
    **kwargs,
) -> Optimizer:
    """
    Factory function to create an optimizer with configurable inner solver.

    Args:
        model: The neural network.
        optimizer_type: One of 'sgd', 'adamw', 'muon_sgd', 'muon_adamw'.
        inner_solver_type: One of 'spectral_clip', 'frank_wolfe', 'dual_ascent',
                           'quasi_newton', 'admm', or 'none'.
        lr: Learning rate.
        momentum: Momentum (for SGD variants).
        weight_decay: Weight decay.
        spectral_budget: Spectral norm budget (None to disable).
        **kwargs: Additional arguments for optimizer/solver.

    Returns:
        Configured optimizer instance.
    """
    # Get inner solver
    if inner_solver_type == "none" or spectral_budget is None:
        inner_solver = None
        spectral_budget = None
    else:
        solver_kwargs = {k: v for k, v in kwargs.items() if k.startswith("solver_")}
        solver_kwargs = {k[7:]: v for k, v in solver_kwargs.items()}  # Remove prefix
        inner_solver = get_inner_solver(inner_solver_type, **solver_kwargs)

    if optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )
    elif optimizer_type == "muon_sgd":
        return MuonSGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            spectral_budget=spectral_budget,
            inner_solver=inner_solver,
            nesterov=kwargs.get("nesterov", False),
        )
    elif optimizer_type == "muon_adamw":
        return MuonAdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            spectral_budget=spectral_budget,
            inner_solver=inner_solver,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

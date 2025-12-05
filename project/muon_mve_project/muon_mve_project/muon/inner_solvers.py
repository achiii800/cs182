
import torch
from abc import ABC, abstractmethod


class BaseInnerSolver(ABC):
    """
    Abstract base class for "Muon-like" inner solvers.

    The job of an inner solver is to take a proposed raw update (e.g. -lr * grad)
    and modify it so that it satisfies some spectral-norm constraint and/or
    geometric constraint (e.g. low-rank structure).

    We intentionally keep the interface simple so you can later swap in a
    faithful implementation of the manifold Muon inner problem.
    """

    @abstractmethod
    def __call__(self, W: torch.Tensor, delta: torch.Tensor, spectral_budget: float) -> torch.Tensor:
        """
        Args:
            W: weight matrix of shape (out_dim, in_dim) or any 2D tensor.
            delta: proposed raw update of same shape as W.
            spectral_budget: maximum allowed spectral norm ||delta||_2.

        Returns:
            A modified update tensor of same shape, respecting the budget.
        """
        raise NotImplementedError


class SpectralClipSolver(BaseInnerSolver):
    """
    Extremely simple baseline:
    - Look at the raw update delta.
    - If its spectral norm exceeds the budget, rescale it down.
    - Otherwise, return it unchanged.

    This is *not* full Muon, but it already enforces a well-defined Lipschitz
    bound on the weight change: ||delta||_2 <= spectral_budget.
    """

    def __call__(self, W: torch.Tensor, delta: torch.Tensor, spectral_budget: float) -> torch.Tensor:
        if spectral_budget is None or spectral_budget <= 0.0:
            return delta

        if delta.ndim != 2:
            # For non-matrix parameters (e.g. biases), fall back to plain update.
            return delta

        # Compute spectral norm of the update using matrix_norm with ord=2.
        # For small-ish layers this is fine; for big layers you might want power iteration.
        with torch.no_grad():
            try:
                sigma = torch.linalg.matrix_norm(delta, ord=2)
            except RuntimeError:
                # Older PyTorch fallback: use largest singular value
                s = torch.linalg.svdvals(delta)
                sigma = s.max()

            if sigma <= spectral_budget or sigma == 0:
                return delta

            scale = spectral_budget / sigma
            return delta * scale


class FrankWolfeSolver(BaseInnerSolver):
    """
    Very simple Frank–Wolfe-style inner solver for a spectral-norm ball.

    Intuition:
      - The inner problem (in the exact Muon derivation) is roughly:
            min_{||Δ||_2 <= budget} <grad, Δ>
        which has solution Δ* = -budget * u v^T, where u, v are the top singular
        vectors of the gradient.

      - Here we only have access to the *proposed* delta, and we want a low-rank
        update that respects the spectral budget and roughly points in the same
        direction.

    Implementation:
      - Compute top singular vectors of delta.
      - Return a rank-1 update with spectral norm = spectral_budget.
      - Optionally blend with the original delta (currently off by default).
    """

    def __init__(self, blend_with_raw: float = 0.0):
        """
        Args:
            blend_with_raw: float in [0, 1]. If >0, we return
                blend_with_raw * delta + (1 - blend_with_raw) * delta_fw
            where delta_fw is the Frank–Wolfe-style rank-1 update.
        """
        super().__init__()
        self.blend_with_raw = blend_with_raw

    def __call__(self, W: torch.Tensor, delta: torch.Tensor, spectral_budget: float) -> torch.Tensor:
        if spectral_budget is None or spectral_budget <= 0.0:
            return delta

        if delta.ndim != 2:
            # Only makes sense for matrices.
            return delta

        with torch.no_grad():
            # Compute top singular vectors via SVD.
            try:
                # full_matrices=False is cheaper and sufficient
                U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            except RuntimeError:
                # If SVD fails for some reason, fall back to plain delta.
                return delta

            u = U[:, 0:1]        # (m, 1)
            v = Vh[0:1, :]       # (1, n)
            # Construct rank-1 update with spectral norm = spectral_budget
            delta_fw = -spectral_budget * (u @ v)

            if self.blend_with_raw <= 0.0:
                return delta_fw
            else:
                return self.blend_with_raw * delta + (1.0 - self.blend_with_raw) * delta_fw

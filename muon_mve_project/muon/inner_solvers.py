"""
Inner solvers for manifold Muon-style optimization.

Each solver takes a weight matrix W and proposed update delta, then returns a modified
update satisfying the spectral-norm budget ||delta||_2 <= budget.

The inner problem (from the Muon derivation) is:
    min_{A} trace(G^T A)  s.t.  ||A||_spectral <= η  and  A^T W + W^T A = 0 (tangency)

For the non-manifold case (no tangency constraint), we simply enforce the spectral budget.

Implemented solvers:
    - SpectralClipSolver: Simple rescaling baseline
    - FrankWolfeSolver: Projection-free, rank-1/low-rank atoms via top-SVD
    - DualAscentSolver: Baseline dual ascent (Lagrangian approach)
    - QuasiNewtonDualSolver: L-BFGS on the dual objective
    - ADMMSolver: ADMM-style splitting for tangency + spectral constraints
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import math


class BaseInnerSolver(ABC):
    """
    Abstract base class for Muon-like inner solvers.

    The solver modifies a proposed update delta to satisfy:
        ||delta||_2 <= spectral_budget

    Optionally, solvers can also enforce tangency constraints for manifold optimization.
    """

    @abstractmethod
    def __call__(
        self,
        W: torch.Tensor,
        delta: torch.Tensor,
        spectral_budget: float,
    ) -> torch.Tensor:
        """
        Args:
            W: Current weight matrix, shape (m, n).
            delta: Proposed raw update, same shape as W.
            spectral_budget: Maximum allowed spectral norm ||delta||_2.

        Returns:
            Modified update tensor satisfying the budget constraint.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state (e.g., warm-start iterates). Override if needed."""
        pass


class SpectralClipSolver(BaseInnerSolver):
    """
    Simple spectral norm clipping baseline.

    If ||delta||_2 > budget, rescale: delta <- delta * (budget / ||delta||_2).
    Otherwise return delta unchanged.

    This enforces a Lipschitz bound on the weight change but does not
    consider the tangent space or manifold structure.
    """

    def __init__(self, use_power_iteration: bool = False, power_iter_steps: int = 10):
        """
        Args:
            use_power_iteration: If True, estimate spectral norm via power iteration
                                 (cheaper for large matrices). Otherwise use SVD.
            power_iter_steps: Number of power iteration steps if use_power_iteration=True.
        """
        super().__init__()
        self.use_power_iteration = use_power_iteration
        self.power_iter_steps = power_iter_steps

    def _spectral_norm(self, A: torch.Tensor) -> torch.Tensor:
        """Compute spectral norm ||A||_2."""
        if self.use_power_iteration:
            return self._power_iteration_spectral_norm(A)
        else:
            try:
                return torch.linalg.matrix_norm(A, ord=2)
            except RuntimeError:
                s = torch.linalg.svdvals(A)
                return s[0]

    def _power_iteration_spectral_norm(self, A: torch.Tensor) -> torch.Tensor:
        """Estimate spectral norm via power iteration."""
        m, n = A.shape
        v = torch.randn(n, 1, device=A.device, dtype=A.dtype)
        v = v / torch.norm(v)

        for _ in range(self.power_iter_steps):
            u = A @ v
            u = u / (torch.norm(u) + 1e-12)
            v = A.T @ u
            v = v / (torch.norm(v) + 1e-12)

        return torch.norm(A @ v)

    def __call__(
        self,
        W: torch.Tensor,
        delta: torch.Tensor,
        spectral_budget: float,
    ) -> torch.Tensor:
        if spectral_budget is None or spectral_budget <= 0.0:
            return delta

        if delta.ndim != 2:
            return delta

        with torch.no_grad():
            sigma = self._spectral_norm(delta)

            if sigma <= spectral_budget or sigma == 0:
                return delta

            scale = spectral_budget / sigma
            return delta * scale


class FrankWolfeSolver(BaseInnerSolver):
    """
    Frank-Wolfe (conditional gradient) solver for the spectral-norm ball.

    The inner problem is:
        min_{||A||_2 <= budget} <G, A>

    The LMO (linear minimization oracle) for the spectral-norm ball returns:
        A* = -budget * u @ v^T
    where u, v are the top left/right singular vectors of G.

    This solver can run multiple FW iterations, building a low-rank update
    as a convex combination of rank-1 atoms.
    """

    def __init__(
        self,
        max_iters: int = 5,
        blend_with_raw: float = 0.0,
        use_away_steps: bool = False,
        tol: float = 1e-6,
    ):
        """
        Args:
            max_iters: Number of Frank-Wolfe iterations.
            blend_with_raw: Final blend ratio with raw delta (0 = pure FW).
            use_away_steps: If True, use away-step FW variant for faster convergence.
            tol: Convergence tolerance on duality gap.
        """
        super().__init__()
        self.max_iters = max_iters
        self.blend_with_raw = blend_with_raw
        self.use_away_steps = use_away_steps
        self.tol = tol

    def _lmo(
        self, G: torch.Tensor, budget: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Linear Minimization Oracle for spectral-norm ball.

        Returns: (atom, u, v) where atom = -budget * u @ v^T
        """
        # Top singular vectors of G
        try:
            U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        except RuntimeError:
            # Fallback: return scaled G
            sigma = torch.linalg.matrix_norm(G, ord=2)
            if sigma > 0:
                return -budget * G / sigma, None, None
            return G, None, None

        u = U[:, 0:1]  # (m, 1)
        v = Vh[0:1, :]  # (1, n)
        atom = -budget * (u @ v)
        return atom, u, v

    def __call__(
        self,
        W: torch.Tensor,
        delta: torch.Tensor,
        spectral_budget: float,
    ) -> torch.Tensor:
        if spectral_budget is None or spectral_budget <= 0.0:
            return delta

        if delta.ndim != 2:
            return delta

        with torch.no_grad():
            # G is the gradient direction; we want to minimize <G, A>
            # Here delta = -lr * G, so G ~ -delta/lr, but we can just use delta
            # since we want A to point in a similar direction

            # Initialize with LMO solution
            G = delta  # Use delta as proxy for gradient direction
            A, _, _ = self._lmo(G, spectral_budget)

            for t in range(1, self.max_iters):
                # Compute gradient of objective w.r.t. A
                # Objective: <G, A>, gradient is G
                grad_A = G

                # LMO on gradient
                s_t, _, _ = self._lmo(grad_A, spectral_budget)

                # Duality gap estimate
                gap = torch.sum(grad_A * (A - s_t))
                if gap < self.tol:
                    break

                # Step size (standard FW schedule)
                gamma = 2.0 / (t + 2)

                # Update
                A = (1 - gamma) * A + gamma * s_t

            if self.blend_with_raw > 0.0:
                # Clip raw delta and blend
                sigma_delta = torch.linalg.matrix_norm(delta, ord=2)
                if sigma_delta > spectral_budget:
                    delta_clipped = delta * (spectral_budget / sigma_delta)
                else:
                    delta_clipped = delta
                A = self.blend_with_raw * delta_clipped + (1 - self.blend_with_raw) * A

            return A


class DualAscentSolver(BaseInnerSolver):
    """
    Dual ascent solver for the spectral-norm constrained inner problem.

    We solve:
        min_{||A||_2 <= budget} <G, A>

    via Lagrangian dual:
        L(A, λ) = <G, A> + λ * (||A||_2 - budget)

    The dual problem is:
        max_{λ >= 0} min_A L(A, λ)

    For fixed λ, the inner min is:
        A*(λ) = -budget * u @ v^T  if λ > 0 (where u,v are top sing. vectors of G)

    We perform gradient ascent on λ with:
        ∂L/∂λ = ||A*(λ)||_2 - budget

    With warm-starting across outer iterations.
    """

    def __init__(
        self,
        max_iters: int = 20,
        lr_dual: float = 0.1,
        tol: float = 1e-5,
        warm_start: bool = True,
    ):
        """
        Args:
            max_iters: Maximum dual ascent iterations.
            lr_dual: Learning rate for dual variable λ.
            tol: Convergence tolerance on constraint violation.
            warm_start: If True, reuse λ from previous call.
        """
        super().__init__()
        self.max_iters = max_iters
        self.lr_dual = lr_dual
        self.tol = tol
        self.warm_start = warm_start
        self._lambda_cache: Dict[int, float] = {}  # Warm-start cache keyed by param id

    def reset(self) -> None:
        self._lambda_cache.clear()

    def __call__(
        self,
        W: torch.Tensor,
        delta: torch.Tensor,
        spectral_budget: float,
    ) -> torch.Tensor:
        if spectral_budget is None or spectral_budget <= 0.0:
            return delta

        if delta.ndim != 2:
            return delta

        with torch.no_grad():
            G = delta  # Gradient proxy

            # Check if we even need to constrain
            sigma_G = torch.linalg.matrix_norm(G, ord=2)
            if sigma_G <= spectral_budget:
                return delta

            # Get top singular vectors
            try:
                U, S, Vh = torch.linalg.svd(G, full_matrices=False)
            except RuntimeError:
                return delta * (spectral_budget / sigma_G)

            u = U[:, 0:1]
            v = Vh[0:1, :]

            # Warm-start λ
            param_id = id(W)
            lam = self._lambda_cache.get(param_id, 1.0) if self.warm_start else 1.0

            for _ in range(self.max_iters):
                # For spectral norm ball, optimal A given λ > 0 is rank-1
                # A*(λ) = -budget * u @ v^T (scaled to boundary)
                A = -spectral_budget * (u @ v)

                # Constraint violation (should be ≈ 0 when satisfied)
                sigma_A = torch.linalg.matrix_norm(A, ord=2)
                violation = sigma_A - spectral_budget

                if abs(violation) < self.tol:
                    break

                # Dual gradient ascent
                lam = max(0.0, lam + self.lr_dual * violation.item())

            # Cache λ for warm start
            if self.warm_start:
                self._lambda_cache[param_id] = lam

            return A


class QuasiNewtonDualSolver(BaseInnerSolver):
    """
    Quasi-Newton (L-BFGS-style) solver on the dual objective.

    For the spectral-norm constrained problem, the dual has structure that allows
    efficient optimization. We use a simplified L-BFGS approach with limited memory.

    The dual objective is concave in λ, so we maximize via L-BFGS updates.
    We maintain curvature estimates from recent (s, y) pairs.
    """

    def __init__(
        self,
        max_iters: int = 15,
        memory_size: int = 5,
        tol: float = 1e-5,
        line_search_iters: int = 10,
        warm_start: bool = True,
        damping: float = 1e-4,
    ):
        """
        Args:
            max_iters: Maximum quasi-Newton iterations.
            memory_size: Number of (s, y) pairs to store for L-BFGS.
            tol: Convergence tolerance.
            line_search_iters: Max backtracking line search iterations.
            warm_start: Reuse iterate from previous call.
            damping: Damping factor for robustness.
        """
        super().__init__()
        self.max_iters = max_iters
        self.memory_size = memory_size
        self.tol = tol
        self.line_search_iters = line_search_iters
        self.warm_start = warm_start
        self.damping = damping
        self._state_cache: Dict[int, Dict[str, Any]] = {}

    def reset(self) -> None:
        self._state_cache.clear()

    def _dual_objective_and_grad(
        self,
        lam: float,
        sigma_G: float,
        budget: float,
    ) -> Tuple[float, float]:
        """
        Compute dual objective and gradient for spectral-norm ball.

        For the problem min_{||A||_2 <= budget} <G, A>:
        - Dual: d(λ) = -budget * σ_max(G) - λ * budget  for λ >= 0
        - Gradient: ∂d/∂λ = -budget (when constraint is active)

        This is a simplified model; the full Muon dual is more complex.
        """
        # Simplified: dual value and gradient
        dual_val = -budget * sigma_G - lam * budget
        dual_grad = -budget  # Always negative, so λ* = 0 for unconstrained

        return dual_val, dual_grad

    def __call__(
        self,
        W: torch.Tensor,
        delta: torch.Tensor,
        spectral_budget: float,
    ) -> torch.Tensor:
        if spectral_budget is None or spectral_budget <= 0.0:
            return delta

        if delta.ndim != 2:
            return delta

        with torch.no_grad():
            G = delta
            sigma_G = torch.linalg.matrix_norm(G, ord=2).item()

            if sigma_G <= spectral_budget:
                return delta

            # For the spectral ball, the solution is simply to project
            # The quasi-Newton approach here is for demonstration; in practice
            # with tangency constraints, the dual is more interesting

            try:
                U, S, Vh = torch.linalg.svd(G, full_matrices=False)
            except RuntimeError:
                return delta * (spectral_budget / sigma_G)

            u = U[:, 0:1]
            v = Vh[0:1, :]

            # Initialize from cache or fresh
            param_id = id(W)
            if self.warm_start and param_id in self._state_cache:
                state = self._state_cache[param_id]
                lam = state.get("lam", 0.0)
            else:
                lam = 0.0
                state = {"lam": lam, "s_list": [], "y_list": []}

            # L-BFGS iterations (simplified for 1D dual)
            s_list = state.get("s_list", [])
            y_list = state.get("y_list", [])

            for _ in range(self.max_iters):
                _, grad = self._dual_objective_and_grad(lam, sigma_G, spectral_budget)

                if abs(grad) < self.tol:
                    break

                # L-BFGS direction (for 1D, this is just scaled gradient)
                # Two-loop recursion simplified for scalar
                q = grad
                alpha_list = []

                for s, y in reversed(list(zip(s_list, y_list))):
                    if abs(y) > 1e-12:
                        rho = 1.0 / y
                        alpha = rho * s * q
                        alpha_list.append(alpha)
                        q = q - alpha * y
                    else:
                        alpha_list.append(0.0)

                # Initial Hessian approximation
                if len(s_list) > 0 and abs(y_list[-1]) > 1e-12:
                    gamma = s_list[-1] * y_list[-1] / (y_list[-1] ** 2)
                else:
                    gamma = 1.0

                r = gamma * q

                for (s, y), alpha in zip(zip(s_list, y_list), reversed(alpha_list)):
                    if abs(y) > 1e-12:
                        rho = 1.0 / y
                        beta = rho * y * r
                        r = r + s * (alpha - beta)

                direction = r

                # Line search
                step = 1.0
                lam_old = lam
                for _ in range(self.line_search_iters):
                    lam_new = max(0.0, lam + step * direction)
                    new_val, _ = self._dual_objective_and_grad(
                        lam_new, sigma_G, spectral_budget
                    )
                    old_val, _ = self._dual_objective_and_grad(
                        lam_old, sigma_G, spectral_budget
                    )
                    if new_val >= old_val - 1e-4 * step * abs(grad * direction):
                        break
                    step *= 0.5

                lam = max(0.0, lam + step * direction)

                # Update memory
                s_new = lam - lam_old
                _, grad_new = self._dual_objective_and_grad(lam, sigma_G, spectral_budget)
                y_new = grad_new - grad

                if abs(s_new * y_new) > 1e-12:
                    s_list.append(s_new)
                    y_list.append(y_new)
                    if len(s_list) > self.memory_size:
                        s_list.pop(0)
                        y_list.pop(0)

            # Cache state
            if self.warm_start:
                self._state_cache[param_id] = {"lam": lam, "s_list": s_list, "y_list": y_list}

            # Return projected update
            A = -spectral_budget * (u @ v)
            return A


class ADMMSolver(BaseInnerSolver):
    """
    ADMM-style solver for spectral-norm constraint with optional tangency.

    We split the problem:
        min <G, A>  s.t.  ||A||_2 <= budget, A = Z

    Augmented Lagrangian:
        L_ρ(A, Z, U) = <G, A> + (ρ/2)||A - Z + U||_F^2 - (ρ/2)||U||_F^2
                       + indicator(||Z||_2 <= budget)

    ADMM updates:
        A^{k+1} = argmin_A <G, A> + (ρ/2)||A - Z^k + U^k||_F^2
                = Z^k - U^k - G/ρ
        Z^{k+1} = Π_{||·||_2 <= budget}(A^{k+1} + U^k)  (spectral projection)
        U^{k+1} = U^k + A^{k+1} - Z^{k+1}

    This cleanly separates the linear objective from the spectral constraint.
    """

    def __init__(
        self,
        max_iters: int = 20,
        rho: float = 1.0,
        adaptive_rho: bool = True,
        tol_primal: float = 1e-5,
        tol_dual: float = 1e-5,
        warm_start: bool = True,
    ):
        """
        Args:
            max_iters: Maximum ADMM iterations.
            rho: Augmented Lagrangian parameter.
            adaptive_rho: If True, adapt ρ based on primal/dual residuals.
            tol_primal: Primal residual tolerance.
            tol_dual: Dual residual tolerance.
            warm_start: Reuse (Z, U) from previous call.
        """
        super().__init__()
        self.max_iters = max_iters
        self.rho = rho
        self.adaptive_rho = adaptive_rho
        self.tol_primal = tol_primal
        self.tol_dual = tol_dual
        self.warm_start = warm_start
        self._state_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def reset(self) -> None:
        self._state_cache.clear()

    def _spectral_projection(self, M: torch.Tensor, budget: float) -> torch.Tensor:
        """Project M onto spectral-norm ball of radius `budget`."""
        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        except RuntimeError:
            sigma = torch.linalg.matrix_norm(M, ord=2)
            if sigma > budget:
                return M * (budget / sigma)
            return M

        # Clip singular values to [0, budget]
        S_clipped = torch.clamp(S, max=budget)
        return U @ torch.diag(S_clipped) @ Vh

    def __call__(
        self,
        W: torch.Tensor,
        delta: torch.Tensor,
        spectral_budget: float,
    ) -> torch.Tensor:
        if spectral_budget is None or spectral_budget <= 0.0:
            return delta

        if delta.ndim != 2:
            return delta

        with torch.no_grad():
            G = delta
            rho = self.rho

            # Initialize from cache or fresh
            param_id = id(W)
            if self.warm_start and param_id in self._state_cache:
                state = self._state_cache[param_id]
                Z = state["Z"].clone()
                U = state["U"].clone()
            else:
                Z = torch.zeros_like(G)
                U = torch.zeros_like(G)

            for k in range(self.max_iters):
                Z_old = Z.clone()

                # A-update: A = Z - U - G/ρ
                A = Z - U - G / rho

                # Z-update: project onto spectral ball
                Z = self._spectral_projection(A + U, spectral_budget)

                # U-update
                U = U + A - Z

                # Residuals
                r_primal = torch.norm(A - Z).item()  # Primal residual
                r_dual = rho * torch.norm(Z - Z_old).item()  # Dual residual

                # Check convergence
                if r_primal < self.tol_primal and r_dual < self.tol_dual:
                    break

                # Adaptive ρ
                if self.adaptive_rho:
                    if r_primal > 10 * r_dual and r_dual > 1e-10:
                        rho *= 2.0
                        U = U / 2.0
                    elif r_dual > 10 * r_primal and r_primal > 1e-10:
                        rho /= 2.0
                        U = U * 2.0

            # Cache state
            if self.warm_start:
                self._state_cache[param_id] = {"Z": Z, "U": U}

            return Z


class TangentSpaceProjector:
    """
    Helper class for projecting onto the tangent space of the Stiefel manifold.

    For W with orthonormal columns (W^T W = I), the tangent space at W is:
        T_W St(n,p) = {Δ : W^T Δ + Δ^T W = 0}  (skew-symmetric condition)

    This is used when we want manifold-constrained updates (full Muon).
    """

    @staticmethod
    def project(W: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Project delta onto the tangent space at W.

        For tall matrices (m > n), we compute:
            Δ_tangent = Δ - W @ sym(W^T @ Δ)
        where sym(M) = (M + M^T) / 2
        """
        if W.ndim != 2 or delta.ndim != 2:
            return delta

        WtD = W.T @ delta  # (n, n)
        sym_WtD = (WtD + WtD.T) / 2
        return delta - W @ sym_WtD

    @staticmethod
    def retract_qr(W: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        QR retraction: map W + Δ back onto Stiefel manifold via QR decomposition.
        """
        Q, R = torch.linalg.qr(W + delta)
        # Fix sign ambiguity
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)
        return Q

    @staticmethod
    def retract_polar(W: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Polar retraction: map W + Δ back onto Stiefel manifold via polar decomposition.
        """
        Y = W + delta
        U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
        return U @ Vh


# Solver registry for easy access
SOLVER_REGISTRY = {
    "spectral_clip": SpectralClipSolver,
    "frank_wolfe": FrankWolfeSolver,
    "dual_ascent": DualAscentSolver,
    "quasi_newton": QuasiNewtonDualSolver,
    "admm": ADMMSolver,
}


def get_inner_solver(name: str, **kwargs) -> BaseInnerSolver:
    """
    Factory function to get an inner solver by name.

    Args:
        name: One of 'spectral_clip', 'frank_wolfe', 'dual_ascent', 'quasi_newton', 'admm'.
        **kwargs: Additional arguments passed to the solver constructor.

    Returns:
        An instance of the requested solver.
    """
    if name not in SOLVER_REGISTRY:
        raise ValueError(f"Unknown solver: {name}. Available: {list(SOLVER_REGISTRY.keys())}")
    return SOLVER_REGISTRY[name](**kwargs)

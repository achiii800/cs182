"""
Metrics and diagnostics for optimizer stability and geometry analysis.

Implements:
    - Spectral norms of weight matrices
    - SAM-style sharpness proxy
    - Gradient noise scale (GNS) / critical batch size proxy
    - Top Hessian eigenvalue estimation via power iteration / Lanczos
    - Subspace drift metrics (principal angles between weight subspaces)
    - Learning rate stability envelope scanning
"""

from typing import Dict, List, Tuple, Optional, Callable, Any
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np


def compute_spectral_norms(
    model: nn.Module,
    max_layers: int = 8,
    layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d),
) -> Dict[str, float]:
    """
    Compute spectral norms of up to `max_layers` 2D weight matrices.

    Args:
        model: The neural network.
        max_layers: Maximum number of layers to compute.
        layer_types: Only consider these layer types.

    Returns:
        Dict mapping "layer_name" -> spectral_norm (σ_max).
    """
    norms: Dict[str, float] = {}
    count = 0

    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                # Reshape conv weights to 2D: (out_channels, in_channels * k * k)
                if w.ndim > 2:
                    w = w.view(w.size(0), -1)
                if w.ndim == 2:
                    with torch.no_grad():
                        try:
                            sigma = torch.linalg.matrix_norm(w, ord=2).item()
                        except RuntimeError:
                            s = torch.linalg.svdvals(w)
                            sigma = s[0].item()
                    norms[name] = sigma
                    count += 1
                    if count >= max_layers:
                        break
    return norms


def compute_all_singular_values(
    model: nn.Module,
    layer_name: str,
) -> Optional[torch.Tensor]:
    """
    Compute all singular values for a specific layer.

    Useful for tracking the full spectrum evolution (conditioning, rank collapse).

    Args:
        model: The neural network.
        layer_name: Name of the layer (from named_modules).

    Returns:
        Tensor of singular values in descending order, or None if layer not found.
    """
    for name, module in model.named_modules():
        if name == layer_name:
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                if w.ndim > 2:
                    w = w.view(w.size(0), -1)
                if w.ndim == 2:
                    with torch.no_grad():
                        return torch.linalg.svdvals(w)
    return None


def estimate_sharpness(
    model: nn.Module,
    loss_fn: Callable,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-3,
    normalize: bool = True,
) -> float:
    """
    SAM-style sharpness proxy:
        sharpness ≈ loss(w + ε * g/||g||) - loss(w)

    where g is the gradient and we optionally normalize.

    This tracks curvature / flatness of the loss landscape.

    Args:
        model: The neural network.
        loss_fn: Callable (model, inputs, targets) -> loss tensor.
        inputs, targets: A minibatch.
        epsilon: Perturbation magnitude.
        normalize: If True, perturb in direction g/||g||. Otherwise use sign(g).

    Returns:
        Scalar sharpness estimate.
    """
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)

    model.zero_grad(set_to_none=True)
    base_loss = loss_fn(model, inputs, targets)
    base_loss.backward()

    # Compute perturbation
    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sq += p.grad.pow(2).sum().item()
    grad_norm = math.sqrt(grad_norm_sq) + 1e-12

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            if normalize:
                p.add_(epsilon * p.grad / grad_norm)
            else:
                p.add_(epsilon * torch.sign(p.grad))

    model.zero_grad(set_to_none=True)
    perturbed_loss = loss_fn(model, inputs, targets)

    # Restore weights
    with torch.no_grad():
        # Need to recompute grads to restore
        model.zero_grad(set_to_none=True)
        restore_loss = loss_fn(model, inputs, targets)
        restore_loss.backward()

        for p in model.parameters():
            if p.grad is None:
                continue
            if normalize:
                p.sub_(epsilon * p.grad / grad_norm)
            else:
                p.sub_(epsilon * torch.sign(p.grad))

    return (perturbed_loss - base_loss).item()


def estimate_gradient_noise_scale(
    model: nn.Module,
    loss_fn: Callable,
    batch1: Tuple[torch.Tensor, torch.Tensor],
    batch2: Tuple[torch.Tensor, torch.Tensor],
) -> float:
    """
    Gradient noise scale proxy from two independent minibatches.

    GNS ≈ ||g₁ - g₂||² / 2

    This is a cheap proxy for the gradient noise scale (CBS) without full sweeps.

    Args:
        model: The neural network.
        loss_fn: Callable (model, inputs, targets) -> loss tensor.
        batch1, batch2: Two (inputs, targets) tuples of the same batch size.

    Returns:
        Scalar GNS proxy.
    """
    device = next(model.parameters()).device

    def grad_vec(batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, x, y)
        loss.backward()
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
        return torch.cat(grads)

    with torch.no_grad():
        g1 = grad_vec(batch1)
        g2 = grad_vec(batch2)
        diff = g1 - g2
        return torch.dot(diff, diff).item() / 2.0


def estimate_top_hessian_eigenvalue(
    model: nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    num_batches: int = 5,
    power_iter_steps: int = 20,
    tol: float = 1e-4,
) -> float:
    """
    Estimate the top eigenvalue of the Hessian via power iteration.

    Uses Hessian-vector products (Pearlmutter trick) without forming the full Hessian.

    λ_max ≈ (v^T H v) / (v^T v)  where v is the dominant eigenvector.

    Args:
        model: The neural network.
        loss_fn: Callable (model, inputs, targets) -> loss tensor.
        data_loader: DataLoader for computing Hessian-vector products.
        num_batches: Number of batches to average HVP over.
        power_iter_steps: Number of power iteration steps.
        tol: Convergence tolerance.

    Returns:
        Estimated top Hessian eigenvalue.
    """
    device = next(model.parameters()).device

    # Initialize random vector
    v = []
    for p in model.parameters():
        v.append(torch.randn_like(p.data))

    def normalize(vec: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
        norm_sq = sum((vi ** 2).sum().item() for vi in vec)
        norm = math.sqrt(norm_sq) + 1e-12
        return [vi / norm for vi in vec], norm

    v, _ = normalize(v)

    def hvp(vec: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute Hessian-vector product via finite differences or autograd."""
        model.zero_grad(set_to_none=True)

        # Accumulate loss over batches
        total_loss = 0.0
        batch_count = 0
        for i, (x, y) in enumerate(data_loader):
            if i >= num_batches:
                break
            x = x.to(device)
            y = y.to(device)
            loss = loss_fn(model, x, y)
            total_loss = total_loss + loss
            batch_count += 1

        if batch_count == 0:
            return [torch.zeros_like(vi) for vi in vec]

        avg_loss = total_loss / batch_count

        # First backward to get gradients
        grads = torch.autograd.grad(avg_loss, model.parameters(), create_graph=True)

        # Compute gradient-vector product
        gv = sum((g * vi).sum() for g, vi in zip(grads, vec))

        # Second backward to get Hessian-vector product
        hvp_result = torch.autograd.grad(gv, model.parameters())

        return [h.detach() for h in hvp_result]

    eigenvalue = 0.0
    for _ in range(power_iter_steps):
        # Hv = H @ v
        Hv = hvp(v)

        # Rayleigh quotient: λ ≈ v^T H v
        eigenvalue_new = sum((vi * hvi).sum().item() for vi, hvi in zip(v, Hv))

        # Check convergence
        if abs(eigenvalue_new - eigenvalue) < tol:
            break
        eigenvalue = eigenvalue_new

        # Normalize Hv to get new v
        v, _ = normalize(Hv)

    return eigenvalue


def estimate_hessian_spectrum_lanczos(
    model: nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    num_batches: int = 5,
    num_eigenvalues: int = 5,
    max_iters: int = 50,
) -> np.ndarray:
    """
    Estimate top-k Hessian eigenvalues via Lanczos iteration.

    More accurate than power iteration for multiple eigenvalues,
    but more expensive.

    Args:
        model: The neural network.
        loss_fn: Callable (model, inputs, targets) -> loss tensor.
        data_loader: DataLoader for HVP computation.
        num_batches: Batches for averaging HVP.
        num_eigenvalues: Number of top eigenvalues to estimate.
        max_iters: Maximum Lanczos iterations.

    Returns:
        Array of estimated eigenvalues in descending order.
    """
    device = next(model.parameters()).device

    # Total parameter count
    num_params = sum(p.numel() for p in model.parameters())

    def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.flatten() for p in params])

    def unflatten_params(flat: torch.Tensor) -> List[torch.Tensor]:
        result = []
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            result.append(flat[offset : offset + numel].view_as(p))
            offset += numel
        return result

    def hvp_flat(v_flat: torch.Tensor) -> torch.Tensor:
        """HVP with flattened vectors."""
        v = unflatten_params(v_flat)

        model.zero_grad(set_to_none=True)
        total_loss = 0.0
        batch_count = 0
        for i, (x, y) in enumerate(data_loader):
            if i >= num_batches:
                break
            x = x.to(device)
            y = y.to(device)
            loss = loss_fn(model, x, y)
            total_loss = total_loss + loss
            batch_count += 1

        if batch_count == 0:
            return torch.zeros_like(v_flat)

        avg_loss = total_loss / batch_count
        grads = torch.autograd.grad(avg_loss, model.parameters(), create_graph=True)
        gv = sum((g * vi).sum() for g, vi in zip(grads, v))
        hvp_result = torch.autograd.grad(gv, model.parameters())

        return flatten_params([h.detach() for h in hvp_result])

    # Lanczos iteration
    k = min(num_eigenvalues + 10, max_iters, num_params)

    # Initialize
    q = torch.randn(num_params, device=device)
    q = q / torch.norm(q)

    Q = [q]
    alpha = []
    beta = []

    for j in range(k):
        w = hvp_flat(Q[-1])

        a = torch.dot(w, Q[-1]).item()
        alpha.append(a)

        if j == 0:
            w = w - a * Q[-1]
        else:
            w = w - a * Q[-1] - beta[-1] * Q[-2]

        b = torch.norm(w).item()

        if b < 1e-10:
            break

        beta.append(b)
        q_new = w / b
        Q.append(q_new)

    # Build tridiagonal matrix T and compute its eigenvalues
    n = len(alpha)
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = alpha[i]
    for i in range(n - 1):
        T[i, i + 1] = beta[i]
        T[i + 1, i] = beta[i]

    eigenvalues = np.linalg.eigvalsh(T)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    return eigenvalues[:num_eigenvalues]


def compute_subspace_drift(
    W_old: torch.Tensor,
    W_new: torch.Tensor,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Compute subspace drift between two weight matrices via principal angles.

    Measures how much the top-k left/right singular subspaces have changed.

    Args:
        W_old: Previous weight matrix.
        W_new: Current weight matrix.
        top_k: Number of top singular vectors to compare.

    Returns:
        Dict with 'left_drift' (max principal angle for left sing. vectors)
        and 'right_drift' (for right sing. vectors), in radians.
    """
    if W_old.ndim > 2:
        W_old = W_old.view(W_old.size(0), -1)
    if W_new.ndim > 2:
        W_new = W_new.view(W_new.size(0), -1)

    with torch.no_grad():
        U_old, _, Vh_old = torch.linalg.svd(W_old, full_matrices=False)
        U_new, _, Vh_new = torch.linalg.svd(W_new, full_matrices=False)

        k = min(top_k, U_old.size(1), U_new.size(1))

        # Left singular vectors (columns of U)
        U_old_k = U_old[:, :k]
        U_new_k = U_new[:, :k]

        # Principal angles: cos(θ) = singular values of U_old^T @ U_new
        cos_angles_left = torch.linalg.svdvals(U_old_k.T @ U_new_k)
        cos_angles_left = torch.clamp(cos_angles_left, -1.0, 1.0)
        angles_left = torch.acos(cos_angles_left)
        max_angle_left = angles_left.max().item()

        # Right singular vectors (rows of Vh)
        V_old_k = Vh_old[:k, :].T  # (n, k)
        V_new_k = Vh_new[:k, :].T

        cos_angles_right = torch.linalg.svdvals(V_old_k.T @ V_new_k)
        cos_angles_right = torch.clamp(cos_angles_right, -1.0, 1.0)
        angles_right = torch.acos(cos_angles_right)
        max_angle_right = angles_right.max().item()

    return {
        "left_drift": max_angle_left,
        "right_drift": max_angle_right,
    }


class SubspaceDriftTracker:
    """
    Track subspace drift for selected layers across training.

    Stores snapshots of weight matrices and computes drift metrics.
    """

    def __init__(self, layer_names: List[str], top_k: int = 5):
        self.layer_names = layer_names
        self.top_k = top_k
        self._snapshots: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
        self._drift_history: Dict[str, List[Dict[str, float]]] = {name: [] for name in layer_names}

    def snapshot(self, model: nn.Module) -> None:
        """Take a snapshot of tracked layers."""
        for name, module in model.named_modules():
            if name in self.layer_names:
                if hasattr(module, "weight") and module.weight is not None:
                    self._snapshots[name].append(module.weight.data.clone().cpu())

    def compute_drift(self) -> Dict[str, Dict[str, float]]:
        """Compute drift between last two snapshots for each layer."""
        result = {}
        for name in self.layer_names:
            snaps = self._snapshots[name]
            if len(snaps) >= 2:
                drift = compute_subspace_drift(snaps[-2], snaps[-1], self.top_k)
                self._drift_history[name].append(drift)
                result[name] = drift
        return result

    def get_history(self, layer_name: str) -> List[Dict[str, float]]:
        """Get drift history for a layer."""
        return self._drift_history.get(layer_name, [])


def lr_stability_scan(
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module, float], torch.optim.Optimizer],
    loss_fn: Callable,
    train_loader: DataLoader,
    lr_range: Tuple[float, float] = (1e-4, 1.0),
    num_lrs: int = 10,
    steps_per_lr: int = 50,
    divergence_threshold: float = 1e6,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Scan learning rates to find the stable LR envelope.

    For each LR, run a short training and check for divergence.

    Args:
        model_factory: Callable returning a fresh model.
        optimizer_factory: Callable (model, lr) -> optimizer.
        loss_fn: Callable (model, inputs, targets) -> loss.
        train_loader: Training data loader.
        lr_range: (min_lr, max_lr) to scan.
        num_lrs: Number of LR values to test.
        steps_per_lr: Training steps per LR.
        divergence_threshold: Loss above this is considered diverged.
        device: Device to use.

    Returns:
        Dict with 'lr_values', 'final_losses', 'converged', 'max_stable_lr'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log-spaced LR values
    lr_values = np.logspace(
        np.log10(lr_range[0]), np.log10(lr_range[1]), num_lrs
    ).tolist()

    final_losses = []
    converged = []

    for lr in lr_values:
        model = model_factory().to(device)
        optimizer = optimizer_factory(model, lr)

        loss_val = 0.0
        diverged = False

        step = 0
        for x, y in train_loader:
            if step >= steps_per_lr:
                break

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model, x, y)
            loss_val = loss.item()

            if loss_val > divergence_threshold or math.isnan(loss_val):
                diverged = True
                break

            loss.backward()
            optimizer.step()
            step += 1

        final_losses.append(loss_val if not diverged else float("inf"))
        converged.append(not diverged)

    # Find max stable LR
    max_stable_lr = lr_values[0]
    for lr, conv in zip(lr_values, converged):
        if conv:
            max_stable_lr = lr
        else:
            break

    return {
        "lr_values": lr_values,
        "final_losses": final_losses,
        "converged": converged,
        "max_stable_lr": max_stable_lr,
    }


class MetricsLogger:
    """
    Unified metrics logger that collects all diagnostics per epoch.

    Designed for easy CSV export and aggregation across seeds.
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def log(self, metrics: Dict[str, Any]) -> None:
        """Log a dict of metrics for one epoch."""
        self.history.append(metrics)

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.history)

    def save_csv(self, path: str) -> None:
        """Save to CSV file."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def save_json(self, path: str) -> None:
        """Save to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    @staticmethod
    def aggregate_seeds(loggers: List["MetricsLogger"]) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple seeds.

        Returns dict with 'mean' and 'std' DataFrames.
        """
        import pandas as pd

        dfs = [logger.to_dataframe() for logger in loggers]

        # Stack and compute statistics
        combined = pd.concat(dfs, keys=range(len(dfs)))

        mean_df = combined.groupby(level=1).mean()
        std_df = combined.groupby(level=1).std()

        return {"mean": mean_df, "std": std_df}


from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def compute_spectral_norms(model: nn.Module, max_layers: int = 8) -> Dict[str, float]:
    """
    Compute spectral norms of up to `max_layers` 2D weight matrices in the model.

    Returns:
        dict mapping "layer_name" -> spectral_norm
    """
    norms: Dict[str, float] = {}
    count = 0
    for name, param in model.named_parameters():
        if param.ndim == 2:
            with torch.no_grad():
                try:
                    sigma = torch.linalg.matrix_norm(param.data, ord=2).item()
                except RuntimeError:
                    s = torch.linalg.svdvals(param.data)
                    sigma = s.max().item()
            norms[name] = sigma
            count += 1
            if count >= max_layers:
                break
    return norms


def estimate_sharpness(
    model: nn.Module,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-3,
) -> float:
    """
    Very crude SAM-style sharpness proxy:
      sharpness ≈ loss(w + epsilon * sign(grad)) - loss(w)

    This is *not* a rigorous Hessian top eigenvalue estimate but is cheap and
    often tracks sharpness qualitatively.

    Args:
        model: the nn.Module.
        loss_fn: a callable (model, inputs, targets) -> loss tensor.
        inputs, targets: a single minibatch.
        epsilon: magnitude of perturbation.

    Returns:
        scalar float sharpness estimate.
    """
    device = next(model.parameters()).device

    model.zero_grad(set_to_none=True)
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Compute base loss and gradient
    base_loss = loss_fn(model, inputs, targets)
    base_loss.backward()

    with torch.no_grad():
        # Perturb in the sign of the gradient
        for p in model.parameters():
            if p.grad is None:
                continue
            p.add_(epsilon * torch.sign(p.grad))

    # Recompute loss at perturbed weights
    perturbed_loss = loss_fn(model, inputs, targets)

    # Restore original weights by subtracting the perturbation
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            p.sub_(epsilon * torch.sign(p.grad))

    return (perturbed_loss - base_loss).item()


def estimate_gradient_noise_scale(
    model: nn.Module,
    loss_fn,
    batch1: Tuple[torch.Tensor, torch.Tensor],
    batch2: Tuple[torch.Tensor, torch.Tensor],
) -> float:
    """
    Very coarse gradient-noise-scale proxy using two mini-batches of the same size.

    We estimate:
        GNS ≈ ||g1 - g2||^2 / 2
    where g1, g2 are flattened gradients from two independent batches.

    This is not the full CBS measurement from the literature but gives a scalar that
    often tracks noise scale qualitatively over training.

    Args:
        model: nn.Module
        loss_fn: callable (model, inputs, targets) -> loss tensor
        batch1, batch2: (inputs, targets) tuples

    Returns:
        float scalar GNS proxy.
    """
    device = next(model.parameters()).device

    def grad_vec(batch):
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

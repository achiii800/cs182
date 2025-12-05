#!/usr/bin/env python3
"""
Unified training script for CIFAR-10 experiments.

Supports:
    - Multiple architectures: small_cnn, resnet18, tiny_vit, mlp_mixer, mlp
    - Multiple optimizers: sgd, adamw, muon_sgd, muon_adamw
    - Multiple inner solvers: spectral_clip, frank_wolfe, dual_ascent, quasi_newton, admm
    - Width multiplier for muP-style transfer experiments
    - Comprehensive logging of metrics

Usage:
    # Quick baseline
    python train.py --model small_cnn --optimizer sgd --epochs 10

    # MuonSGD with spectral clipping on ResNet-18
    python train.py --model resnet18 --optimizer muon_sgd --inner-solver spectral_clip \\
                    --spectral-budget 0.1 --epochs 50

    # Width sweep for muP experiments
    for w in 0.5 1.0 2.0; do
        python train.py --model mlp --width-mult $w --optimizer muon_sgd --seed 0
    done

    # Multi-seed runs
    for seed in 0 1 2 3 4; do
        python train.py --model resnet18 --optimizer muon_sgd --seed $seed
    done
"""

import argparse
import os
import json
import time
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from muon import (
    create_optimizer,
    compute_spectral_norms,
    estimate_sharpness,
    estimate_gradient_noise_scale,
    estimate_top_hessian_eigenvalue,
    MetricsLogger,
)
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train models on CIFAR-10 with various optimizers and inner solvers."
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="small_cnn",
        choices=["small_cnn", "resnet18", "tiny_vit", "mlp_mixer", "mlp"],
        help="Model architecture."
    )
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier for muP experiments.")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes.")

    # Optimizer
    parser.add_argument(
        "--optimizer", type=str, default="muon_sgd",
        choices=["sgd", "adamw", "muon_sgd", "muon_adamw"],
        help="Base optimizer."
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (for SGD).")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--nesterov", action="store_true", help="Use Nesterov momentum.")

    # Inner solver
    parser.add_argument(
        "--inner-solver", type=str, default="spectral_clip",
        choices=["none", "spectral_clip", "frank_wolfe", "dual_ascent", "quasi_newton", "admm"],
        help="Inner solver for spectral-norm constraint."
    )
    parser.add_argument("--spectral-budget", type=float, default=0.1, help="Spectral norm budget.")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")

    # Learning rate schedule
    parser.add_argument(
        "--lr-schedule", type=str, default="cosine",
        choices=["none", "cosine", "step", "linear"],
        help="Learning rate schedule."
    )
    parser.add_argument("--warmup-epochs", type=int, default=0, help="LR warmup epochs.")

    # Logging
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for logs.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (auto-generated if None).")
    parser.add_argument("--log-interval", type=int, default=100, help="Log every N batches.")

    # Metrics
    parser.add_argument("--compute-hessian-eig", action="store_true", help="Compute top Hessian eigenvalue (slow).")
    parser.add_argument("--hessian-eig-interval", type=int, default=5, help="Compute Hessian eig every N epochs.")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if None).")

    return parser.parse_args()


def make_loaders(
    batch_size: int,
    num_workers: int = 4,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train/test data loaders."""

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def ce_loss_fn(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss function compatible with metrics API."""
    return F.cross_entropy(model(x), y)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a data loader. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule_type: str,
    epochs: int,
    warmup_epochs: int = 0,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""

    if schedule_type == "none":
        return None
    elif schedule_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
    elif schedule_type == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[epochs // 2, 3 * epochs // 4], gamma=0.1
        )
    elif schedule_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs - warmup_epochs
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """Apply linear warmup to learning rate."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def main():
    args = parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create experiment name
    if args.exp_name is None:
        args.exp_name = (
            f"{args.model}_"
            f"{args.optimizer}_"
            f"{args.inner_solver}_"
            f"budget{args.spectral_budget}_"
            f"lr{args.lr}_"
            f"wd{args.weight_decay}_"
            f"width{args.width_mult}_"
            f"seed{args.seed}"
        )

    # Create log directory
    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, f"{args.exp_name}.csv")
    config_path = os.path.join(args.logdir, f"{args.exp_name}_config.json")

    # Save config
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Experiment: {args.exp_name}")
    print(f"Device: {device}")
    print(f"Config saved to: {config_path}")

    # Create data loaders
    train_loader, test_loader = make_loaders(args.batch_size, args.num_workers)

    # Create model
    model = get_model(
        args.model,
        num_classes=args.num_classes,
        width_mult=args.width_mult,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, Parameters: {num_params:,}")

    # Create optimizer
    spectral_budget = args.spectral_budget if args.inner_solver != "none" else None
    optimizer = create_optimizer(
        model,
        optimizer_type=args.optimizer,
        inner_solver_type=args.inner_solver,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        spectral_budget=spectral_budget,
        nesterov=args.nesterov,
    )

    # Create LR scheduler
    scheduler = create_lr_scheduler(
        optimizer, args.lr_schedule, args.epochs, args.warmup_epochs
    )

    # Metrics logger
    logger = MetricsLogger()

    # CSV header
    with open(log_path, "w") as f:
        f.write(
            "epoch,train_loss,train_acc,val_loss,val_acc,"
            "max_spectral_norm,sharpness,grad_noise_scale,hessian_eig,lr,time\n"
        )

    print("\nStarting training...")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Warmup
        if epoch <= args.warmup_epochs:
            warmup_lr(optimizer, epoch - 1, args.warmup_epochs, args.lr)

        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        # Get batches for GNS estimation
        gns_batches = []
        for i, (x, y) in enumerate(train_loader):
            gns_batches.append((x, y))
            if len(gns_batches) >= 2:
                break

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            if (batch_idx + 1) % args.log_interval == 0:
                batch_loss = running_loss / total_samples
                batch_acc = running_correct / total_samples
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {batch_loss:.4f} Acc: {batch_acc:.4f}")

        train_loss = running_loss / total_samples
        train_acc = running_correct / total_samples

        # Validation
        val_loss, val_acc = evaluate(model, test_loader, device)

        # Compute metrics
        spec_norms = compute_spectral_norms(model, max_layers=8)
        max_spec = max(spec_norms.values()) if spec_norms else 0.0

        sharpness = estimate_sharpness(
            model, ce_loss_fn,
            gns_batches[0][0], gns_batches[0][1],
            epsilon=1e-3,
        )

        gns = estimate_gradient_noise_scale(
            model, ce_loss_fn,
            gns_batches[0], gns_batches[1],
        )

        # Hessian eigenvalue (expensive, optional)
        hessian_eig = 0.0
        if args.compute_hessian_eig and epoch % args.hessian_eig_interval == 0:
            try:
                hessian_eig = estimate_top_hessian_eigenvalue(
                    model, ce_loss_fn, train_loader,
                    num_batches=3, power_iter_steps=10,
                )
            except Exception as e:
                print(f"  Warning: Hessian estimation failed: {e}")

        # Get current LR
        current_lr = optimizer.param_groups[0]["lr"]

        # Step scheduler (after warmup)
        if scheduler is not None and epoch > args.warmup_epochs:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "max_spectral_norm": max_spec,
            "sharpness": sharpness,
            "grad_noise_scale": gns,
            "hessian_eig": hessian_eig,
            "lr": current_lr,
            "time": epoch_time,
        }
        logger.log(metrics)

        # Write to CSV
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                f"{val_loss:.6f},{val_acc:.6f},"
                f"{max_spec:.6f},{sharpness:.6f},{gns:.6f},{hessian_eig:.6f},"
                f"{current_lr:.6f},{epoch_time:.2f}\n"
            )

        print(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.4f}/{train_acc:.4f} | "
            f"Val: {val_loss:.4f}/{val_acc:.4f} | "
            f"σ_max: {max_spec:.3f} | "
            f"Sharp: {sharpness:.4f} | "
            f"GNS: {gns:.2e} | "
            f"LR: {current_lr:.5f} | "
            f"Time: {epoch_time:.1f}s"
        )

    print("-" * 80)
    print(f"Training complete. Logs saved to: {log_path}")

    # Save final metrics summary
    summary_path = os.path.join(args.logdir, f"{args.exp_name}_summary.json")
    summary = {
        "final_train_loss": train_loss,
        "final_train_acc": train_acc,
        "final_val_loss": val_loss,
        "final_val_acc": val_acc,
        "best_val_acc": max(m["val_acc"] for m in logger.history),
        "final_max_spectral_norm": max_spec,
        "total_epochs": args.epochs,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()


"""
Minimal CIFAR-10 experiment scaffold for comparing SGD vs MuonSGD inner solvers.

This is meant as a *starting point* you can quickly run in Colab or locally.
It intentionally keeps the model small and the training short so you can iterate
on inner solvers and logging without waiting hours.

Usage (from project root):

    python train_cifar_muon.py --epochs 5 --batch-size 256 --lr 0.1

On Colab:
    - Upload this project zip and unzip.
    - pip install -r requirements.txt
    - Run this script.

You can later:
    - Swap the network for a ResNet-18 or ViT.
    - Plug in a faithful manifold Muon inner solver.
    - Add more metrics / logging.
"""

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from muon import MuonSGD, SpectralClipSolver, FrankWolfeSolver
from muon.metrics import compute_spectral_norms, estimate_sharpness, estimate_gradient_noise_scale


class SmallConvNet(nn.Module):
    """
    Tiny CNN for CIFAR-10.

    This is intentionally small so that:
      - inner solvers (including SVD) are cheap,
      - you can run multiple seeds / configurations quickly.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # [B, 32, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 16, 16]
        x = self.pool(x)                # [B, 64, 8, 8]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def make_loaders(batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def ce_loss_fn(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits, y)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--spectral-budget", type=float, default=0.1)
    parser.add_argument("--inner-solver", type=str, default="spectral_clip",
                        choices=["spectral_clip", "frank_wolfe", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, f"cifar_muon_seed{args.seed}.csv")

    device = torch.device(args.device)

    train_loader, test_loader = make_loaders(args.batch_size)

    model = SmallConvNet(num_classes=10).to(device)

    if args.inner_solver == "spectral_clip":
        inner_solver = SpectralClipSolver()
    elif args.inner_solver == "frank_wolfe":
        inner_solver = FrankWolfeSolver(blend_with_raw=0.3)
    else:
        inner_solver = None

    if inner_solver is None:
        # Plain SGD baseline
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = MuonSGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            spectral_budget=args.spectral_budget,
            inner_solver=inner_solver,
        )

    # Simple cosine LR decay (optional)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # CSV header
    with open(log_path, "w") as f:
        f.write(
            "epoch,train_loss,train_acc,val_loss,val_acc,"
            "max_spectral_norm,sharpness,grad_noise_scale,lr\n"
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        # Grab two batches for GNS estimation once per epoch
        gns_batches = []
        for i, (x, y) in enumerate(train_loader):
            gns_batches.append((x, y))
            if len(gns_batches) >= 2:
                break

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            def closure():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                return loss

            optimizer.zero_grad(set_to_none=True)
            loss = closure()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = model(x).argmax(dim=1)
            running_correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        val_loss, val_acc = evaluate(model, test_loader, device)

        # Diagnostics
        spec_norms = compute_spectral_norms(model, max_layers=8)
        max_spec = max(spec_norms.values()) if spec_norms else 0.0

        sharpness = estimate_sharpness(
            model, ce_loss_fn, gns_batches[0][0], gns_batches[0][1], epsilon=1e-3
        )

        gns = estimate_gradient_noise_scale(
            model,
            ce_loss_fn,
            gns_batches[0],
            gns_batches[1],
        )

        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"max_spec={max_spec:.3f}, sharpness={sharpness:.4f}, gns={gns:.4f}"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                f"{val_loss:.6f},{val_acc:.6f},"
                f"{max_spec:.6f},{sharpness:.6f},{gns:.6f},{current_lr:.6f}\n"
            )


if __name__ == "__main__":
    main()

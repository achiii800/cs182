#!/usr/bin/env python3
"""
Helper scripts for running batch experiments.

Provides:
    - Multi-seed runs with aggregation
    - Learning rate stability envelope scans
    - muP-style width transfer sweeps
    - Optimizer comparison experiments

Usage:
    # Run multi-seed experiment
    python scripts/run_experiments.py multi_seed --model resnet18 --optimizer muon_sgd --seeds 5

    # LR stability scan
    python scripts/run_experiments.py lr_scan --model small_cnn --optimizer muon_sgd

    # Width transfer sweep
    python scripts/run_experiments.py width_sweep --model mlp --optimizer muon_sgd

    # Full comparison
    python scripts/run_experiments.py compare_optimizers --model resnet18 --epochs 50
"""

import argparse
import os
import subprocess
import json
from typing import List, Dict, Any
import time


def run_command(cmd: List[str], verbose: bool = True) -> int:
    """Run a shell command and return exit code."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not verbose)
    return result.returncode


def multi_seed_experiment(
    model: str,
    optimizer: str,
    inner_solver: str = "spectral_clip",
    spectral_budget: float = 0.1,
    lr: float = 0.1,
    epochs: int = 50,
    num_seeds: int = 5,
    logdir: str = "logs",
    **kwargs,
):
    """
    Run the same experiment across multiple seeds.

    Outputs logs that can be aggregated with plot_logs.py --aggregate-seeds.
    """
    print(f"\n{'='*60}")
    print(f"Multi-seed experiment: {model} + {optimizer} + {inner_solver}")
    print(f"Seeds: {num_seeds}, Epochs: {epochs}")
    print(f"{'='*60}\n")

    exp_base = f"{model}_{optimizer}_{inner_solver}_budget{spectral_budget}"

    for seed in range(num_seeds):
        cmd = [
            "python", "train.py",
            "--model", model,
            "--optimizer", optimizer,
            "--inner-solver", inner_solver,
            "--spectral-budget", str(spectral_budget),
            "--lr", str(lr),
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--logdir", logdir,
            "--exp-name", f"{exp_base}_seed{seed}",
        ]

        print(f"\n--- Seed {seed} ---")
        run_command(cmd)

    print(f"\nCompleted {num_seeds} seeds. Logs in: {logdir}/{exp_base}_seed*.csv")
    print(f"Aggregate with: python plot_logs.py --logfiles {logdir}/{exp_base}_seed*.csv --aggregate-seeds")


def lr_stability_scan(
    model: str,
    optimizer: str,
    inner_solver: str = "spectral_clip",
    spectral_budget: float = 0.1,
    lr_min: float = 1e-4,
    lr_max: float = 1.0,
    num_lrs: int = 10,
    steps_per_lr: int = 100,
    logdir: str = "logs",
    **kwargs,
):
    """
    Scan learning rates to find stable LR envelope.

    Quick runs at each LR to detect convergence/divergence threshold.
    """
    import numpy as np

    print(f"\n{'='*60}")
    print(f"LR Stability Scan: {model} + {optimizer} + {inner_solver}")
    print(f"LR range: [{lr_min}, {lr_max}], {num_lrs} values")
    print(f"{'='*60}\n")

    lr_values = np.logspace(np.log10(lr_min), np.log10(lr_max), num_lrs)

    results = {
        "lr_values": [],
        "final_losses": [],
        "converged": [],
    }

    for lr in lr_values:
        cmd = [
            "python", "train.py",
            "--model", model,
            "--optimizer", optimizer,
            "--inner-solver", inner_solver,
            "--spectral-budget", str(spectral_budget),
            "--lr", str(lr),
            "--epochs", "1",  # Quick scan
            "--batch-size", "256",
            "--seed", "0",
            "--logdir", logdir,
            "--exp-name", f"lr_scan_{model}_{optimizer}_lr{lr:.6f}",
            "--lr-schedule", "none",
        ]

        print(f"\nTesting LR = {lr:.6f}")
        returncode = run_command(cmd, verbose=False)

        # Check if converged by reading log
        log_path = os.path.join(logdir, f"lr_scan_{model}_{optimizer}_lr{lr:.6f}.csv")
        try:
            with open(log_path) as f:
                lines = f.readlines()
                if len(lines) > 1:
                    final_loss = float(lines[-1].split(",")[1])
                    converged = final_loss < 1e6 and not np.isnan(final_loss)
                else:
                    final_loss = float("inf")
                    converged = False
        except:
            final_loss = float("inf")
            converged = False

        results["lr_values"].append(float(lr))
        results["final_losses"].append(final_loss)
        results["converged"].append(converged)

        status = "✓ Converged" if converged else "✗ Diverged"
        print(f"  Loss: {final_loss:.4f} {status}")

    # Find max stable LR
    max_stable_lr = lr_values[0]
    for lr, conv in zip(lr_values, results["converged"]):
        if conv:
            max_stable_lr = lr
        else:
            break

    results["max_stable_lr"] = float(max_stable_lr)

    # Save results
    results_path = os.path.join(logdir, f"lr_scan_{model}_{optimizer}_{inner_solver}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Max stable LR: {max_stable_lr:.6f}")
    print(f"Results saved to: {results_path}")


def width_transfer_sweep(
    model: str,
    optimizer: str,
    inner_solver: str = "spectral_clip",
    spectral_budget: float = 0.1,
    lr: float = 0.1,
    epochs: int = 30,
    width_mults: List[float] = [0.5, 0.75, 1.0, 1.5, 2.0],
    seed: int = 0,
    logdir: str = "logs",
    **kwargs,
):
    """
    muP-style width transfer experiment.

    Trains the same model at different widths with fixed hyperparameters
    to test if spectral constraints enable better transfer.
    """
    print(f"\n{'='*60}")
    print(f"Width Transfer Sweep: {model} + {optimizer} + {inner_solver}")
    print(f"Widths: {width_mults}")
    print(f"{'='*60}\n")

    results = {}

    for width in width_mults:
        cmd = [
            "python", "train.py",
            "--model", model,
            "--optimizer", optimizer,
            "--inner-solver", inner_solver,
            "--spectral-budget", str(spectral_budget),
            "--lr", str(lr),
            "--epochs", str(epochs),
            "--width-mult", str(width),
            "--seed", str(seed),
            "--logdir", logdir,
            "--exp-name", f"width_{model}_{optimizer}_w{width}_seed{seed}",
        ]

        print(f"\n--- Width mult = {width} ---")
        run_command(cmd)

        # Read summary
        summary_path = os.path.join(logdir, f"width_{model}_{optimizer}_w{width}_seed{seed}_summary.json")
        try:
            with open(summary_path) as f:
                summary = json.load(f)
                results[width] = {
                    "val_acc": summary.get("best_val_acc", 0),
                    "val_loss": summary.get("final_val_loss", 0),
                }
        except:
            results[width] = {"val_acc": 0, "val_loss": float("inf")}

    # Save aggregated results
    results_path = os.path.join(logdir, f"width_sweep_{model}_{optimizer}_{inner_solver}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Width Transfer Results:")
    for width, metrics in sorted(results.items()):
        print(f"  Width {width}: Val Acc = {metrics['val_acc']:.4f}")
    print(f"Results saved to: {results_path}")


def compare_optimizers(
    model: str,
    epochs: int = 50,
    seed: int = 0,
    logdir: str = "logs",
    **kwargs,
):
    """
    Compare different optimizer configurations on the same model.

    Runs:
        - SGD baseline
        - AdamW baseline
        - MuonSGD with each inner solver
    """
    print(f"\n{'='*60}")
    print(f"Optimizer Comparison: {model}")
    print(f"{'='*60}\n")

    configs = [
        {"optimizer": "sgd", "inner_solver": "none", "lr": 0.1},
        {"optimizer": "adamw", "inner_solver": "none", "lr": 1e-3},
        {"optimizer": "muon_sgd", "inner_solver": "spectral_clip", "lr": 0.1},
        {"optimizer": "muon_sgd", "inner_solver": "dual_ascent", "lr": 0.1},
        {"optimizer": "muon_sgd", "inner_solver": "frank_wolfe", "lr": 0.1},
        {"optimizer": "muon_sgd", "inner_solver": "quasi_newton", "lr": 0.1},
        {"optimizer": "muon_sgd", "inner_solver": "admm", "lr": 0.1},
    ]

    results = {}

    for config in configs:
        opt = config["optimizer"]
        solver = config["inner_solver"]
        lr = config["lr"]

        exp_name = f"compare_{model}_{opt}_{solver}_seed{seed}"

        cmd = [
            "python", "train.py",
            "--model", model,
            "--optimizer", opt,
            "--inner-solver", solver,
            "--spectral-budget", "0.1",
            "--lr", str(lr),
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--logdir", logdir,
            "--exp-name", exp_name,
        ]

        print(f"\n--- {opt} + {solver} ---")
        run_command(cmd)

        # Read summary
        summary_path = os.path.join(logdir, f"{exp_name}_summary.json")
        try:
            with open(summary_path) as f:
                summary = json.load(f)
                results[f"{opt}_{solver}"] = {
                    "val_acc": summary.get("best_val_acc", 0),
                    "final_val_acc": summary.get("final_val_acc", 0),
                }
        except:
            results[f"{opt}_{solver}"] = {"val_acc": 0, "final_val_acc": 0}

    # Print comparison
    print(f"\n{'='*60}")
    print("Optimizer Comparison Results:")
    print("-" * 40)
    for name, metrics in sorted(results.items(), key=lambda x: -x[1]["val_acc"]):
        print(f"  {name:30s}: Best Val Acc = {metrics['val_acc']:.4f}")

    # Save
    results_path = os.path.join(logdir, f"compare_{model}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Run batch experiments.")
    subparsers = parser.add_subparsers(dest="command", help="Experiment type")

    # Multi-seed
    multi_seed_parser = subparsers.add_parser("multi_seed", help="Multi-seed experiment")
    multi_seed_parser.add_argument("--model", type=str, default="resnet18")
    multi_seed_parser.add_argument("--optimizer", type=str, default="muon_sgd")
    multi_seed_parser.add_argument("--inner-solver", type=str, default="spectral_clip")
    multi_seed_parser.add_argument("--spectral-budget", type=float, default=0.1)
    multi_seed_parser.add_argument("--lr", type=float, default=0.1)
    multi_seed_parser.add_argument("--epochs", type=int, default=50)
    multi_seed_parser.add_argument("--seeds", type=int, default=5)
    multi_seed_parser.add_argument("--logdir", type=str, default="logs")

    # LR scan
    lr_scan_parser = subparsers.add_parser("lr_scan", help="LR stability scan")
    lr_scan_parser.add_argument("--model", type=str, default="small_cnn")
    lr_scan_parser.add_argument("--optimizer", type=str, default="muon_sgd")
    lr_scan_parser.add_argument("--inner-solver", type=str, default="spectral_clip")
    lr_scan_parser.add_argument("--spectral-budget", type=float, default=0.1)
    lr_scan_parser.add_argument("--lr-min", type=float, default=1e-4)
    lr_scan_parser.add_argument("--lr-max", type=float, default=1.0)
    lr_scan_parser.add_argument("--num-lrs", type=int, default=10)
    lr_scan_parser.add_argument("--logdir", type=str, default="logs")

    # Width sweep
    width_parser = subparsers.add_parser("width_sweep", help="Width transfer sweep")
    width_parser.add_argument("--model", type=str, default="mlp")
    width_parser.add_argument("--optimizer", type=str, default="muon_sgd")
    width_parser.add_argument("--inner-solver", type=str, default="spectral_clip")
    width_parser.add_argument("--spectral-budget", type=float, default=0.1)
    width_parser.add_argument("--lr", type=float, default=0.1)
    width_parser.add_argument("--epochs", type=int, default=30)
    width_parser.add_argument("--widths", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.5, 2.0])
    width_parser.add_argument("--seed", type=int, default=0)
    width_parser.add_argument("--logdir", type=str, default="logs")

    # Optimizer comparison
    compare_parser = subparsers.add_parser("compare_optimizers", help="Compare optimizers")
    compare_parser.add_argument("--model", type=str, default="resnet18")
    compare_parser.add_argument("--epochs", type=int, default=50)
    compare_parser.add_argument("--seed", type=int, default=0)
    compare_parser.add_argument("--logdir", type=str, default="logs")

    args = parser.parse_args()

    if args.command == "multi_seed":
        multi_seed_experiment(
            model=args.model,
            optimizer=args.optimizer,
            inner_solver=args.inner_solver,
            spectral_budget=args.spectral_budget,
            lr=args.lr,
            epochs=args.epochs,
            num_seeds=args.seeds,
            logdir=args.logdir,
        )
    elif args.command == "lr_scan":
        lr_stability_scan(
            model=args.model,
            optimizer=args.optimizer,
            inner_solver=args.inner_solver,
            spectral_budget=args.spectral_budget,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            num_lrs=args.num_lrs,
            logdir=args.logdir,
        )
    elif args.command == "width_sweep":
        width_transfer_sweep(
            model=args.model,
            optimizer=args.optimizer,
            inner_solver=args.inner_solver,
            spectral_budget=args.spectral_budget,
            lr=args.lr,
            epochs=args.epochs,
            width_mults=args.widths,
            seed=args.seed,
            logdir=args.logdir,
        )
    elif args.command == "compare_optimizers":
        compare_optimizers(
            model=args.model,
            epochs=args.epochs,
            seed=args.seed,
            logdir=args.logdir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

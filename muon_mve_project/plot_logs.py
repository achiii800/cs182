#!/usr/bin/env python3
"""
Publication-ready plotting utilities for experiment logs.

Features:
    - Overlay multiple runs (different optimizers, solvers, architectures)
    - Error bars from multiple seeds
    - Customizable axis labels, log/linear scales
    - Generate individual or combined plots

Usage:
    # Single run
    python plot_logs.py --logfiles logs/exp1.csv --output-dir plots/

    # Compare multiple runs
    python plot_logs.py --logfiles logs/sgd.csv logs/muon_sgd.csv --labels SGD MuonSGD

    # Aggregate seeds
    python plot_logs.py --logfiles logs/exp_seed*.csv --aggregate-seeds --output-dir plots/

    # Custom plot types
    python plot_logs.py --logfiles logs/*.csv --plot-types loss acc spectral sharpness
"""

import argparse
import os
import glob
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Style settings for publication-quality plots
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_logs(paths: List[str]) -> List[pd.DataFrame]:
    """Load CSV log files into DataFrames."""
    dfs = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            df["source"] = Path(path).stem
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    return dfs


def aggregate_seeds(dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate multiple seed runs into mean and std DataFrames.

    Assumes all DataFrames have the same columns and number of epochs.
    """
    if len(dfs) == 0:
        raise ValueError("No DataFrames to aggregate")

    # Stack all DataFrames
    combined = pd.concat(dfs, ignore_index=True)

    # Group by epoch and compute statistics
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "epoch"]

    mean_df = combined.groupby("epoch")[numeric_cols].mean().reset_index()
    std_df = combined.groupby("epoch")[numeric_cols].std().reset_index()

    return mean_df, std_df


def plot_metric(
    ax: plt.Axes,
    dfs: List[pd.DataFrame],
    x_col: str,
    y_col: str,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    std_dfs: Optional[List[pd.DataFrame]] = None,
    y_log: bool = False,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    alpha_fill: float = 0.2,
):
    """
    Plot a single metric across multiple DataFrames.

    Args:
        ax: Matplotlib axes.
        dfs: List of DataFrames with the data.
        x_col, y_col: Column names for x and y axes.
        labels: Legend labels for each DataFrame.
        colors: Line colors.
        linestyles: Line styles.
        std_dfs: Optional list of DataFrames with std for error bands.
        y_log: Use log scale for y-axis.
        x_label, y_label, title: Axis labels and title.
        alpha_fill: Transparency for error bands.
    """
    if colors is None:
        colors = list(mcolors.TABLEAU_COLORS.values())
    if linestyles is None:
        linestyles = ["-"] * len(dfs)
    if labels is None:
        labels = [df.get("source", [f"Run {i}"])[0] if "source" in df.columns else f"Run {i}"
                  for i, df in enumerate(dfs)]

    for i, df in enumerate(dfs):
        if y_col not in df.columns:
            print(f"Warning: Column '{y_col}' not found in DataFrame {i}")
            continue

        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        label = labels[i] if i < len(labels) else f"Run {i}"

        x = df[x_col].values
        y = df[y_col].values

        ax.plot(x, y, color=color, linestyle=linestyle, label=label, linewidth=1.5)

        # Error bands
        if std_dfs is not None and i < len(std_dfs):
            std_df = std_dfs[i]
            if y_col in std_df.columns:
                y_std = std_df[y_col].values
                ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=alpha_fill)

    if y_log:
        ax.set_yscale("log")

    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    if title:
        ax.set_title(title)
    ax.legend()


def plot_loss_curves(
    dfs: List[pd.DataFrame],
    labels: Optional[List[str]] = None,
    std_dfs: Optional[List[pd.DataFrame]] = None,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """Plot train and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_metric(
        axes[0], dfs, "epoch", "train_loss",
        labels=labels, std_dfs=std_dfs,
        x_label="Epoch", y_label="Loss", title="Training Loss",
    )

    plot_metric(
        axes[1], dfs, "epoch", "val_loss",
        labels=labels, std_dfs=std_dfs,
        x_label="Epoch", y_label="Loss", title="Validation Loss",
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_accuracy_curves(
    dfs: List[pd.DataFrame],
    labels: Optional[List[str]] = None,
    std_dfs: Optional[List[pd.DataFrame]] = None,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """Plot train and validation accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_metric(
        axes[0], dfs, "epoch", "train_acc",
        labels=labels, std_dfs=std_dfs,
        x_label="Epoch", y_label="Accuracy", title="Training Accuracy",
    )

    plot_metric(
        axes[1], dfs, "epoch", "val_acc",
        labels=labels, std_dfs=std_dfs,
        x_label="Epoch", y_label="Accuracy", title="Validation Accuracy",
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_spectral_norms(
    dfs: List[pd.DataFrame],
    labels: Optional[List[str]] = None,
    std_dfs: Optional[List[pd.DataFrame]] = None,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """Plot maximum spectral norm trajectory."""
    fig, ax = plt.subplots(figsize=(6, 4))

    plot_metric(
        ax, dfs, "epoch", "max_spectral_norm",
        labels=labels, std_dfs=std_dfs,
        x_label="Epoch", y_label="Max σ (spectral norm)",
        title="Maximum Spectral Norm Trajectory",
    )

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_sharpness(
    dfs: List[pd.DataFrame],
    labels: Optional[List[str]] = None,
    std_dfs: Optional[List[pd.DataFrame]] = None,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """Plot sharpness proxy trajectory."""
    fig, ax = plt.subplots(figsize=(6, 4))

    plot_metric(
        ax, dfs, "epoch", "sharpness",
        labels=labels, std_dfs=std_dfs,
        x_label="Epoch", y_label="Sharpness (SAM proxy)",
        title="Sharpness Trajectory",
    )

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_gns(
    dfs: List[pd.DataFrame],
    labels: Optional[List[str]] = None,
    std_dfs: Optional[List[pd.DataFrame]] = None,
    output_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = True,
):
    """Plot gradient noise scale trajectory."""
    fig, ax = plt.subplots(figsize=(6, 4))

    plot_metric(
        ax, dfs, "epoch", "grad_noise_scale",
        labels=labels, std_dfs=std_dfs,
        y_log=log_scale,
        x_label="Epoch", y_label="Gradient Noise Scale",
        title="GNS Trajectory",
    )

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_combined_metrics(
    dfs: List[pd.DataFrame],
    labels: Optional[List[str]] = None,
    std_dfs: Optional[List[pd.DataFrame]] = None,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """Plot all metrics in a single figure (2x3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    metrics = [
        ("train_loss", "Training Loss", False),
        ("val_loss", "Validation Loss", False),
        ("val_acc", "Validation Accuracy", False),
        ("max_spectral_norm", "Max Spectral Norm", False),
        ("sharpness", "Sharpness", False),
        ("grad_noise_scale", "Gradient Noise Scale", True),
    ]

    for ax, (col, title, log_y) in zip(axes.flatten(), metrics):
        plot_metric(
            ax, dfs, "epoch", col,
            labels=labels, std_dfs=std_dfs,
            y_log=log_y,
            x_label="Epoch", y_label=title, title=title,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_lr_envelope(
    lr_values: List[float],
    final_losses: List[float],
    converged: List[bool],
    max_stable_lr: float,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """Plot learning rate stability envelope."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Separate converged and diverged points
    lr_conv = [lr for lr, c in zip(lr_values, converged) if c]
    loss_conv = [loss for loss, c in zip(final_losses, converged) if c]
    lr_div = [lr for lr, c in zip(lr_values, converged) if not c]
    loss_div = [loss for loss, c in zip(final_losses, converged) if not c]

    ax.scatter(lr_conv, loss_conv, c="green", label="Converged", s=60, zorder=3)
    ax.scatter(lr_div, [min(loss_conv) if loss_conv else 1.0] * len(lr_div),
               c="red", marker="x", label="Diverged", s=60, zorder=3)

    ax.axvline(max_stable_lr, color="blue", linestyle="--",
               label=f"Max stable LR: {max_stable_lr:.4f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Final Loss")
    ax.set_title("Learning Rate Stability Envelope")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_width_transfer(
    results: Dict[float, Dict[str, float]],
    metric: str = "val_acc",
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot muP-style width transfer results.

    Args:
        results: Dict mapping width_mult -> {metric: value} for each width.
        metric: Metric to plot (e.g., 'val_acc', 'val_loss').
    """
    widths = sorted(results.keys())
    values = [results[w].get(metric, 0) for w in widths]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(widths, values, "o-", markersize=8, linewidth=2)

    ax.set_xlabel("Width Multiplier")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"muP Transfer: {metric.replace('_', ' ').title()} vs Width")
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot experiment logs.")
    parser.add_argument(
        "--logfiles", type=str, nargs="+", required=True,
        help="Path(s) to CSV log files (supports glob patterns)."
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="Labels for each log file in the legend."
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots",
        help="Directory to save plots."
    )
    parser.add_argument(
        "--aggregate-seeds", action="store_true",
        help="Aggregate multiple seed runs into mean ± std."
    )
    parser.add_argument(
        "--plot-types", type=str, nargs="+",
        default=["loss", "acc", "spectral", "sharpness", "gns", "combined"],
        choices=["loss", "acc", "spectral", "sharpness", "gns", "combined", "all"],
        help="Types of plots to generate."
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    parser.add_argument("--format", type=str, default="png", help="Output format (png, pdf, svg).")

    args = parser.parse_args()

    # Expand glob patterns
    all_paths = []
    for pattern in args.logfiles:
        expanded = glob.glob(pattern)
        if not expanded:
            print(f"Warning: No files match pattern '{pattern}'")
        all_paths.extend(expanded)

    if not all_paths:
        print("Error: No log files found.")
        return

    print(f"Found {len(all_paths)} log file(s)")

    # Load data
    dfs = load_logs(all_paths)

    if not dfs:
        print("Error: Could not load any log files.")
        return

    # Aggregate seeds if requested
    std_dfs = None
    if args.aggregate_seeds and len(dfs) > 1:
        print("Aggregating across seeds...")
        mean_df, std_df = aggregate_seeds(dfs)
        dfs = [mean_df]
        std_dfs = [std_df]
        if args.labels is None:
            args.labels = ["Mean ± Std"]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle "all" plot type
    if "all" in args.plot_types:
        args.plot_types = ["loss", "acc", "spectral", "sharpness", "gns", "combined"]

    # Generate plots
    if "loss" in args.plot_types:
        plot_loss_curves(
            dfs, labels=args.labels, std_dfs=std_dfs,
            output_path=os.path.join(args.output_dir, f"loss_curves.{args.format}"),
            show=args.show,
        )

    if "acc" in args.plot_types:
        plot_accuracy_curves(
            dfs, labels=args.labels, std_dfs=std_dfs,
            output_path=os.path.join(args.output_dir, f"accuracy_curves.{args.format}"),
            show=args.show,
        )

    if "spectral" in args.plot_types:
        plot_spectral_norms(
            dfs, labels=args.labels, std_dfs=std_dfs,
            output_path=os.path.join(args.output_dir, f"spectral_norms.{args.format}"),
            show=args.show,
        )

    if "sharpness" in args.plot_types:
        plot_sharpness(
            dfs, labels=args.labels, std_dfs=std_dfs,
            output_path=os.path.join(args.output_dir, f"sharpness.{args.format}"),
            show=args.show,
        )

    if "gns" in args.plot_types:
        plot_gns(
            dfs, labels=args.labels, std_dfs=std_dfs,
            output_path=os.path.join(args.output_dir, f"gns.{args.format}"),
            show=args.show,
        )

    if "combined" in args.plot_types:
        plot_combined_metrics(
            dfs, labels=args.labels, std_dfs=std_dfs,
            output_path=os.path.join(args.output_dir, f"combined_metrics.{args.format}"),
            show=args.show,
        )

    print(f"\nPlots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

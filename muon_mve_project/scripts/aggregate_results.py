#!/usr/bin/env python3
"""
Aggregate results from multiple seed runs.

Usage:
    python scripts/aggregate_results.py --pattern "logs/exp_seed*.csv" --output results/aggregated.json
"""

import argparse
import glob
import json
import os
from typing import List, Dict, Any

import pandas as pd
import numpy as np


def aggregate_csv_logs(patterns: List[str]) -> Dict[str, Any]:
    """
    Aggregate CSV logs matching the given patterns.

    Returns:
        Dict with 'mean', 'std', 'min', 'max' statistics for each metric.
    """
    # Expand patterns
    all_paths = []
    for pattern in patterns:
        all_paths.extend(glob.glob(pattern))

    if not all_paths:
        raise ValueError(f"No files found matching patterns: {patterns}")

    print(f"Found {len(all_paths)} log files")

    # Load all DataFrames
    dfs = []
    for path in all_paths:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    if not dfs:
        raise ValueError("Could not load any log files")

    # Combine and compute statistics
    combined = pd.concat(dfs, keys=range(len(dfs)))

    # Get numeric columns (excluding epoch)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    if "epoch" in numeric_cols:
        numeric_cols.remove("epoch")

    # Group by epoch and compute statistics
    grouped = combined.groupby(level=1)

    stats = {
        "num_seeds": len(dfs),
        "num_epochs": int(combined.index.get_level_values(1).max()) + 1,
        "metrics": {},
    }

    for col in numeric_cols:
        mean_vals = grouped[col].mean().tolist()
        std_vals = grouped[col].std().tolist()
        min_vals = grouped[col].min().tolist()
        max_vals = grouped[col].max().tolist()

        stats["metrics"][col] = {
            "mean": mean_vals,
            "std": std_vals,
            "min": min_vals,
            "max": max_vals,
            "final_mean": mean_vals[-1] if mean_vals else None,
            "final_std": std_vals[-1] if std_vals else None,
        }

    # Best metrics
    if "val_acc" in stats["metrics"]:
        val_acc_mean = stats["metrics"]["val_acc"]["mean"]
        stats["best_val_acc_mean"] = max(val_acc_mean)
        stats["best_val_acc_epoch"] = val_acc_mean.index(max(val_acc_mean))

    return stats


def aggregate_summaries(patterns: List[str]) -> Dict[str, Any]:
    """
    Aggregate summary JSON files from multiple seeds.

    Returns:
        Dict with mean ± std for each summary metric.
    """
    # Expand patterns
    all_paths = []
    for pattern in patterns:
        all_paths.extend(glob.glob(pattern))

    if not all_paths:
        raise ValueError(f"No files found matching patterns: {patterns}")

    print(f"Found {len(all_paths)} summary files")

    # Load all summaries
    summaries = []
    for path in all_paths:
        try:
            with open(path) as f:
                summaries.append(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    if not summaries:
        raise ValueError("Could not load any summary files")

    # Aggregate
    all_keys = set()
    for s in summaries:
        all_keys.update(s.keys())

    stats = {"num_seeds": len(summaries)}

    for key in all_keys:
        values = [s.get(key) for s in summaries if key in s and isinstance(s.get(key), (int, float))]
        if values:
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }

    return stats


def print_summary_table(stats: Dict[str, Any]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 60)
    print("Aggregated Results")
    print("=" * 60)
    print(f"Number of seeds: {stats.get('num_seeds', 'N/A')}")

    if "metrics" in stats:
        # CSV log stats
        print(f"Number of epochs: {stats.get('num_epochs', 'N/A')}")
        print("\nFinal Epoch Metrics (mean ± std):")
        print("-" * 40)

        for metric, values in stats["metrics"].items():
            if values["final_mean"] is not None:
                mean = values["final_mean"]
                std = values["final_std"] if values["final_std"] else 0
                print(f"  {metric:25s}: {mean:.4f} ± {std:.4f}")

        if "best_val_acc_mean" in stats:
            print(f"\n  Best Val Acc (mean): {stats['best_val_acc_mean']:.4f} "
                  f"(epoch {stats['best_val_acc_epoch']})")
    else:
        # Summary JSON stats
        print("\nMetrics (mean ± std):")
        print("-" * 40)

        for key, values in stats.items():
            if isinstance(values, dict) and "mean" in values:
                print(f"  {key:25s}: {values['mean']:.4f} ± {values['std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experiment results.")
    parser.add_argument(
        "--pattern", type=str, nargs="+", required=True,
        help="Glob pattern(s) for log files (CSV or JSON)."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for aggregated results (JSON)."
    )
    parser.add_argument(
        "--format", type=str, default="auto", choices=["auto", "csv", "json"],
        help="Input file format (auto-detect by default)."
    )

    args = parser.parse_args()

    # Determine format
    if args.format == "auto":
        sample_files = []
        for p in args.pattern:
            sample_files.extend(glob.glob(p))
        if sample_files:
            if sample_files[0].endswith(".csv"):
                args.format = "csv"
            else:
                args.format = "json"
        else:
            print("Error: No files found")
            return

    # Aggregate
    if args.format == "csv":
        stats = aggregate_csv_logs(args.pattern)
    else:
        stats = aggregate_summaries(args.pattern)

    # Print
    print_summary_table(stats)

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()

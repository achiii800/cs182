
"""
Quick-and-dirty plotting script for logs produced by train_cifar_muon.py.

Usage:
    python plot_logs.py --logfile logs/cifar_muon_seed0.csv

You can also pass multiple --logfile arguments to overlay runs (e.g. different
inner solvers or seeds).
"""

import argparse
import csv
from typing import List

import matplotlib.pyplot as plt


def load_log(path: str):
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    max_spec = []
    sharpness = []
    gns = []
    lrs = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))
            max_spec.append(float(row["max_spectral_norm"]))
            sharpness.append(float(row["sharpness"]))
            gns.append(float(row["grad_noise_scale"]))
            lrs.append(float(row["lr"]))
    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "max_spec": max_spec,
        "sharpness": sharpness,
        "gns": gns,
        "lr": lrs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logfile",
        type=str,
        action="append",
        required=True,
        help="Path to a CSV log file (can be passed multiple times).",
    )
    args = parser.parse_args()

    logs = [load_log(p) for p in args.logfile]

    # Loss
    plt.figure()
    for path, log in zip(args.logfile, logs):
        plt.plot(log["epochs"], log["train_loss"], label=f"{path} train")
        plt.plot(log["epochs"], log["val_loss"], linestyle="--", label=f"{path} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=200)

    # Accuracy
    plt.figure()
    for path, log in zip(args.logfile, logs):
        plt.plot(log["epochs"], log["train_acc"], label=f"{path} train")
        plt.plot(log["epochs"], log["val_acc"], linestyle="--", label=f"{path} val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Val Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curves.png", dpi=200)

    # Spectral norm
    plt.figure()
    for path, log in zip(args.logfile, logs):
        plt.plot(log["epochs"], log["max_spec"], label=path)
    plt.xlabel("Epoch")
    plt.ylabel("Max spectral norm")
    plt.title("Max spectral norm trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectral_norms.png", dpi=200)

    # Sharpness
    plt.figure()
    for path, log in zip(args.logfile, logs):
        plt.plot(log["epochs"], log["sharpness"], label=path)
    plt.xlabel("Epoch")
    plt.ylabel("Sharpness proxy")
    plt.title("Sharpness trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sharpness.png", dpi=200)

    # GNS
    plt.figure()
    for path, log in zip(args.logfile, logs):
        plt.plot(log["epochs"], log["gns"], label=path)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient noise scale (proxy)")
    plt.title("GNS trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gns.png", dpi=200)

    print("Saved: loss_curves.png, accuracy_curves.png, spectral_norms.png, sharpness.png, gns.png")


if __name__ == "__main__":
    main()

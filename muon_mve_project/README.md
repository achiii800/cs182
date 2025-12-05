# Manifold Muon: Spectral-Norm Constrained Optimization

**EECS 182 Final Project - Category 1: Optimizers & Hyperparameter Transfer**

This project implements and evaluates spectral-norm constrained optimizers inspired by **Manifold Muon** (Bernstein, 2025). We explore how constraining the spectral norm of weight updates affects training stability, generalization, and hyperparameter transfer across model widths.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
- [Results](#results)
- [Mathematical Background](#mathematical-background)
- [Citation](#citation)

## 🎯 Overview

### The Problem

Standard optimizers like SGD and Adam allow unbounded growth of weight matrix spectral norms, which can lead to:
- Exploding/vanishing gradients
- Training instability
- Difficulty in hyperparameter tuning across model scales

### Our Approach

We implement **MuonSGD**, an SGD variant that constrains the spectral norm of updates:

$$\min_{A} \langle G, A \rangle \quad \text{s.t.} \quad \|A\|_2 \leq \eta$$

This ensures that each update's effect on layer outputs is bounded, providing:
- Better spectral norm control during training
- Potential for hyperparameter transfer across widths (muP-style)
- Multiple inner solver options with different trade-offs

### Inner Solvers Implemented

| Solver | Method | Complexity | Key Property |
|--------|--------|------------|--------------|
| `SpectralClip` | Simple rescaling | O(SVD) | Fast baseline |
| `DualAscent` | Lagrangian dual | O(k × SVD) | Warm-starting |
| `QuasiNewton` | L-BFGS on dual | O(k × SVD) | Faster convergence |
| `FrankWolfe` | Conditional gradient | O(k × top-SVD) | Low-rank updates |
| `ADMM` | Splitting method | O(k × SVD) | Adaptive ρ |

## 🔧 Installation

```bash
# Clone or download the project
cd muon_mve_project

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy, matplotlib, pandas, scipy
- tqdm (for notebooks)
- einops (optional, for some models)

## 🚀 Quick Start

### Command Line

```bash
# Basic training with MuonSGD
python train.py --model small_cnn --optimizer muon_sgd --inner-solver spectral_clip --epochs 10

# ResNet-18 with dual ascent solver
python train.py --model resnet18 --optimizer muon_sgd --inner-solver dual_ascent \
                --spectral-budget 0.1 --lr 0.1 --epochs 50

# Compare optimizers
python scripts/run_experiments.py compare_optimizers --model resnet18 --epochs 30

# Width transfer experiment
python scripts/run_experiments.py width_sweep --model mlp --optimizer muon_sgd
```

### Python API

```python
from muon import MuonSGD, get_inner_solver
from models import get_model

# Create model
model = get_model('resnet18', num_classes=10, width_mult=1.0)

# Create optimizer with spectral constraints
solver = get_inner_solver('dual_ascent', max_iters=20, warm_start=True)
optimizer = MuonSGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    spectral_budget=0.1,
    inner_solver=solver
)

# Training loop
for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()  # Inner solver enforces spectral constraint
```

### Notebooks

For interactive exploration, see the Jupyter notebooks:

1. **`notebooks/00_quickstart.ipynb`** - Colab-ready demo
2. **`notebooks/01_theory_and_motivation.ipynb`** - Mathematical background
3. **`notebooks/02_experiments.ipynb`** - Run all experiments
4. **`notebooks/03_analysis.ipynb`** - Analyze results, generate figures

## 📁 Project Structure

```
muon_mve_project/
├── muon/                      # Core optimizer module
│   ├── __init__.py
│   ├── inner_solvers.py       # SpectralClip, FrankWolfe, DualAscent, etc.
│   ├── muon_sgd.py            # MuonSGD and MuonAdamW optimizers
│   └── metrics.py             # Spectral norms, sharpness, GNS, Hessian
│
├── models/                    # Model architectures
│   └── __init__.py            # SmallCNN, ResNet18, TinyViT, MLPMixer, MLP
│
├── scripts/                   # Experiment automation
│   ├── run_experiments.py     # Multi-seed, LR scan, width sweep
│   └── aggregate_results.py   # Aggregate multi-seed results
│
├── notebooks/                 # Jupyter notebooks
│   ├── 00_quickstart.ipynb    # Quick demo (Colab-ready)
│   ├── 01_theory_and_motivation.ipynb
│   ├── 02_experiments.ipynb
│   └── 03_analysis.ipynb
│
├── train.py                   # Main training script
├── plot_logs.py               # Publication-ready plotting
├── requirements.txt
└── README.md
```

## 🧪 Experiments

### Core Experiments

1. **Baseline Comparison**
   ```bash
   python train.py --model small_cnn --optimizer sgd --epochs 20
   python train.py --model small_cnn --optimizer muon_sgd --inner-solver spectral_clip --epochs 20
   ```

2. **Inner Solver Comparison**
   ```bash
   for solver in spectral_clip dual_ascent quasi_newton frank_wolfe admm; do
       python train.py --model resnet18 --optimizer muon_sgd --inner-solver $solver \
                       --spectral-budget 0.1 --epochs 50 --seed 42
   done
   ```

3. **Width Transfer (muP-style)**
   ```bash
   python scripts/run_experiments.py width_sweep --model mlp --optimizer muon_sgd \
                                                  --widths 0.5 0.75 1.0 1.5 2.0
   ```

4. **LR Stability Envelope**
   ```bash
   python scripts/run_experiments.py lr_scan --model small_cnn --optimizer muon_sgd
   ```

5. **Multi-Seed Runs**
   ```bash
   python scripts/run_experiments.py multi_seed --model resnet18 --optimizer muon_sgd --seeds 5
   ```

### Generating Plots

```bash
# Single experiment
python plot_logs.py --logfiles logs/exp.csv --output-dir plots/

# Compare multiple runs
python plot_logs.py --logfiles logs/sgd*.csv logs/muon*.csv --labels SGD MuonSGD

# Aggregate seeds with error bars
python plot_logs.py --logfiles logs/exp_seed*.csv --aggregate-seeds
```

## 📊 Results

### Key Findings

1. **Spectral Norm Control**: MuonSGD effectively bounds σ_max throughout training
2. **Comparable Accuracy**: Spectral constraints don't hurt final accuracy
3. **Width Transfer**: Fixed hyperparameters work better across widths with constraints
4. **Solver Trade-offs**: 
   - SpectralClip: Fast but crude
   - DualAscent: Good balance
   - FrankWolfe: Low-rank updates

### Metrics Tracked

- Training/validation loss and accuracy
- Maximum spectral norm (σ_max)
- SAM-style sharpness proxy
- Gradient noise scale (GNS/CBS proxy)
- Top Hessian eigenvalue (optional)

## 📐 Mathematical Background

### The Muon Inner Problem

Given gradient G, find optimal update A:

$$\min_{A \in \mathbb{R}^{m \times n}} \text{trace}(G^T A) \quad \text{s.t.} \quad \|A\|_2 \leq \eta$$

**Closed-form solution** (spectral norm ball):
$$A^* = -\eta \cdot u_1 v_1^T$$

where u₁, v₁ are top singular vectors of G.

### With Tangency Constraint (Manifold Muon)

For Stiefel manifold (W^T W = I):

$$\min_A \langle G, A \rangle \quad \text{s.t.} \quad \|A\|_2 \leq \eta, \quad A^T W + W^T A = 0$$

Solved via:
- **Dual Ascent**: Lagrangian relaxation
- **ADMM**: Splitting tangency and spectral constraints
- **Frank-Wolfe**: Projection-free with rank-1 atoms

### Key Insight

The spectral budget η directly bounds worst-case output change:

$$\|(W + A)x - Wx\| = \|Ax\| \leq \|A\|_2 \|x\| \leq \eta$$

This provides a **width-independent interpretation** of the step size, enabling muP-style hyperparameter transfer.

## 📚 References

1. Bernstein, J. (2025). *Modular Manifolds*. Thinking Machines Blog.
2. Jordan, K. (2024). *Muon: MomentUm Orthogonalized by Newton-Schulz*.
3. Yang, G. & Hu, E. (2021). *Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks*.
4. Foret, P. et al. (2021). *Sharpness-Aware Minimization for Efficiently Improving Generalization*.

## 📄 License

GPL-3.0 License - see LICENSE file.

---

**EECS 182 - Deep Learning for Visual Data**  
*Jeshwanth Mohan, Fantine Mpacko Priso, Akshay Rao*

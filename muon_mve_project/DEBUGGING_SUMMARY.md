# Debugging Summary & Quick Reference

## What Was Wrong

Your notebook was failing because of three issues:

1. **Wrong script name**: Called `train_cifar_muon.py` but the actual script is **`train.py`**

2. **Wrong argument format**: Used `--inner_solver` (underscores) but the script uses **`--inner-solver`** (dashes)

3. **Missing setup**: No `os.chdir()` to project root, no module path setup

## Exit Code 512 Explained

Exit code 512 = Python couldn't find the script file. This is `2 * 256` where 2 is the shell error for "file not found".

---

## Corrected Command Format

```bash
# WRONG (what you had):
python train_cifar_muon.py --arch small_cnn --optimizer muon_sgd --inner_solver spectral_clip

# CORRECT:
python train.py --model small_cnn --optimizer muon_sgd --inner-solver spectral_clip
```

### Key argument differences:

| Your Command | Correct Command |
|-------------|-----------------|
| `train_cifar_muon.py` | `train.py` |
| `--arch` | `--model` |
| `--inner_solver` | `--inner-solver` |
| `--log_path file.csv` | `--logdir dir --exp-name name` |

---

## Quick Test Commands

Run these from the `muon_mve_project` directory:

```bash
# Quick sanity check (5 epochs)
python train.py --model small_cnn --optimizer muon_sgd --inner-solver spectral_clip \
    --spectral-budget 0.1 --lr 0.01 --epochs 5 --logdir logs/test --exp-name test1

# Check that the log was created
cat logs/test/test1.csv
```

---

## Running Experiments

### Experiment 1: Inner Solver Comparison (Quick, ~10 min)

```bash
for solver in spectral_clip dual_ascent quasi_newton frank_wolfe admm; do
    python train.py --model small_cnn --optimizer muon_sgd \
        --inner-solver $solver --spectral-budget 1.0 --lr 0.01 --epochs 20 \
        --logdir logs/exp1 --exp-name solver_$solver
done
```

### Experiment 2: Multi-Seed Baselines (~2-3 hours on GPU)

```bash
for seed in 0 1 2; do
    # SGD baseline
    python train.py --model resnet18 --optimizer sgd --inner-solver none \
        --lr 0.1 --epochs 50 --seed $seed --logdir logs/exp2 --exp-name sgd_seed$seed
    
    # MuonSGD with DualAscent
    python train.py --model resnet18 --optimizer muon_sgd --inner-solver dual_ascent \
        --spectral-budget 0.1 --lr 0.1 --epochs 50 --seed $seed \
        --logdir logs/exp2 --exp-name muon_dual_seed$seed
done
```

### Experiment 3: Width Transfer (~1 hour on GPU)

```bash
for width in 0.5 0.75 1.0 1.5 2.0; do
    for seed in 0 1 2; do
        python train.py --model mlp --width-mult $width --optimizer muon_sgd \
            --inner-solver dual_ascent --spectral-budget 0.1 --lr 0.01 \
            --epochs 30 --seed $seed --logdir logs/exp3 --exp-name muon_w${width}_s${seed}
    done
done
```

---

## Log File Format

Logs are saved as CSV with columns:
```
epoch,train_loss,train_acc,val_loss,val_acc,max_spectral_norm,sharpness,grad_noise_scale,hessian_eig,lr,time
```

Additionally, `{exp_name}_config.json` and `{exp_name}_summary.json` are created.

---

## Colab Setup (if using Google Colab)

1. Upload `muon_mve_project` folder to Drive
2. Use this setup cell:

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys
PROJECT_ROOT = '/content/drive/MyDrive/cs182/muon_mve_project'  # Adjust path
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

!pip install torch torchvision numpy matplotlib scipy pandas tqdm einops -q
```

---

## New Notebook Created

I created a fully working notebook at:
```
notebooks/EECS182_Experiments_Colab.ipynb
```

This notebook:
- Handles Colab/local detection automatically
- Uses correct script name and arguments
- Runs all 3 experiments with proper logging
- Generates publication-quality plots with error bars

---

## Top 5 Experiments to Run First

| # | Experiment | Command | Expected Time | Purpose |
|---|------------|---------|---------------|---------|
| 1 | **Quick sanity check** | 5 epochs SmallCNN | 1 min | Verify everything works |
| 2 | **Exp1: Solver ablation** | 20 epochs SmallCNN × 5 solvers | 15 min | Compare solver performance |
| 3 | **Exp2: SGD baseline** | 50 epochs ResNet-18, 3 seeds | 45 min | Error bars for paper |
| 4 | **Exp2: MuonSGD** | 50 epochs ResNet-18, 3 seeds | 45 min | Main result |
| 5 | **Exp3: Width transfer** | 30 epochs MLP, 5 widths × 3 seeds | 1 hour | muP-style transfer |

---

## What to Look For in Results

### Experiment 1 (Solver Comparison)
- **DualAscent**: Should have best accuracy (your pilot showed 40.4%)
- **SpectralClip**: Fast baseline, good but slightly lower
- **FrankWolfe**: Lowest accuracy but fastest (rank-1 updates)
- **σ_max trajectories**: All should stay bounded near budget

### Experiment 2 (Baselines with Error Bars)
- SGD: ~93-94% val acc on ResNet-18
- AdamW: ~94-95% val acc
- MuonSGD: Should be competitive with better spectral control
- **Key metric**: Variance across seeds (smaller = more stable)

### Experiment 3 (Width Transfer)
- **SGD**: Performance should degrade at extreme widths
- **MuonSGD**: Performance should be more stable across widths
- If MuonSGD shows flatter curve → spectral constraints help transfer

---

## Files You Have

```
muon_mve_project/
├── train.py                 # Main training script ← USE THIS
├── muon/
│   ├── __init__.py
│   ├── muon_sgd.py         # MuonSGD, MuonAdamW, create_optimizer
│   ├── inner_solvers.py    # 5 solvers + TangentSpaceProjector
│   └── metrics.py          # spectral norms, sharpness, GNS, etc.
├── models/
│   └── __init__.py         # SmallCNN, ResNet18, TinyViT, MLPMixer, MLP
├── notebooks/
│   └── EECS182_Experiments_Colab.ipynb  # NEW - use this!
├── scripts/
│   ├── run_priority1_experiments.sh
│   └── generate_paper_figures.py
└── logs/                   # Output directory
```

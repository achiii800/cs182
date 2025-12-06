# 🔧 Quick Fix for Colab

## The Problem

Your training failed with **exit code 1** because of two likely issues:

1. **`num-workers` > 0** causes multiprocessing errors in Colab
2. **CIFAR-10 data** may not be downloading to the expected location

## The Solution

I created a **new fixed notebook**: `notebooks/EECS182_Experiments_FIXED.ipynb`

This notebook:
- Runs training **inline** (no subprocess issues)
- Uses `num_workers=0` (required for Colab)
- Pre-downloads CIFAR-10 before training
- Has proper error handling and debug output

---

## How to Use in Colab

### Option 1: Push to GitHub, then pull in Colab

On your local machine:
```bash
cd /Users/akshayrao/Projects/cs182
git add -A
git commit -m "Add fixed Colab notebooks"
git push
```

Then in Colab:
```python
!git clone https://github.com/achiii800/cs182.git /content/cs182
# or if already cloned:
!cd /content/cs182 && git pull
```

Open: `/content/cs182/muon_mve_project/notebooks/EECS182_Experiments_FIXED.ipynb`

### Option 2: Manual Upload

1. Upload the entire `muon_mve_project` folder to Google Drive
2. In Colab, mount Drive and navigate to it:
```python
from google.colab import drive
drive.mount('/content/drive')
PROJECT_ROOT = '/content/drive/MyDrive/muon_mve_project'  # adjust path
```

---

## Key Fixes Made

### 1. Use `num_workers=0`
```python
# WRONG (fails in Colab):
train_loader = DataLoader(train_set, batch_size=128, num_workers=4)

# CORRECT:
train_loader = DataLoader(train_set, batch_size=128, num_workers=0)
```

### 2. Pre-download Data
```python
# Add this BEFORE training:
data_dir = './data'
torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
```

### 3. Inline Training (bypass subprocess)
The fixed notebook runs training directly in Python cells instead of calling `train.py` via subprocess. This shows actual errors and avoids multiprocessing issues.

---

## What Each Cell Does

| Cell | Purpose |
|------|---------|
| 1 | Clone repo and set PROJECT_ROOT |
| 2 | Install dependencies |
| 3 | **Pre-download CIFAR-10** |
| 4 | Debug imports (verify everything works) |
| 5 | Debug subprocess call (shows actual error) |
| 6 | **Inline training test** (if subprocess fails) |
| 7 | **Experiment 1: Solver comparison** |
| 8 | Plot Exp 1 results |
| 9 | **Experiment 2: Multi-seed baselines** |
| 10 | Plot Exp 2 with error bars |
| 11 | **Experiment 3: Width transfer** |
| 12 | Plot Exp 3 results |

---

## Quick Test Commands (Local)

If you want to test locally before Colab:

```bash
cd /Users/akshayrao/Projects/cs182/muon_mve_project

# Quick test with 0 workers
python train.py --model small_cnn --optimizer muon_sgd --inner-solver dual_ascent \
    --spectral-budget 0.1 --lr 0.01 --epochs 2 \
    --logdir logs/test --exp-name test --num-workers 0
```

---

## Files Created

```
muon_mve_project/
├── notebooks/
│   ├── EECS182_Experiments_Colab.ipynb     # Original (has subprocess issues)
│   └── EECS182_Experiments_FIXED.ipynb     # NEW - use this!
├── DEBUGGING_SUMMARY.md                     # This file
└── ...
```

---

## Estimated Run Times (Colab L4 GPU)

| Experiment | Time |
|------------|------|
| Exp 1: Solver comparison (5 solvers × 20 epochs) | ~15 min |
| Exp 2: Baselines (4 configs × 3 seeds × 50 epochs) | ~2-3 hours |
| Exp 3: Width transfer (2 configs × 5 widths × 3 seeds × 30 epochs) | ~1-2 hours |

---

## Next Steps

1. **Push the new notebook to GitHub**
2. **Pull in Colab** 
3. **Run Cell 6** (inline training test) first to verify everything works
4. **Run Cells 7-12** for the full experiments

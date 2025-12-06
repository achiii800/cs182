# Experiment Plan for EECS 182 Project

## Priority 1: Core Baseline Experiments (Run First)

These establish your baseline and should be run on the JupyterHub GPU. Each takes ~30-60 min on a single GPU.

### 1A. Optimizer Comparison on ResNet-18 (5 seeds)
**Goal**: Show that MuonSGD with spectral constraints is competitive with baselines

```bash
# On JupyterHub, run:
cd muon_mve_project

# SGD baseline (5 seeds)
for seed in 0 1 2 3 4; do
    python train.py --model resnet18 --optimizer sgd --lr 0.1 --epochs 100 \
                    --seed $seed --exp-name resnet18_sgd_seed${seed}
done

# AdamW baseline (5 seeds)
for seed in 0 1 2 3 4; do
    python train.py --model resnet18 --optimizer adamw --lr 1e-3 --epochs 100 \
                    --seed $seed --exp-name resnet18_adamw_seed${seed}
done

# MuonSGD with DualAscent (5 seeds) - your best inner solver
for seed in 0 1 2 3 4; do
    python train.py --model resnet18 --optimizer muon_sgd --inner-solver dual_ascent \
                    --spectral-budget 0.1 --lr 0.1 --epochs 100 \
                    --seed $seed --exp-name resnet18_muon_dual_seed${seed}
done

# MuonSGD with SpectralClip (5 seeds) - fast baseline
for seed in 0 1 2 3 4; do
    python train.py --model resnet18 --optimizer muon_sgd --inner-solver spectral_clip \
                    --spectral-budget 0.1 --lr 0.1 --epochs 100 \
                    --seed $seed --exp-name resnet18_muon_clip_seed${seed}
done
```

**Expected outcome**: 
- SGD: ~93-94% val acc
- AdamW: ~94-95% val acc  
- MuonSGD: Should be competitive (93-95%) with better spectral norm control

**Analysis to include in paper**:
- Training/val loss curves with error bands
- Final accuracy comparison (mean ± std)
- Spectral norm trajectories showing MuonSGD maintains tighter bounds
- Sharpness comparison (hypothesis: MuonSGD finds flatter minima)

---

### 1B. Inner Solver Ablation on ResNet-18 (3 seeds each)
**Goal**: Compare all 5 inner solvers to justify your choice

```bash
# Run all 5 solvers with 3 seeds each (15 total runs)
for solver in spectral_clip dual_ascent frank_wolfe quasi_newton admm; do
    for seed in 0 1 2; do
        python train.py --model resnet18 --optimizer muon_sgd --inner-solver $solver \
                        --spectral-budget 0.1 --lr 0.1 --epochs 50 \
                        --seed $seed --exp-name resnet18_solver_${solver}_seed${seed}
    done
done
```

**Expected outcomes**:
- DualAscent: Best accuracy (based on your SmallCNN results)
- SpectralClip: Fast but slightly lower accuracy
- FrankWolfe: Fastest wall-clock, low-rank updates (good for analysis)
- QuasiNewton: May underperform (as in draft) - interesting failure case
- ADMM: Should be robust, worth documenting

**Analysis**:
- Accuracy vs. wall-clock time trade-off plot
- Update rank analysis (FrankWolfe should show rank ~10-20, others full-rank)
- Inner iteration count statistics

---

## Priority 2: muP-style Width Transfer (Core Category 1 Experiment)

This is critical for Category 1. Tests if hyperparameters transfer across widths.

### 2A. MLP Width Sweep (5 widths, 3 seeds each)

```bash
# Fixed hyperparameters: LR=0.1, budget=0.1, 30 epochs
for width in 0.5 0.75 1.0 1.5 2.0; do
    for seed in 0 1 2; do
        python train.py --model mlp --optimizer muon_sgd --inner-solver dual_ascent \
                        --width-mult $width --lr 0.1 --spectral-budget 0.1 --epochs 30 \
                        --seed $seed --exp-name mlp_width${width}_muon_seed${seed}
    done
done

# Compare with SGD (same widths)
for width in 0.5 0.75 1.0 1.5 2.0; do
    for seed in 0 1 2; do
        python train.py --model mlp --optimizer sgd --width-mult $width \
                        --lr 0.1 --epochs 30 \
                        --seed $seed --exp-name mlp_width${width}_sgd_seed${seed}
    done
done
```

**Expected outcomes**:
- **If muP-like transfer works**: MuonSGD accuracy should be more stable across widths than SGD
- **If it doesn't**: SGD might outperform at some widths - still interesting!

**Analysis**:
- Plot: Accuracy vs. width_mult for both optimizers (with error bars)
- Plot: Spectral norm vs. width_mult (should scale predictably with MuonSGD)
- Table: Variance in accuracy across widths (lower = better transfer)

---

## Priority 3: Learning Rate Stability Analysis

### 3A. LR Envelope Scan

```bash
python scripts/run_experiments.py lr_scan --model resnet18 --optimizer muon_sgd \
                                          --inner-solver dual_ascent --num-lrs 15
                                          
python scripts/run_experiments.py lr_scan --model resnet18 --optimizer sgd \
                                          --num-lrs 15
```

**Analysis**:
- Plot max stable LR for each optimizer
- Hypothesis: MuonSGD should have wider stable LR range due to spectral constraint

---

## Priority 4: Extended Analysis (If Time Permits)

### 4A. Hessian Eigenvalue Tracking
Expensive but valuable. Run on 1-2 seeds only.

```bash
python train.py --model resnet18 --optimizer muon_sgd --inner-solver dual_ascent \
                --compute-hessian-eig --hessian-eig-interval 10 --epochs 50
```

### 4B. Tiny Transformer Experiment
Test on attention layers to show generality.

```bash
for seed in 0 1 2; do
    python train.py --model tiny_vit --optimizer muon_sgd --inner-solver dual_ascent \
                    --lr 0.01 --epochs 100 --seed $seed
done
```

---

## How to Run on JupyterHub

### Setup (First Time Only)

```bash
# In JupyterHub terminal
cd ~
git clone https://github.com/achiii800/cs182.git
cd cs182/muon_mve_project

# Create conda environment
conda create -n muon python=3.10 -y
conda activate muon
pip install -r requirements.txt
```

### Running Experiments

**Option 1: Interactive (for debugging)**
```bash
conda activate muon
python train.py --model small_cnn --optimizer muon_sgd --epochs 5
```

**Option 2: Background Jobs (for long runs)**
```bash
conda activate muon
nohup python train.py --model resnet18 --epochs 100 > train.log 2>&1 &
tail -f train.log  # Monitor progress
```

**Option 3: Batch Script (recommended)**
Create `run_baseline.sh`:
```bash
#!/bin/bash
conda activate muon
for seed in 0 1 2 3 4; do
    echo "Running SGD seed $seed"
    python train.py --model resnet18 --optimizer sgd --seed $seed --epochs 100
done
```

Then: `bash run_baseline.sh`

---

## Generating Plots for Paper

After experiments complete:

```bash
# Aggregate multi-seed runs
python plot_logs.py --logfiles logs/resnet18_sgd_seed*.csv \
                    --aggregate-seeds --labels "SGD" \
                    --output-dir plots/baselines

python plot_logs.py --logfiles logs/resnet18_muon_dual_seed*.csv \
                    --aggregate-seeds --labels "MuonSGD (DualAscent)" \
                    --output-dir plots/baselines

# Compare all optimizers
python plot_logs.py --logfiles logs/resnet18_*_seed0.csv \
                    --labels "SGD" "AdamW" "MuonSGD" \
                    --output-dir plots/comparison \
                    --plot-types combined

# Width transfer plot (will need custom script)
python scripts/plot_width_transfer.py --pattern "logs/mlp_width*" --output plots/width_transfer.pdf
```

---

## Estimated Compute Time

With EECS 182 JupyterHub GPU (likely V100 or A100):

| Experiment | Time per seed | Total | Priority |
|-----------|---------------|-------|----------|
| ResNet-18 (100 epochs) | ~45 min | 15 seeds = 11 hours | **P1** |
| Inner solver ablation (50 epochs) | ~25 min | 15 seeds = 6 hours | **P1** |
| MLP width sweep (30 epochs) | ~15 min | 30 runs = 7.5 hours | **P2** |
| LR scans | ~2 hours | 2 optimizers = 4 hours | **P3** |

**Total for Priority 1-2: ~29 hours** (can run in parallel if multiple GPUs, or overnight)

---

## Paper Writing Plan

While experiments run, work on:

1. **Literature Review Expansion** (2-3 pages):
   - muP and variants (u-muP, µnit scaling)
   - Spectral normalization history (Miyato 2018 → Muon 2024)
   - Shampoo/Muon optimizer family
   - Manifold optimization basics (Absil et al. 2008)

2. **Methods Section**:
   - Mathematical formulation is already excellent in draft
   - Add pseudocode for each inner solver
   - Justify hyperparameter choices (LR=0.1 is standard for SGD on CIFAR, budget=0.1 is X% of typical spectral norms)

3. **Results Section** (once experiments done):
   - Start with qualitative observations from plots
   - Then quantitative tables (mean ± std)
   - Discuss failure modes (why QuasiNewton underperforms)

4. **Related Work**:
   - Position vs. Shampoo (full-matrix vs. spectral-only)
   - Position vs. original Muon (you're studying inner solvers)
   - Difference from spectral normalization (update constraint vs. weight constraint)

---

## Troubleshooting Common Issues

**Issue: CUDA out of memory**
```bash
# Reduce batch size
python train.py --batch-size 64  # instead of default 128
```

**Issue: Experiments too slow**
```bash
# Use fewer epochs for ablations
python train.py --epochs 30  # instead of 100

# Or use smaller model first
python train.py --model small_cnn
```

**Issue: Can't find previous results**
```bash
# All logs are in logs/ directory
ls logs/
# Summaries:
cat logs/resnet18_sgd_seed0_summary.json
```

---

## Success Criteria

You'll have a strong Category 1 project if you complete:

✅ **Priority 1 experiments** (baselines + inner solver ablation)  
✅ **Priority 2 experiments** (width transfer)  
✅ **Plots with error bars** from multi-seed runs  
✅ **Expanded literature review** (2-3 pages)  
✅ **Analysis of results** explaining when/why each solver works  

**Stretch goals** (if time):
- Hessian eigenvalue analysis
- Transformer experiments
- Subspace drift metrics
- Custom muP scaling rules

Good luck! The code is in great shape - execution is the main task now.

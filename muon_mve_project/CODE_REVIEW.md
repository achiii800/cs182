# EECS 182 Project Code Review & Assessment
## Manifold Muon: Inner Solver Comparison for Spectral-Norm Constrained Optimization

**Date**: December 5, 2024  
**Reviewer**: Claude (Sonnet 4.5)  
**Project Members**: Akshay Rao, Jeshwanth Mohan, Fantine Mpacko Priso

---

## Executive Summary

**Overall Assessment**: ✅ **Strong foundation, ready for systematic experiments**

Your codebase is **exceptionally well-structured** for a student project. The implementation is clean, modular, and publication-ready. The main work ahead is **execution** - running the experiments systematically, collecting results with proper error bars, and writing up the analysis.

**Estimated Time to Completion**: 
- Experiments: ~30 hours GPU time (can parallelize)
- Writing: ~15-20 hours
- **Total**: 2-3 weeks if starting now

---

## Category 1 Requirements Checklist

| Requirement | Status | Evidence | Priority |
|------------|--------|----------|----------|
| **Clear research question** | ✅ Done | "Which inner solver works best for spectral-norm constrained optimization?" | - |
| **Literature review (2-3 pages)** | ⚠️ Needs expansion | Draft has 1 page, needs muP, Shampoo details | **HIGH** |
| **Systematic experiments** | ⚠️ Partial | SmallCNN done, need ResNet-18 + width sweeps | **HIGH** |
| **Meaningful error bars** | ❌ Missing | Only single-seed runs so far | **CRITICAL** |
| **Loss/accuracy curves** | ⚠️ Code ready | Plotting utilities exist, need actual multi-seed plots | **HIGH** |
| **Controlled comparisons** | ⚠️ Partial | Need SGD/AdamW baselines with same setup | **HIGH** |
| **Hyperparameter justification** | ⚠️ Partial | Need to document why LR=0.1, budget=0.1 | **MEDIUM** |
| **Width transfer (muP)** | ❌ Not done | Core Category 1 experiment | **CRITICAL** |
| **GitHub with replication** | ✅ Done | All code committed, requirements.txt ready | - |

---

## Code Quality Analysis

### ✅ Strengths

1. **Excellent Modularity**
   - Inner solvers as abstract base class → easy to add new methods
   - Clean separation: optimizer (`MuonSGD`) ↔ inner solver ↔ metrics
   - Model factory pattern makes architecture experiments trivial

2. **Comprehensive Metrics Suite**
   - Spectral norms ✅
   - SAM-style sharpness proxy ✅
   - Gradient noise scale ✅
   - Hessian eigenvalue (power iteration) ✅
   - Missing only: Subspace drift (implemented but unused)

3. **Production-Ready Experiment Infrastructure**
   - `scripts/run_experiments.py`: Multi-seed, LR scan, width sweep all automated
   - `plot_logs.py`: Publication-ready plotting with error bands
   - `aggregate_results.py`: Statistical aggregation across seeds
   - CSV logging for reproducibility

4. **Well-Documented**
   - Every function has clear docstrings
   - Mathematical formulations explained
   - Examples in README

### ⚠️ Areas for Improvement

#### 1. Inner Solver Implementation Issues

**QuasiNewtonDualSolver** (lines 340-430 in `inner_solvers.py`):
```python
# Current implementation has oversimplified dual objective
def _dual_objective_and_grad(self, lam, sigma_G, budget):
    dual_val = -budget * sigma_G - lam * budget  # Too simple!
    dual_grad = -budget  # Always negative
    return dual_val, dual_grad
```

**Problem**: For the spectral ball alone (no tangency), the solution is trivial - just project onto the ball boundary. The L-BFGS machinery adds unnecessary complexity.

**Fix Options**:
1. **Document limitation**: Add comment explaining this is most useful with tangency constraints
2. **Implement full dual**: Add tangency constraint to make L-BFGS meaningful
3. **Remove or simplify**: Replace with simpler projected gradient descent

**Recommendation**: Option 1 (document) + add tangency constraint as stretch goal.

---

#### 2. GNS Estimation Robustness

**Current** (`train.py` line 185):
```python
# Uses only 2 batches - noisy estimate
gns_batches = []
for i, (x, y) in enumerate(train_loader):
    gns_batches.append((x, y))
    if len(gns_batches) >= 2:  # Only 2!
        break
```

**Fix**:
```python
if len(gns_batches) >= 5:  # Use 5 batches
    break
```

**Impact**: More stable GNS estimates, critical for comparing optimizers.

---

#### 3. Tangent Space Projection Unused

**File**: `muon/inner_solvers.py`, lines 536-588

You implemented `TangentSpaceProjector` with QR and polar retractions, but it's never integrated into the main optimizer flow.

**Decision needed**:
- **Keep it**: Integrate into a `ManifoldMuonSGD` variant that enforces Stiefel constraints
- **Remove it**: Simplify to non-manifold spectral constraints only
- **Document it**: Explain it's for future work / full Muon implementation

**Recommendation**: Document as "future work" in comments and paper. Full manifold Muon is ambitious but out of scope for this project timeline.

---

#### 4. Missing Baselines

You should add a trivial "no solver" case for ablation:

```python
class NoOpSolver(BaseInnerSolver):
    """Pass-through solver that does nothing (baseline)."""
    def __call__(self, W, delta, spectral_budget):
        return delta  # Just return raw delta
```

This helps isolate the effect of spectral constraints.

---

## Experimental Progress Assessment

### ✅ What's Working (SmallCNN Results)

From your draft (Figure 1, Table 1):
- **DualAscent**: 40.4% accuracy, 425s
- **QuasiNewton**: 31.2% accuracy, 441s
- **FrankWolfe**: 29.2% accuracy, 354s (fastest)

**Key observations**:
1. DualAscent clearly wins on accuracy
2. FrankWolfe is fastest (rank-1 updates → sparse)
3. QuasiNewton underperforms - suggests L-BFGS overkill for this problem

**Why QuasiNewton might be failing**:
- Over-fitting to dual geometry (curvature hurts more than helps)
- Needs more inner iterations to converge
- Warm-starting not working as intended

**Diagnostic needed**: Add inner iteration count logging to all solvers.

---

### ❌ What's Missing

#### 1. **Multi-Seed Runs** (CRITICAL)
No error bars in current results. Need minimum 3 seeds per config, ideally 5.

**Action**: See `EXPERIMENT_PLAN.md` and `scripts/run_priority1_experiments.sh`

---

#### 2. **ResNet-18 Experiments** (CRITICAL)
SmallCNN is too toy - reviewers won't take seriously. ResNet-18 on CIFAR-10 is the standard benchmark.

**Expected ResNet-18 results** (literature baselines):
- SGD: 93-94% val acc
- AdamW: 94-95% val acc
- MuonSGD: Should be competitive with tighter spectral norm control

**Action**: Run Priority 1 experiments (see bash scripts)

---

#### 3. **Width Transfer (muP)** (CRITICAL for Category 1)
This is the core novelty claim: spectral constraints enable hyperparameter transfer.

**Hypothesis**: With spectral budget, LR and other hyperparameters should transfer better across widths than vanilla SGD.

**Test**: Fix LR=0.1, budget=0.1, train at widths [0.5, 0.75, 1.0, 1.5, 2.0], measure variance in accuracy.

**Success metric**: Lower variance across widths = better transfer.

**Action**: Run Priority 2 experiments (see bash scripts)

---

#### 4. **Baseline Comparisons**
Need clean SGD and AdamW baselines with same training setup:
- Same data augmentation
- Same epochs
- Same batch size
- Different LR (0.1 for SGD, 1e-3 for AdamW)

---

## Code Fixes Needed

### High Priority

1. **Fix GNS batch count** (Easy, 5 min)
   ```bash
   # File: train.py, line 185-190
   # Change: 2 → 5 batches
   ```

2. **Add diagnostic logging to inner solvers** (Medium, 30 min)
   ```python
   # In each solver's __call__, add:
   self._last_num_iters = num_iters  # Track convergence
   ```

3. **Document QuasiNewton limitations** (Easy, 10 min)
   ```python
   # Add to docstring:
   # "Note: For spectral ball alone, this reduces to simple projection.
   #  Most useful when combined with tangency constraints."
   ```

### Medium Priority

4. **Add NoOpSolver baseline** (Easy, 15 min)
   
5. **Fix Hessian estimation batching** (Medium, 20 min)
   ```python
   # Use consistent batch count across epochs
   # Currently inconsistent which batches are used
   ```

### Low Priority (Nice to Have)

6. **Implement full dual for QuasiNewton** (Hard, 2-3 hours)
7. **Add subspace drift tracking to training loop** (Medium, 1 hour)
8. **Integrate TangentSpaceProjector** (Hard, 3-4 hours)

---

## Paper Writing Guidance

### What to Add to Draft

#### 1. Literature Review Expansion (2-3 pages needed)

**Current**: 1 page covering basics  
**Needed**:
- **muP and variants** (1 page):
  - Original muP (Yang et al. 2021)
  - u-μP (Blake et al. 2024)
  - μnit scaling (Narayan et al. 2024)
  - Why width transfer matters for scaling

- **Spectral methods** (0.5 pages):
  - Spectral normalization (Miyato et al. 2018)
  - Orthogonal initialization (Saxe et al. 2013)
  - When/why spectral control helps

- **Shampoo/Muon family** (0.5 pages):
  - Shampoo (Gupta et al. 2018)
  - Original Muon (Jordan 2024)
  - Modular Manifolds (Bernstein 2025)
  - Position your work: "We study inner solvers for Muon, not full manifold optimization"

- **Related optimizer work** (0.5 pages):
  - L-BFGS in neural nets
  - Frank-Wolfe for structured constraints
  - ADMM in deep learning

**Tip**: For each paper, write 2-3 sentences explaining:
1. What they did
2. How it relates to your work
3. What you do differently

---

#### 2. Methods Section Improvements

**Add**:
- **Pseudocode** for each inner solver (2-3 pages)
- **Hyperparameter justification table**:

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Learning rate (SGD) | 0.1 | Standard for CIFAR-10 (He et al. 2016) |
| Spectral budget | 0.1 | ~10% of typical σ_max in ResNet-18 |
| Inner solver iters | 20 | Empirically chosen for convergence |
| Batch size | 128 | GPU memory constraint |

- **Computational complexity analysis**:

| Solver | Complexity per step | Memory | Notes |
|--------|---------------------|--------|-------|
| SpectralClip | O(SVD) = O(mn²) | Low | Baseline |
| DualAscent | O(k × SVD) | Low | k ≈ 20 iters |
| QuasiNewton | O(k × SVD + m²) | Medium | L-BFGS memory |
| FrankWolfe | O(k × top-SVD) | Low | Rank-1 atoms |
| ADMM | O(k × SVD) | Medium | Auxiliary vars |

---

#### 3. Results Section (Once Experiments Done)

**Structure**:

1. **Baseline Comparison** (Figure 1):
   - 4-panel plot: train loss, val acc, spectral norm, sharpness
   - Caption: "MuonSGD achieves competitive accuracy while maintaining tighter spectral norm bounds"

2. **Inner Solver Ablation** (Figure 2 + Table 2):
   - Accuracy vs. wall-clock time scatter plot
   - Table with mean ± std for each solver
   - Analysis: "DualAscent offers best accuracy-compute trade-off"

3. **Width Transfer** (Figure 3 + Table 3):
   - Plot: Accuracy vs. width for SGD vs. MuonSGD
   - Metric: Variance across widths (lower = better transfer)
   - Discussion: Does it work? If yes, why? If no, what's missing?

4. **Ablation Studies** (optional, if time):
   - Effect of spectral budget value (0.05, 0.1, 0.2)
   - Effect of inner solver iterations (5, 10, 20, 50)

---

#### 4. Discussion Section

**Key points to address**:

1. **Why does DualAscent win?**
   - Warm-starting helps
   - Direct dual optimization is efficient
   - No need for complex curvature estimates

2. **Why does QuasiNewton fail?**
   - Spectral ball problem too simple for L-BFGS
   - Curvature estimates hurt more than help
   - Lesson: Match solver complexity to problem structure

3. **Why is FrankWolfe fast but less accurate?**
   - Rank-1 updates → low memory, fast SVD
   - But under-parameterizes the update space
   - Trade-off: speed vs. expressiveness

4. **Does width transfer work?**
   - If yes: "Spectral constraints provide width-invariant parameterization"
   - If no: "Width transfer requires additional scaling rules beyond spectral constraints"

5. **Limitations**:
   - Only studied spectral ball, not full Stiefel manifold
   - CIFAR-10 only, not large-scale
   - Didn't test transformers extensively

6. **Future Work**:
   - Full manifold Muon with tangency
   - Large-scale experiments (ImageNet)
   - Transformer pretraining
   - Automatic spectral budget tuning

---

## Priority Action Plan

### Week 1 (Dec 5-11): Experiments
1. **Mon-Tue**: Set up JupyterHub, run Priority 1 experiments (background jobs)
2. **Wed-Thu**: Monitor jobs, run Priority 2 experiments
3. **Fri-Sun**: Process results, generate figures

### Week 2 (Dec 12-18): Writing
1. **Mon-Tue**: Expand literature review
2. **Wed-Thu**: Write results section with actual plots
3. **Fri-Sun**: Write discussion, polish draft

### Week 3 (Dec 19-22): Polish
1. **Mon-Tue**: Internal review, fix issues
2. **Wed**: Final draft
3. **Thu**: Submit

---

## Concrete Next Steps (Do These Now)

### 1. On JupyterHub (GPU needed):

```bash
# SSH or open JupyterLab terminal
cd ~
git clone https://github.com/achiii800/cs182.git
cd cs182/muon_mve_project

# Setup
conda create -n muon python=3.10 -y
conda activate muon
pip install -r requirements.txt

# Test small run
python train.py --model small_cnn --optimizer muon_sgd --epochs 5

# If that works, run Priority 1 in background:
chmod +x scripts/run_priority1_experiments.sh
nohup bash scripts/run_priority1_experiments.sh > priority1.log 2>&1 &

# Monitor:
tail -f priority1.log
```

### 2. On Local Machine (Writing while experiments run):

```bash
cd muon_mve_project

# Fix GNS batches
# Open train.py, line 185, change 2 → 5

# Add diagnostics
# Open muon/inner_solvers.py
# In each solver, track self._last_num_iters

# Commit changes
git add -A
git commit -m "Fix GNS estimation, add solver diagnostics"
git push
```

### 3. Writing Tasks (Parallel):

- [ ] Expand literature review (muP, Shampoo, spectral methods)
- [ ] Write hyperparameter justification table
- [ ] Prepare figure captions for expected results
- [ ] Start discussion section outline

---

## Questions to Resolve

1. **Compute access**: Do you have unlimited GPU time on JupyterHub? Or quota limits?
   - If limited: Prioritize Priority 1A (baselines) over 1B (full solver ablation)

2. **Deadline**: When is the final report due?
   - This determines if you have time for stretch goals

3. **Team division**: Who's doing what?
   - Suggest: One person monitors experiments, others write in parallel

4. **Paper length**: Confirm 5-10 pages core + appendices
   - You'll likely use all 10 pages given the scope

---

## Final Assessment

**Strengths**:
- ✅ Code quality is excellent
- ✅ Research question is clear and well-motivated
- ✅ All infrastructure is ready to go

**Blockers**:
- ⚠️ Need to run experiments ASAP (30 GPU-hours)
- ⚠️ Literature review needs expansion
- ⚠️ Error bars missing from all results

**Risk Level**: **MEDIUM**
- Code is ready → LOW RISK
- Time crunch for experiments → MEDIUM RISK
- Writing is manageable → LOW RISK

**Confidence in success**: **HIGH** if you start experiments this week.

---

## Contact / Questions

If you need clarification on:
- Experiment setup → Check `EXPERIMENT_PLAN.md`
- Code issues → Check this document's "Code Fixes" section
- Paper structure → Check "Paper Writing Guidance" section

Good luck! You're in great shape - just need to execute. 🚀

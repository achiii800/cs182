
# Muon MVE Scaffold – Stabilizing Inner Solvers & Geometry Diagnostics

This repo is a **Minimum Viable Experiment (MVE)** scaffold for your EECS 182
Category (1) project on **optimizers, manifold Muon, and geometry/stability
diagnostics**.

The goal is to give you something you can **run immediately on Colab or locally**
to:

- Compare a baseline optimizer vs a simple Muon-like spectral-norm-constrained
  optimizer (`MuonSGD`).
- Log **geometry + stability metrics**:
  - spectral norms of selected layers,
  - a SAM-style **sharpness** proxy,
  - a rough **gradient noise scale (GNS)** proxy.
- Produce CSV logs you can plot into the **loss curves + error bars** the project
  report requires fileciteturn0file160.

This is deliberately **simple and transparent**, so that Opus / Claude Code can
read, refactor, and extend it (e.g., plug in the real manifold Muon inner step
from the Thinking Machines blog, add ResNet-18, ARC-like tasks, operator
benchmarks, etc.).

---

## 1. What’s implemented here

### 1.1 Optimizer: `MuonSGD`

File: `muon/muon_sgd.py`

- A small, **SGD-like optimizer** that:
  - supports **momentum** and **weight decay**,
  - treats all 2D parameters (e.g., Linear weights) as matrices,
  - for those matrices:
    - builds a raw update `delta = -lr * grad_eff`,
    - passes `(W, delta)` through an **inner solver** to enforce a **spectral-norm
      budget**,
    - applies the modified update.

- For 1D parameters (bias, LayerNorm weights) it falls back to standard SGD.

This is not a full manifold Muon, but it is:
- a clean place to plug in a more faithful inner problem (dual ascent / quasi-Newton /
  Frank–Wolfe / ADMM variants as per your proposal fileciteturn0file158),
- already aligned with the **“step size is a worst-case output change bound”** intuition.

### 1.2 Inner solvers

File: `muon/inner_solvers.py`

- `SpectralClipSolver`:
  - compute `||delta||_2` via `torch.linalg.matrix_norm(ord=2)`,
  - if `||delta||_2 > budget`: rescale `delta <- delta * (budget / ||delta||_2)`,
  - otherwise return `delta` unchanged.
  - This is an ultra-simple “spectral norm clipping” baseline.

- `FrankWolfeSolver`:
  - compute SVD of `delta`,
  - take top singular vectors `u, v`,
  - return a **rank-1** update `delta_fw = -budget * u vᵀ`,
  - optionally blend with the raw delta:  
    `delta_out = α * delta + (1-α) * delta_fw`.

This is a **toy Frank–Wolfe inner step**: no projections, low-rank atoms, one
top-SVD call per matrix. It’s designed so you can later compare:

- baseline dual ascent from the modular manifolds repo fileciteturn0file158,
- quasi-Newton on the dual,
- this Frank–Wolfe-style variant.

### 1.3 Metrics: stability & geometry

File: `muon/metrics.py`

Implements three diagnostics that match the **stability / geometry toolkit** we
discussed and the **project requirements** fileciteturn0file159:

1. `compute_spectral_norms(model, max_layers=8)`
   - Grab up to `max_layers` 2D parameters and compute `||W||_2`.
   - Gives you a quick view of **σ_max** evolution over training.

2. `estimate_sharpness(model, loss_fn, batch, epsilon)`
   - One-step SAM-style proxy:
     - compute gradients on a batch,
     - perturb weights by `+ε * sign(grad)`,
     - measure `loss(w + ε·sign(g)) - loss(w)`.
   - Crude but often tracks **sharpness / curvature** trends.

3. `estimate_gradient_noise_scale(model, loss_fn, batch1, batch2)`
   - Build flattened gradients `g₁, g₂` on two mini-batches of the same size,
   - return `||g₁ - g₂||² / 2`.
   - A cheap proxy for **gradient noise scale (CBS)** without full-blown curves.

These are explicitly in the spirit of the **“optimizer behavior via CBS/Sharpness/Curvature”**
bullets in your draft content fileciteturn0file157 and the HW8/HW9
state-space/attention diagnostics fileciteturn0file155turn0file154.

### 1.4 CIFAR-10 training script

File: `train_cifar_muon.py`

- Defines a **small CNN** (`SmallConvNet`) that runs fast on Colab / laptop.
- Builds CIFAR-10 train/test loaders with simple augmentation.
- Chooses optimizer based on `--inner-solver` flag:
  - `"none"` → plain `torch.optim.SGD`,
  - `"spectral_clip"` → `MuonSGD` + `SpectralClipSolver`,
  - `"frank_wolfe"` → `MuonSGD` + `FrankWolfeSolver(blend_with_raw=0.3)`.

- For each epoch, logs:
  - train loss / acc,
  - val loss / acc,
  - max spectral norm over up to 8 layers,
  - sharpness proxy,
  - gradient noise scale (GNS) proxy,
  - learning rate.

- Writes a CSV to `logs/cifar_muon_seed{seed}.csv`.

This gives you the **loss curves + error bars + stability metrics** that the final
project guidelines demand fileciteturn0file160.

---

## 2. How to run this (Colab & local)

### 2.1 Colab

1. Download `muon_mve_project.zip` to your machine.
2. Open a Colab notebook.
3. Upload the zip or mount your Drive and copy it in.
4. In a cell:

```bash
!unzip -q muon_mve_project.zip
%cd muon_mve_project
!pip install -r requirements.txt
```

5. Run an experiment:

```bash
!python train_cifar_muon.py --epochs 5 --batch-size 256 --lr 0.1 \
    --inner-solver spectral_clip --spectral-budget 0.1 --seed 0
```

6. Download the logs:

```bash
from google.colab import files
files.download("logs/cifar_muon_seed0.csv")
```

### 2.2 Local (git + virtualenv / conda)

```bash
unzip muon_mve_project.zip
cd muon_mve_project
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt

python train_cifar_muon.py --epochs 10 --batch-size 256 --lr 0.1 \
    --inner-solver spectral_clip --spectral-budget 0.1 --seed 0
```

---

## 3. Where Opus / Claude Code comes in

When you hand this zip + a prompt to Opus 4.5 or Claude Code, you want it to:

1. **Read and critique** the current implementation:
   - Is the MuonSGD interface clean?
   - Are the inner solvers correct as *toy* baselines?
   - Are the metrics enough? What would it add?

2. **Integrate real manifold Muon**:
   - Pull the official modular manifolds / Muon code from the Thinking Machines
     repo fileciteturn0file158,
   - Replace `SpectralClipSolver` with a faithful inner problem implementation:
     - baseline dual ascent,
     - quasi-Newton on the dual,
     - Frank–Wolfe,
     - ADMM-style, if feasible.

3. **Scale up the experiments**:
   - Swap `SmallConvNet` with:
     - ResNet-18 on CIFAR-10,
     - a tiny Transformer / MLP-Mixer on CIFAR or synthetic sequence data.
   - Add a **muP-compatible** width sweep for hyperparameter transfer (Category 1
     requirement fileciteturn0file160).
   - Add ARC-style toy tasks as a second bench (optional but aligned with your
     interests).

4. **Extend metrics & plots**:
   - Better Hessian eigs / sharpness estimates (e.g. Lanczos / power iteration).
   - More detailed CBS curves.
   - Subspace drift: track principal angles for key layers across seeds.
   - Error bars across 5–10 seeds, with plots the project report can drop in.

5. **Hook into operator-learning / DeepONet-style experiments**:
   - Add a separate script where the same inner solvers are used on:
     - a toy operator-learning problem (e.g. Burgers’ equation PDE surrogate),
     - or a DeepONet-like architecture, perhaps with the NVIDIA operator library
       if available.

---

## 4. Requirements file

File: `requirements.txt`

We keep it minimal and Colab-friendly:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib` (for your own plotting scripts)

---

## 5. Next steps for you (human-in-the-loop)

1. **Run this baseline MVE**:
   - Plain SGD vs `MuonSGD + SpectralClip`.
   - 3–5 seeds, 5–10 epochs.
   - Quick plots of:
     - train/val loss,
     - accuracy,
     - max spectral norm,
     - sharpness,
     - GNS.

2. **Hand the zip + the seed prompt (below) to Opus 4.5 / Claude Code**:
   - Ask it to:
     - read the code,
     - refactor inner solvers into a clean module,
     - insert real manifold Muon variants,
     - propose a more ambitious but still runnable benchmark suite.

3. **Iterate**:
   - Once Opus/Claude propose changes, re-run and look at:
     - stable LR envelope,
     - CBS improvement,
     - seed variance reduction.

4. **Start wiring this into your NeurIPS-style draft**:
   - Use the project report guidelines fileciteturn0file159turn0file160
     to shape figures & sections,
   - Tie results back to HW1’s SVD / Lyapunov view (stability regions, convergence
     base) and to the Muon blog / Fantastic Optimizers paper as needed.

---

## 6. Seed prompt to give Opus / Claude Code

Here is a suggested prompt you can paste directly into Opus/Claude Code
**after** uploading this repo into its filesystem:

```text
You have access to my local filesystem. I just uploaded a small project called
`muon_mve_project`. Please:

1. Inspect the codebase:
   - `train_cifar_muon.py`
   - `muon/muon_sgd.py`
   - `muon/inner_solvers.py`
   - `muon/metrics.py`
   - `requirements.txt`
   - `README.md`

2. Understand the project context:
   - This is an EECS 182 Category (1) project on optimizers & hyperparameter
     transfer.
   - Conceptually, we are studying manifold Muon and related structure-preserving
     priors:
       * controlling singular values / induced norms of weight matrices,
       * measuring stability via spectral norms, sharpness, and gradient noise,
       * comparing different inner solvers for the Muon-style spectral constraint
         (dual ascent, quasi-Newton, Frank–Wolfe, ADMM-style),
       * and checking how well hyperparameters transfer across widths (muP-style).

   - The current code is a *minimum viable experiment*:
       * a small CNN on CIFAR-10,
       * a simple "MuonSGD" optimizer with pluggable inner solvers,
       * basic geometry diagnostics (spectral norms, sharpness, GNS proxies),
       * CSV logging.

3. Your tasks:
   (A) Clean up and document the current implementation.
       - Check that MuonSGD and the inner solvers are implemented sanely
         and efficiently.
       - Add comments or small refactors to improve clarity.

   (B) Integrate real manifold Muon inner solvers.
       - Pull in the reference implementation from the modular manifolds /
         Muon repo (dual ascent, matrix sign, retraction).
       - Implement at least:
           * baseline dual ascent inner solver,
           * quasi-Newton variant on the dual,
           * Frank–Wolfe-style solver (projection-free, low-rank atoms),
           * optionally an ADMM-style splitting if time permits.
       - Make the interface consistent with `BaseInnerSolver`.

   (C) Extend the experiment suite.
       - Add a ResNet-18 on CIFAR-10 experiment.
       - Add a tiny Transformer or Mixer experiment to see how these inner
         solvers behave on attention-like architectures.
       - Add command-line flags to select:
           * optimizer (SGD, AdamW, MuonSGD),
           * inner solver type,
           * spectral budget,
           * model architecture,
           * and width (for muP-style transfer sweeps).

   (D) Extend metrics and logging.
       - Keep the existing spectral norm, sharpness, and GNS measures.
       - Add:
           * better Hessian top-eigen estimates (e.g. Lanczos / power iteration),
           * subspace drift metrics across seeds (principal angles between
             weight subspaces),
           * stable learning-rate envelope scans (detect divergence threshold),
           * error bars across multiple seeds (e.g. mean +/- std).

   (E) Prepare plotting utilities.
       - Implement a small `plot_logs.py` that reads the CSV logs and produces:
           * train/val loss curves,
           * train/val accuracy curves,
           * spectral norm trajectories,
           * sharpness and GNS trajectories,
           * comparison plots across inner solvers / optimizers.

   (F) (Optional, stretch) Operator-learning / ARC-like scaffolding.
       - Add a separate script that sets up a toy operator-learning problem
         (e.g. mapping from input function samples to PDE solution snapshots),
         using a simple DeepONet or Fourier-style model.
       - Reuse the same MuonSGD + inner solver machinery there, so we can see
         if the geometry/stability benefits transfer outside of CIFAR.

4. While you edit code, please:
   - Preserve a clear separation between:
       * model definitions,
       * optimizer/inner-solver logic,
       * experiment scripts,
       * and metrics/logging.
   - Add TODO comments where it's better to stub something rather than guess.
   - Assume I will run everything on either:
       * Colab (single GPU, 12–16 GB),
       * or a local dev machine with one mid-range GPU.

5. Finally, propose:
   - A concrete list of 3–5 core experiments I should run first,
   - The exact command lines,
   - And what plots I should generate for the project report.
```

You can tweak this seed prompt to taste, but the idea is:
- Give Opus/Claude Code both the **code** and the **research framing**,
- Ask it to both **refactor** and **extend** toward the NeurIPS-style report and
  the Category (1) requirements.

---

If you’d like, I can also generate a small `plot_logs.py` scaffold in this repo
to read the CSVs from `logs/` and make Matplotlib figures directly usable in
your draft.

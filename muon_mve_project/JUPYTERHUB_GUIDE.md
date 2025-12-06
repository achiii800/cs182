# JupyterHub Setup and Execution Guide

## Initial Setup (First Time Only)

### 1. Access JupyterHub
- URL: http://149.165.151.110.nip.io:8000/user/akshayrao/lab
- Should already be logged in from browser

### 2. Clone Repository
```bash
# Open Terminal in JupyterLab
cd ~
git clone https://github.com/achiii800/cs182.git
cd cs182/muon_mve_project
```

### 3. Create Conda Environment
```bash
# Create environment
conda create -n muon python=3.10 -y

# Activate it
conda activate muon

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### 4. Test Run
```bash
# Quick 5-epoch test on Small CNN
python train.py --model small_cnn --optimizer muon_sgd --inner-solver dual_ascent --epochs 5

# Check logs
ls logs/
cat logs/*.csv | head -20
```

If that works, you're ready to run experiments!

---

## Running Experiments

### Option 1: Interactive (for testing)
Good for debugging, but terminal must stay open.

```bash
conda activate muon
python train.py --model resnet18 --optimizer muon_sgd --epochs 100
```

### Option 2: Background Jobs (RECOMMENDED)
Runs even if you close browser.

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run Priority 1 experiments in background
conda activate muon
nohup bash scripts/run_priority1_experiments.sh > priority1.log 2>&1 &

# Get job PID
echo $!  # Save this number!

# Monitor progress
tail -f priority1.log

# Check what's running
ps aux | grep python

# If you need to kill it later
kill <PID>
```

### Option 3: Screen/Tmux (BEST for long runs)
Most robust - survives disconnects.

```bash
# Start screen session
screen -S muon_experiments

# Inside screen:
conda activate muon
bash scripts/run_priority1_experiments.sh

# Detach: Press Ctrl+A, then D
# Your experiments keep running!

# Reattach later:
screen -r muon_experiments

# List all screens:
screen -ls

# Kill a screen:
screen -X -S muon_experiments quit
```

---

## Monitoring Progress

### Check Logs
```bash
# List recent experiments
ls -lt logs/ | head -20

# View specific log
tail -n 50 logs/resnet18_sgd_seed0.csv

# Count completed experiments
ls logs/*.csv | wc -l

# Watch live updates
watch -n 10 'ls -lt logs/ | head'
```

### Check GPU Usage
```bash
# If nvidia-smi available:
nvidia-smi

# Watch GPU in real-time:
watch -n 1 nvidia-smi
```

### Estimate Remaining Time
```bash
# Check how many experiments done vs. total
DONE=$(ls logs/resnet18_*_seed*.csv 2>/dev/null | wc -l)
TOTAL=20  # Priority 1A has 20 runs
echo "Progress: $DONE/$TOTAL ($((DONE*100/TOTAL))%)"

# If each run is ~45 min:
REMAINING=$((TOTAL - DONE))
echo "Est. time remaining: $((REMAINING * 45)) minutes"
```

---

## Generating Results

### After Priority 1 Completes:

```bash
conda activate muon

# Generate all paper figures
python scripts/generate_paper_figures.py

# Check outputs
ls paper_figures/

# View a plot (if X11 forwarding enabled)
display paper_figures/figure1_optimizer_comparison.png

# Or download to local machine - see below
```

### Downloading Results to Local Machine

From your local terminal (not JupyterHub):
```bash
# Create local directory
mkdir -p ~/Downloads/muon_results

# SCP logs
scp -r <jupyterhub>:~/cs182/muon_mve_project/logs ~/Downloads/muon_results/

# SCP figures
scp -r <jupyterhub>:~/cs182/muon_mve_project/paper_figures ~/Downloads/muon_results/

# If that doesn't work, use JupyterLab's download feature:
# 1. Right-click folder in file browser
# 2. "Download"
```

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python train.py --batch-size 64  # instead of 128
```

### "Conda environment not found"
```bash
# Recreate it
conda env remove -n muon
conda create -n muon python=3.10 -y
conda activate muon
pip install -r requirements.txt
```

### "Module not found"
```bash
conda activate muon
pip install -r requirements.txt
# If still fails, try:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Experiments running too slow
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# If False, PyTorch not detecting GPU - reinstall:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Killed by system (OOM)
```bash
# System ran out of RAM (not GPU memory)
# Reduce batch size or model size:
python train.py --model small_cnn --batch-size 64
```

---

## Experiment Checklist

### Priority 1A: Baselines (20 runs, ~15 hours)
- [ ] SGD (5 seeds)
- [ ] AdamW (5 seeds)
- [ ] MuonSGD + DualAscent (5 seeds)
- [ ] MuonSGD + SpectralClip (5 seeds)

### Priority 1B: Solver Ablation (15 runs, ~6 hours)
- [ ] SpectralClip (3 seeds)
- [ ] DualAscent (3 seeds)
- [ ] FrankWolfe (3 seeds)
- [ ] QuasiNewton (3 seeds)
- [ ] ADMM (3 seeds)

### Priority 2: Width Transfer (30 runs, ~7.5 hours)
- [ ] MuonSGD widths [0.5, 0.75, 1.0, 1.5, 2.0] (3 seeds each)
- [ ] SGD widths [0.5, 0.75, 1.0, 1.5, 2.0] (3 seeds each)

### Results Generated
- [ ] All logs in `logs/` directory
- [ ] Figures in `paper_figures/`
- [ ] Tables in `paper_figures/*.md`

---

## Quick Commands Reference

```bash
# Setup
conda activate muon
cd ~/cs182/muon_mve_project

# Test
python train.py --model small_cnn --epochs 5

# Run Priority 1 (background)
nohup bash scripts/run_priority1_experiments.sh > priority1.log 2>&1 &

# Run Priority 2 (background)
nohup bash scripts/run_priority2_experiments.sh > priority2.log 2>&1 &

# Monitor
tail -f priority1.log
ls -lt logs/ | head

# Generate figures
python scripts/generate_paper_figures.py

# Check progress
ls logs/*.csv | wc -l
```

---

## Best Practices

1. **Always use screen/tmux for long runs** - JupyterHub sessions can timeout
2. **Monitor GPU usage** - Make sure you're actually using the GPU
3. **Save logs frequently** - Download to local machine periodically
4. **Run small tests first** - Test with `--epochs 5` before full runs
5. **Check disk space** - `df -h` to see available space
6. **Use nohup for background jobs** - Survives terminal disconnects

---

## Time Estimates

| Task | Time | Can Parallelize? |
|------|------|------------------|
| Setup environment | 10 min | No |
| Test run | 5 min | No |
| Priority 1A (20 runs × 45 min) | 15 hours | Yes (if multiple GPUs) |
| Priority 1B (15 runs × 25 min) | 6 hours | Yes |
| Priority 2 (30 runs × 15 min) | 7.5 hours | Yes |
| Generate figures | 5 min | No |
| **Total** | **~29 hours** | **Can be ~15 hours if parallelized** |

If you have access to multiple GPUs or can run multiple experiments in parallel, the total time can be cut in half.

---

## Support

- **JupyterHub issues**: Check Ed Discussion or contact course staff
- **Code issues**: See CODE_REVIEW.md
- **Experiment setup**: See EXPERIMENT_PLAN.md
- **Paper writing**: See CODE_REVIEW.md section "Paper Writing Guidance"

Good luck! 🚀

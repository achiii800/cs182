#!/bin/bash
# Run Priority 1 baseline experiments on JupyterHub
# This script runs all the core baseline comparisons needed for the paper

set -e  # Exit on error

echo "============================================================"
echo "EECS 182 Project: Priority 1 Baseline Experiments"
echo "This will run ~29 hours of experiments (can parallelize)"
echo "============================================================"
echo ""

# Check if conda environment exists
if ! conda env list | grep -q muon; then
    echo "Creating conda environment 'muon'..."
    conda create -n muon python=3.10 -y
    conda activate muon
    pip install -r requirements.txt
else
    echo "Activating conda environment 'muon'..."
    conda activate muon
fi

# Create logs directory
mkdir -p logs plots

echo ""
echo "============================================================"
echo "EXPERIMENT 1A: ResNet-18 Optimizer Comparison (20 runs)"
echo "Estimated time: ~15 hours"
echo "============================================================"
echo ""

# SGD baseline (5 seeds)
echo "Running SGD baseline..."
for seed in 0 1 2 3 4; do
    echo "  Seed $seed/5..."
    python train.py --model resnet18 --optimizer sgd --lr 0.1 --epochs 100 \
                    --seed $seed --exp-name resnet18_sgd_seed${seed} \
                    --batch-size 128 --num-workers 4
done

# AdamW baseline (5 seeds)
echo "Running AdamW baseline..."
for seed in 0 1 2 3 4; do
    echo "  Seed $seed/5..."
    python train.py --model resnet18 --optimizer adamw --lr 1e-3 --epochs 100 \
                    --seed $seed --exp-name resnet18_adamw_seed${seed} \
                    --batch-size 128 --num-workers 4
done

# MuonSGD with DualAscent (5 seeds)
echo "Running MuonSGD (DualAscent)..."
for seed in 0 1 2 3 4; do
    echo "  Seed $seed/5..."
    python train.py --model resnet18 --optimizer muon_sgd --inner-solver dual_ascent \
                    --spectral-budget 0.1 --lr 0.1 --epochs 100 \
                    --seed $seed --exp-name resnet18_muon_dual_seed${seed} \
                    --batch-size 128 --num-workers 4
done

# MuonSGD with SpectralClip (5 seeds)
echo "Running MuonSGD (SpectralClip)..."
for seed in 0 1 2 3 4; do
    echo "  Seed $seed/5..."
    python train.py --model resnet18 --optimizer muon_sgd --inner-solver spectral_clip \
                    --spectral-budget 0.1 --lr 0.1 --epochs 100 \
                    --seed $seed --exp-name resnet18_muon_clip_seed${seed} \
                    --batch-size 128 --num-workers 4
done

echo ""
echo "Experiment 1A complete! Generating plots..."

# Generate comparison plots
python plot_logs.py --logfiles logs/resnet18_sgd_seed*.csv \
                    --aggregate-seeds --output-dir plots/exp1a \
                    --plot-types all

python plot_logs.py --logfiles logs/resnet18_adamw_seed*.csv \
                    --aggregate-seeds --output-dir plots/exp1a \
                    --plot-types all

python plot_logs.py --logfiles logs/resnet18_muon_dual_seed*.csv \
                    --aggregate-seeds --output-dir plots/exp1a \
                    --plot-types all

python plot_logs.py --logfiles logs/resnet18_muon_clip_seed*.csv \
                    --aggregate-seeds --output-dir plots/exp1a \
                    --plot-types all

# Aggregate results
python scripts/aggregate_results.py --pattern "logs/resnet18_sgd_seed*_summary.json" \
                                   --output plots/exp1a/sgd_aggregated.json

python scripts/aggregate_results.py --pattern "logs/resnet18_adamw_seed*_summary.json" \
                                   --output plots/exp1a/adamw_aggregated.json

python scripts/aggregate_results.py --pattern "logs/resnet18_muon_dual_seed*_summary.json" \
                                   --output plots/exp1a/muon_dual_aggregated.json

python scripts/aggregate_results.py --pattern "logs/resnet18_muon_clip_seed*_summary.json" \
                                   --output plots/exp1a/muon_clip_aggregated.json

echo ""
echo "============================================================"
echo "EXPERIMENT 1B: Inner Solver Ablation (15 runs)"
echo "Estimated time: ~6 hours"
echo "============================================================"
echo ""

# Run all 5 solvers with 3 seeds each
for solver in spectral_clip dual_ascent frank_wolfe quasi_newton admm; do
    echo "Testing solver: $solver"
    for seed in 0 1 2; do
        echo "  Seed $seed/3..."
        python train.py --model resnet18 --optimizer muon_sgd --inner-solver $solver \
                        --spectral-budget 0.1 --lr 0.1 --epochs 50 \
                        --seed $seed --exp-name resnet18_solver_${solver}_seed${seed} \
                        --batch-size 128 --num-workers 4
    done
done

echo ""
echo "Experiment 1B complete! Generating plots..."

# Generate per-solver plots with error bars
for solver in spectral_clip dual_ascent frank_wolfe quasi_newton admm; do
    python plot_logs.py --logfiles logs/resnet18_solver_${solver}_seed*.csv \
                        --aggregate-seeds --output-dir plots/exp1b \
                        --plot-types all
done

# Create comparison plot (seed 0 only for clarity)
python plot_logs.py --logfiles logs/resnet18_solver_*_seed0.csv \
                    --labels "SpectralClip" "DualAscent" "FrankWolfe" "QuasiNewton" "ADMM" \
                    --output-dir plots/exp1b \
                    --plot-types combined

# Aggregate results per solver
for solver in spectral_clip dual_ascent frank_wolfe quasi_newton admm; do
    python scripts/aggregate_results.py \
           --pattern "logs/resnet18_solver_${solver}_seed*_summary.json" \
           --output plots/exp1b/${solver}_aggregated.json
done

echo ""
echo "============================================================"
echo "All Priority 1 experiments complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - logs/ (CSV logs and JSON summaries)"
echo "  - plots/exp1a/ (Experiment 1A plots and aggregated stats)"
echo "  - plots/exp1b/ (Experiment 1B plots and aggregated stats)"
echo ""
echo "Next steps:"
echo "  1. Review plots in plots/ directory"
echo "  2. Run Priority 2 experiments (width transfer) with:"
echo "     bash scripts/run_priority2_experiments.sh"
echo "  3. Start writing paper results section"
echo ""
echo "To get a summary table:"
echo "  cat plots/exp1a/*_aggregated.json"
echo "  cat plots/exp1b/*_aggregated.json"
echo ""

#!/bin/bash
# Run Priority 2: muP-style Width Transfer Experiments

set -e

echo "============================================================"
echo "EXPERIMENT 2: muP-style Width Transfer"
echo "Estimated time: ~7.5 hours"
echo "============================================================"
echo ""

conda activate muon

mkdir -p logs plots/exp2

echo "Running MLP width sweep with MuonSGD..."
for width in 0.5 0.75 1.0 1.5 2.0; do
    echo "Width multiplier: $width"
    for seed in 0 1 2; do
        echo "  Seed $seed/3..."
        python train.py --model mlp --optimizer muon_sgd --inner-solver dual_ascent \
                        --width-mult $width --lr 0.1 --spectral-budget 0.1 --epochs 30 \
                        --seed $seed --exp-name mlp_width${width}_muon_seed${seed} \
                        --batch-size 128 --num-workers 4
    done
done

echo ""
echo "Running MLP width sweep with SGD baseline..."
for width in 0.5 0.75 1.0 1.5 2.0; do
    echo "Width multiplier: $width"
    for seed in 0 1 2; do
        echo "  Seed $seed/3..."
        python train.py --model mlp --optimizer sgd --width-mult $width \
                        --lr 0.1 --epochs 30 \
                        --seed $seed --exp-name mlp_width${width}_sgd_seed${seed} \
                        --batch-size 128 --num-workers 4
    done
done

echo ""
echo "Generating width transfer analysis..."

# Aggregate results per width
for width in 0.5 0.75 1.0 1.5 2.0; do
    python scripts/aggregate_results.py \
           --pattern "logs/mlp_width${width}_muon_seed*_summary.json" \
           --output plots/exp2/muon_width${width}_aggregated.json
    
    python scripts/aggregate_results.py \
           --pattern "logs/mlp_width${width}_sgd_seed*_summary.json" \
           --output plots/exp2/sgd_width${width}_aggregated.json
done

# Create width transfer plot (need custom script)
python -c "
import json
import matplotlib.pyplot as plt
import numpy as np

widths = [0.5, 0.75, 1.0, 1.5, 2.0]

muon_accs = []
muon_stds = []
sgd_accs = []
sgd_stds = []

for w in widths:
    with open(f'plots/exp2/muon_width{w}_aggregated.json') as f:
        data = json.load(f)
        if 'best_val_acc' in data:
            muon_accs.append(data['best_val_acc']['mean'])
            muon_stds.append(data['best_val_acc']['std'])
        else:
            muon_accs.append(0)
            muon_stds.append(0)
    
    with open(f'plots/exp2/sgd_width{w}_aggregated.json') as f:
        data = json.load(f)
        if 'best_val_acc' in data:
            sgd_accs.append(data['best_val_acc']['mean'])
            sgd_stds.append(data['best_val_acc']['std'])
        else:
            sgd_accs.append(0)
            sgd_stds.append(0)

plt.figure(figsize=(8, 5))
plt.errorbar(widths, muon_accs, yerr=muon_stds, marker='o', label='MuonSGD', linewidth=2, markersize=8)
plt.errorbar(widths, sgd_accs, yerr=sgd_stds, marker='s', label='SGD', linewidth=2, markersize=8)
plt.xlabel('Width Multiplier', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.title('muP-style Width Transfer Experiment', fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/exp2/width_transfer_comparison.pdf', dpi=300)
plt.savefig('plots/exp2/width_transfer_comparison.png', dpi=150)
print('Saved width transfer plot to plots/exp2/')

# Calculate transfer quality (lower variance = better transfer)
muon_var = np.var(muon_accs)
sgd_var = np.var(sgd_accs)
print(f'\\nTransfer Quality (variance in accuracy across widths):')
print(f'  MuonSGD: {muon_var:.6f}')
print(f'  SGD:     {sgd_var:.6f}')
print(f'  Improvement: {(sgd_var - muon_var) / sgd_var * 100:.1f}%')
"

echo ""
echo "============================================================"
echo "Experiment 2 complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - plots/exp2/width_transfer_comparison.pdf"
echo "  - plots/exp2/*_aggregated.json"
echo ""

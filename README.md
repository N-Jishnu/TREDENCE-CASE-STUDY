# The Self-Pruning Neural Network — Tredence Case Study

This repository contains a PyTorch implementation of a self-pruning feed-forward network for CIFAR-10, based on learnable gates and L1 sparsity regularization.

## Environment

- Platform: Kaggle Notebook
- Accelerator: NVIDIA Tesla T4 GPU
- Framework: PyTorch + torchvision

## Setup

```bash
pip install -r requirements.txt
```

## Run Command

Use this command in Kaggle/Colab terminal or notebook cell:

```bash
python self_pruning_network.py --mode train --output-dir . --report-filename report.md
```

Recommended lambda sweep used for the submitted results:

```bash
python self_pruning_network.py --mode train --epochs 50 --lambdas "1e-5,3e-5,1e-4" --gate-lr-multiplier 2 --output-dir . --report-filename report.md
```

## Output Files

- `self_pruning_network.py`: Complete single-file implementation (model, training, evaluation, plotting, report generation).
- `report.md`: Final experiment report with metrics tables and analysis.
- `gate_distribution.png`: Histogram of learned gate values for the selected best trade-off model.
- `gate_evolution.png`: Sparsity and gate-behavior trends across epochs for each lambda.
- `sparsity_accuracy_tradeoff.png`: Accuracy and sparsity comparison across lambda values.
- `training_curves.png`: Training dynamics (accuracy, sparsity, classification loss, total loss).

## Notes

- The script tracks both best-epoch and final-epoch metrics per lambda.
- Late-epoch over-pruning can reduce final accuracy; best-epoch metrics are used for trade-off reporting.
- The script auto-generates plots and `report.md` at the end of training.

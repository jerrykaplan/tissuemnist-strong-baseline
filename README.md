# A Strong Baseline for TissueMNIST

Code accompanying the paper:

> Kaplan, J. (2026). A Strong Baseline for TissueMNIST: Geometric Augmentation Outperforms Recent Specialized Architectures. *Pattern Recognition Letters* (under review).

## Overview

This repository contains the experimental scripts used to produce the results
reported in the paper. The work shows that a vanilla ResNet-18 trained with
standard geometric data augmentation achieves 0.7344 ± 0.0010 test accuracy on
TissueMNIST 28×28, exceeding the published state of the art (MedViTV2-large,
0.716) by 1.8 percentage points across four random seeds.

The paper additionally examines the augmentation mechanism through a rotation
angle ablation and a battery of non-geometric regularization techniques,
finding that the benefit derives specifically from coherent geometric
transformations of the image rather than from generic regularization.

## Requirements

- Python 3.11+
- PyTorch 2.0+
- NumPy
- scikit-learn 1.4+
- A GPU is recommended; experiments were run on a single NVIDIA RTX 4090
  (~14 minutes per training run)

## Data

The TissueMNIST dataset is part of the MedMNIST collection. Download from:
https://medmnist.com/

The scripts assume the data is saved as `tissuemnist.npz` containing arrays
`train_X`, `train_y`, `val_X`, `val_y`, `test_X`, `test_y`. The data file used
in this work matches the official MedMNIST distribution exactly: 165,466
training samples, 23,640 validation samples, and 47,280 test samples, all at
28×28 uint8 grayscale.

## Scripts

| Script | Purpose |
|--------|---------|
| `model.py` | ResNet-18 architecture (adapted for 28×28 inputs) and TissueMNIST dataset class with optional augmentation |
| `multiseed_full_aug.py` | Trains ResNet-18 with full geometric augmentation... |
| `auc_multiseed.py` | Multi-seed retraining with AUC computation... |
| `memorization_battery.py` | Tests nine non-geometric regularization techniques... |

## Reproducing the main results

```bash
# Strong baseline (Table 1): ~60 minutes
python3 auc_multiseed.py --seeds 1 7 13 42 --epochs 40

# Rotation angle ablation (Table 2): see paper for individual rotation configs
# Each configuration is a separate run with the rotation range modified

# Non-geometric regularization battery (Table 3): ~2 hours
python3 memorization_battery.py --epochs 40
```

## Architecture

ResNet-18 is adapted for 28×28 single-channel inputs by replacing the initial
7×7 stride-2 convolution with a 3×3 stride-1 convolution and removing the
initial max-pool layer. This is the standard low-resolution adaptation used in
MedMNIST baselines.

## Training recipe

- Optimizer: Adam, initial learning rate 1e-3
- Schedule: MultiStepLR with decay factor 10 at 50% and 75% of training
- Batch size: 128
- Epochs: 40
- Loss: cross-entropy
- Normalization: inputs to [-1, 1]

The augmentation recipe applies, per training sample at each epoch:
rotation uniformly drawn from [-15°, +15°], translation up to ±2 pixels,
and horizontal flip with probability 0.5.

## License

MIT License. See LICENSE file.

## Contact

Jerry Kaplan, Stanford University. jerrykaplan@stanford.edu

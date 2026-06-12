# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparseFormer — entropy-driven adaptive pruning for Swin Transformer in DINO object detection. The core idea is **dual-dimensional information redundancy**: (1) spatial — low-KL-divergence windows skip attention, (2) depth — low-entropy-variation windows skip FFN. Built on MMDetection 3.3.0.

## Common Commands

```bash
# Training
bash scripts/dist_train.sh configs/dino/<config>.py <num_gpus>

# Single-GPU training (debugging)
python tools/train.py configs/dino/<config>.py

# Testing
bash scripts/dist_test.sh configs/dino/<config>.py <checkpoint> <num_gpus>

# Plot training loss curves
python tools/plot_log.py <log_file>

# Compute FLOPs of a model
python tools/analysis_tools/get_flops_2.py <config>
```

## Architecture

```
sparse_former/
├── models/
│   ├── backbones/          # Swin Transformer variants with pruning
│   │   ├── swin_baseline_v3.py   # Current main: dual KL+INC pruning, learnable gate
│   │   ├── swin_baseline_v4.py   # Next iteration (in progress)
│   │   ├── swin_entropy.py       # Original entropy-based implementation
│   │   └── ...                   # v1, v2, baseline (earlier iterations)
│   ├── detectors/
│   │   └── dino_with_gate_loss.py  # DINO detector that collects gate loss from backbone
│   ├── datasets/           # PandaDataset, CocoDataset
│   └── utils/
│       └── entropy_vis_hook.py  # Hook for entropy visualization & score collection
├── configs/dino/           # MMDetection config files
├── tools/                  # train.py, test.py, analysis scripts
├── scripts/                # dist_train.sh, dist_test.sh
└── outputs/                # Training outputs by experiment
```

## Key Design: SwinTransformerV3 (swin_baseline_v3.py)

Three pruning strategies (`strategy=` config):
- **`kl`**: KL-based window pruning — computes per-window KL divergence from global distribution, masks low-KL windows to skip attention
- **`inc`**: Entropy variation pruning — compares entropy across layers/stages, masks stable windows to skip FFN
- **`kl_inc`**: Two-stage cascade — KL filters attention windows, then INC further filters FFN windows from KL survivors

Key components within `SwinBlockV3`:
- **`ThresholdPredictor`**: A learnable MLP that predicts per-block pruning thresholds from score statistics (mean, std, p50, max) + stage/block embeddings. Output is sigmoid-activated to [0,1].
- **`compute_gate_loss`**: The gate loss function. Current form in `compute_gate_loss()` (line 379-416) only returns `reg_term = lambda * activation_ratio` (the `info_term` is computed and logged but NOT included in the loss — see line 409: `loss = reg_term`).
- **`compute_soft_mask`**: `m_i = sigmoid((S_i - tau) / T)` — soft mask for differentiable pruning
- **`collect_stats`**: Computes statistics (mean, std, median, max) of scores, with scores detached to prevent gradient issues

Loss flow: Each `SwinBlockV3` forward returns `block_gate_loss` tuples → `SwinBlockSequenceV3` collects them → `SwinTransformerV3.forward()` aggregates them into buffers → `DINOWithGateLoss.loss()` reads them and adds `loss_kl_gate` / `loss_inc_gate` to the total loss dict.

## Config Notes

- Backbone uses `strategy='kl'`, `'inc'`, or `'kl_inc'`
- `stage_config` controls which blocks get pruning and the keep ratio per stage
- `use_learnable_gate=True` enables the `ThresholdPredictor` (learnable threshold instead of fixed Top-K)
- `temperature=0.2` controls soft mask sharpness (lower = harder mask)
- `lambda_kl` / `lambda_inc` control the gate loss regularization strength
- Only W-MSA blocks (not SW-MSA) participate in pruning (shift_size==0 check)

## Environment

- Python 3.9, PyTorch 2.3.1+cu121, CUDA 12.0
- MMDetection 3.3.0, MMEngine 0.10.7
- conda env: `sparse-former`

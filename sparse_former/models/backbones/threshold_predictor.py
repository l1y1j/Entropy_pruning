"""Information-aware Threshold Predictor for sparse computation.

Re-encapsulated from swin_baseline_v3.py.  Independent module — does not import
from V3 so the interface can evolve separately for SparseNet's needs.
"""

import torch
import torch.nn as nn


class InformationThresholdPredictor(nn.Module):
    """Learnable MLP that predicts per-block pruning threshold τ.

    Inputs:
        soft_hist  (B, 16) — 16-bin soft histogram of window scores
        C_b        (B,  1) — image-level complexity
        stage_idx  int|tensor — which stage (0..3)
        block_idx  int|tensor — which block within stage (0..depth-1)

    Output:
        τ  (B, 1) ∈ [0, 1] — predicted pruning threshold

    Architecture:
        soft_hist + C_b  →  Linear(17→64) → ReLU  ─┐
        stage_emb        →  Embedding(4, 16) ───────┤
        block_emb        →  Embedding(8, 16) ───────┤
                                                     ├→ Linear(96→96)→ReLU
                                                     │       ↓
                                                     │  Linear(96→48)→ReLU
                                                     │       ↓
                                                     │  Linear(48→1)→Sigmoid → τ

    Stage/block embeddings let the same MLP learn different pruning strategies
    for different positions in the network (early layers keep more, etc.).
    """

    def __init__(self, hidden_dim=64, num_stages=4, num_blocks=8):
        super().__init__()
        # Statistics encoder: 16-bin histogram + C_b → hidden
        self.stat_encoder = nn.Sequential(
            nn.Linear(17, hidden_dim),
            nn.ReLU(),
        )
        # Positional embeddings
        self.stage_emb = nn.Embedding(num_stages, 16)
        self.block_emb = nn.Embedding(num_blocks, 16)
        # Fusion + prediction network
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim + 32),
            nn.ReLU(),
            nn.Linear(hidden_dim + 32, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, soft_hist, C_b, stage_idx, block_idx):
        """Predict threshold τ from score distribution statistics and position.

        Args:
            soft_hist: (B, 16)  soft histogram of window scores
            C_b:       (B, 1)   image-level complexity
            stage_idx: int or (B,) tensor — current stage index
            block_idx: int or (B,) tensor — current block index within stage

        Returns:
            threshold: (B, 1) ∈ [0, 1]
        """
        # Convert Python int → tensor (expand to batch if needed)
        if not isinstance(stage_idx, torch.Tensor):
            stage_idx = torch.tensor(
                [stage_idx], device=soft_hist.device, dtype=torch.long
            )
            if soft_hist.shape[0] > 1:
                stage_idx = stage_idx.expand(soft_hist.shape[0])
        if not isinstance(block_idx, torch.Tensor):
            block_idx = torch.tensor(
                [block_idx], device=soft_hist.device, dtype=torch.long
            )
            if soft_hist.shape[0] > 1:
                block_idx = block_idx.expand(soft_hist.shape[0])

        # Encode statistics
        stats = torch.cat([soft_hist, C_b], dim=1)  # (B, 17)
        h_stat = self.stat_encoder(stats)            # (B, hidden_dim)

        # Position embeddings
        h_stage = self.stage_emb(stage_idx)  # (B, 16)
        h_block = self.block_emb(block_idx)  # (B, 16)

        # Fuse and predict
        h = torch.cat([h_stat, h_stage, h_block], dim=1)  # (B, hidden_dim+32)
        return self.net(h)  # (B, 1) ∈ [0, 1]

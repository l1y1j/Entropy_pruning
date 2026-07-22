"""InformationScoreNet — replaces SparseNet's score_net(Linear) with KL scoring.

This is the core replacement: where SparseNet used a Linear layer on residual
features to score windows, we use KL divergence + ThresholdPredictor.

Interface alignment with score_net:
    score_net:    residual_windows (B*N, ws²*C) → Linear → (B*N, 1) → Top-K
    This module:  x_windows (B*N, ws, ws, C) → KL → ThresholdPredictor → keep_indices
"""

import torch
import torch.nn as nn

from .entropy_utils import (
    compute_window_relative_entropy,
    compute_soft_histogram,
    compute_gate_loss,
    min_max_normalize,
)


class InformationScoreNet(nn.Module):
    """KL-divergence based window scoring network.

    Replaces SparseNet's score_net (a single Linear layer) with an information-
    theoretic scoring pipeline:
        1. KL divergence per window (how unique is this window?)
        2. Soft histogram (what does the score distribution look like?)
        3. ThresholdPredictor (what threshold to use for this image+stage+block?)
        4. Soft mask + hard mask → keep_indices

    Args:
        window_size:        spatial window size (default 7)
        threshold_predictor: InformationThresholdPredictor instance (shared across blocks)
        temperature:        soft mask temperature (lower = harder mask)
        lambda_reg:         gate loss sparsity weight (0 = no gate loss)
    """

    def __init__(self, window_size, threshold_predictor,
                 temperature=1.0, lambda_reg=0.1):
        super().__init__()
        self.window_size = window_size
        # Use object.__setattr__ to avoid registering threshold_predictor as
        # a submodule.  The predictor is owned by SparseNetEntropy and shared
        # across all InformationScoreNet instances.
        object.__setattr__(self, 'threshold_predictor', threshold_predictor)
        self.temperature = temperature
        self.lambda_reg = lambda_reg

    def forward(self, x_windows, B, C_b, stage_idx, block_idx):
        """Score windows and produce keep indices.

        Args:
            x_windows:  (B*N, ws, ws, C)  window-partitioned features
            B:          batch size
            C_b:        (B, 1)  image-level complexity
            stage_idx:  int  current stage index (0..3)
            block_idx:  int  current block index within stage

        Returns:
            keep_indices:  (B, K)  indices of windows to keep (for gather)
            soft_mask:     (B, N)  soft mask values ∈ (0, 1) (for weighting + gate loss)
            gate_loss:     scalar tensor or None
        """
        ws = self.window_size
        total_windows = x_windows.shape[0]
        N = total_windows // B  # windows per image

        # ---- 1. KL divergence scoring ----
        kl_scores = compute_window_relative_entropy(x_windows, B, ws)  # (B*N,)
        kl_2d = kl_scores.view(B, N)  # (B, N)

        # ---- 2. Min-Max normalize to [0, 1] ----
        kl_norm = min_max_normalize(kl_2d)  # (B, N)

        # ---- 3. Soft histogram (distribution shape) ----
        soft_hist = compute_soft_histogram(kl_norm.view(-1), B, K=16)  # (B, 16)

        # ---- 4. ThresholdPredictor → per-image adaptive τ ----
        tau = self.threshold_predictor(
            soft_hist, C_b, stage_idx, block_idx
        )  # (B, 1)

        # ---- 5. Soft mask:  m_i = σ((score_i - τ) / T) ----
        tau_tiled = tau.squeeze(-1).unsqueeze(-1)  # (B, 1) → (B, N)
        soft_mask = torch.sigmoid((kl_norm - tau_tiled) / self.temperature)
        soft_mask = torch.clamp(soft_mask, min=1e-6, max=1 - 1e-6)

        # ---- 6. Hard mask: keep where score > τ ----
        hard_mask = kl_norm > tau_tiled  # (B, N)

        # Fallback: at least keep the single highest-scoring window per image
        fallback = torch.zeros_like(hard_mask)
        fallback[
            torch.arange(B, device=hard_mask.device), kl_2d.argmax(dim=1)
        ] = True
        hard_mask = hard_mask | fallback

        # ---- 7. Build keep_indices ----
        # Images may have different K_b = hard_mask[b].sum().
        # torch.gather needs uniform (B, K). We take K = max(K_b) and pad
        # shorter images with their highest-scoring extra windows.
        K_per_image = hard_mask.sum(dim=1)  # (B,)
        K_max = K_per_image.max().item()

        # Sort by soft_mask descending → top-K_max per image
        _, sorted_idx = soft_mask.sort(dim=1, descending=True)  # (B, N)
        keep_indices = sorted_idx[:, :K_max]  # (B, K_max)

        # ---- 8. Gate loss + keep ratio ----
        gate_loss = None
        keep_ratio = None
        if self.lambda_reg > 0:
            gate_loss, info_ratio, act_ratio = compute_gate_loss(
                soft_mask.view(-1), kl_norm.view(-1), self.lambda_reg
            )
            # act_ratio = mean(m_i) ≈ fraction of windows "active" = keep ratio
            keep_ratio = act_ratio

        return keep_indices, soft_mask, gate_loss, keep_ratio

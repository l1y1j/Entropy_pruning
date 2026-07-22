"""EntropyBlockSequence — BlockSequence with progressive entropy-driven scoring.

Stage-entry:   KL  scores window importance (static information).
Per-block:     INC re-scores after each conv to track information evolution.

Story:
    KL  → "Which windows are worth computing?"
    INC → "After this conv, which windows still have new information to gain?"

Inherits from sparsenet.BlockSequence.  GlobalBlock, Gather→Conv→Scatter,
and Re-parameter are inherited unchanged.
"""

import torch
import torch.nn.functional as F

from .sparsenet import BlockSequence
from .information_score_net import InformationScoreNet
from .entropy_utils import (
    compute_soft_histogram,
    compute_gate_loss,
    min_max_normalize,
)


class EntropyBlockSequence(BlockSequence):
    """SparseNet stage with progressive information refinement.

    Flow within one stage (depth=2 example):
        Feature
          │
          ├─ KL score → keep⁰ → Global Injection
          │
          ├─ [Block 0] H_before → Gather(keep⁰) → Conv → Scatter → H_after
          │     │
          │     └─ INC = |H_after - H_before|
          │           → Histogram → ThresholdPredictor(INC) → keep¹
          │
          ├─ [Block 1] H_before → Gather(keep¹) → Conv → Scatter → H_after
          │     │
          │     └─ INC = |H_after - H_before|
          │           → Histogram → ThresholdPredictor(INC) → keep²
          │
          └─ Re-parameter → PatchMerging
    """

    def __init__(self, *args, **kwargs):
        # Pop entropy-specific params BEFORE super().__init__()
        _entropy_temperature = kwargs.pop('temperature', 1.0)
        _entropy_lambda_kl = kwargs.pop('lambda_kl', 0.1)
        _entropy_lambda_inc = kwargs.pop('lambda_inc', 0.1)
        _entropy_kl_predictor = kwargs.pop('kl_predictor', None)
        _entropy_inc_predictor = kwargs.pop('inc_predictor', None)
        _entropy_use_gate = kwargs.pop('use_learnable_gate', False)

        super().__init__(*args, **kwargs)

        # Now safe to set attributes
        self._entropy_temperature = _entropy_temperature
        self._entropy_lambda_kl = _entropy_lambda_kl
        self._entropy_lambda_inc = _entropy_lambda_inc
        # Use object.__setattr__ — predictors owned by SparseNetEntropy, shared
        object.__setattr__(self, '_entropy_kl_predictor', _entropy_kl_predictor)
        object.__setattr__(self, '_entropy_inc_predictor', _entropy_inc_predictor)
        self._entropy_use_gate = _entropy_use_gate

        # InformationScoreNet for KL (stage-entry scoring, used once)
        self.information_score_net = InformationScoreNet(
            window_size=self.window_size,
            threshold_predictor=self._entropy_kl_predictor,
            temperature=self._entropy_temperature,
            lambda_reg=self._entropy_lambda_kl if self._entropy_use_gate else 0.0,
        )

        # Gate loss accumulator: list of (kl_gate_loss, inc_gate_loss) tuples
        self.gate_losses = []

    # ==================================================================
    #  Forward: progressive information refinement
    # ==================================================================
    def forward(self, x, hw_shape, C_b=None):
        """Forward with progressive INC rescoring after each block.

        Args:
            x:        (B, L, C)  input features
            hw_shape: (H, W)     spatial shape
            C_b:      (B, 1)     image complexity (computed upstream)
        """
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'

        if C_b is None:
            C_b = torch.zeros(B, 1, device=x.device)

        # ---- Stage entry: KL score (initial window importance) ----
        keep_indices, soft_mask, x_score, x_att_global = self._kl_score(
            x, hw_shape, C_b
        )

        # ---- Global Injection (unchanged from SparseNet) ----
        x = x + x_att_global * (1 - x_score)

        # ---- Progressive execution with per-block INC rescoring ----
        # INC is only defined on computed windows.
        # For each block I:
        #   1. Compute H_before on the K_I windows that will be computed
        #   2. Execute block I (conv on K_I, identity on N-K_I)
        #   3. Compute H_after on those same K_I windows
        #   4. INC = |H_after - H_before| on K_I windows
        #   5. Scatter INC back to (B,N) — non-computed windows: INC ≡ 0
        #   6. Feed INC map to ThresholdPredictor → keep_indices for block I+1

        N_windows = self._compute_window_entropy(x, hw_shape).shape[0] // B

        for i, block in enumerate(self.local_blocks):
            # --- H_before: only on windows that will be computed ---
            H_before_all = self._compute_window_entropy(x, hw_shape)  # (B*N,)
            H_before_kept = H_before_all.view(B, -1).gather(
                1, keep_indices
            )  # (B, K)

            # --- Execute this block ---
            x = block(x, hw_shape, keep_indices)

            # --- INC rescore for next block (except after the last) ---
            if i < len(self.local_blocks) - 1:
                H_after_all = self._compute_window_entropy(x, hw_shape)  # (B*N,)
                H_after_kept = H_after_all.view(B, -1).gather(
                    1, keep_indices
                )  # (B, K)

                # INC only on computed windows
                inc_kept = torch.abs(H_after_kept - H_before_kept)  # (B, K)

                # Scatter to full (B, N) map — non-computed windows: INC ≡ 0
                inc_full = torch.zeros(B, N_windows, device=x.device)
                inc_full.scatter_(1, keep_indices, inc_kept)
                inc_scores = inc_full.view(-1)  # (B*N,)

                # Rescore for the NEXT block
                keep_indices, inc_gate_loss, inc_keep_ratio = self._inc_rescore(
                    inc_scores=inc_scores,
                    B=B,
                    C_b=C_b,
                    stage_idx=0,    # TODO: inject actual value
                    block_idx=i,    # this block's index
                )
                if inc_gate_loss is not None:
                    self.gate_losses.append((None, inc_gate_loss, None, inc_keep_ratio))

        # ---- Re-parameter + Downsample (unchanged) ----
        x_score = 1 + x_score - x_score.detach()
        x = x * x_score

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

    # ==================================================================
    #  KL scoring (stage entry — same logic as before, refactored)
    # ==================================================================
    def _kl_score(self, x, hw_shape, C_b):
        """KL-based initial window importance scoring at stage entry.

        Returns:
            keep_indices:  (B, K)       indices for gather
            soft_mask:     (B, N_win)   window-level soft mask
            x_score:       (B, L, C)    token-level mask (for global injection)
            x_att_global:  (B, L, C)    global context features
        """
        B, L, C = x.shape
        H, W = hw_shape
        x_2d = x.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x_padded = F.pad(x_2d, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x_padded.shape[1], x_padded.shape[2]

        # Pool for GlobalBlock
        x_pooled = self.pooling(x_padded.permute(0, 3, 1, 2))

        # KL scoring
        x_windows = self.window_partition(x_padded)
        keep_indices, soft_mask, kl_gate_loss, kl_keep_ratio = \
            self.information_score_net(
                x_windows, B, C_b, stage_idx=0, block_idx=0,
            )
        if kl_gate_loss is not None:
            # (kl_loss, inc_loss, kl_keep_ratio, inc_keep_ratio)
            self.gate_losses.append((kl_gate_loss, None, kl_keep_ratio, None))

        # Upsample soft_mask → x_score (token-level)
        H_win = H_pad // self.window_size
        W_win = W_pad // self.window_size

        x_score = soft_mask.view(
            B, 1, H_win, 1, W_win, 1
        ).repeat(
            1, C, 1, self.window_size, 1, self.window_size
        ).view(B, C, H_pad, W_pad).permute(0, 2, 3, 1)

        if pad_r > 0 or pad_b > 0:
            x_score = x_score[:, :H, :W, :].contiguous().view(B, -1, C)
        else:
            x_score = x_score.contiguous().view(B, -1, C)

        # GlobalBlock (unchanged from SparseNet)
        x_att_global = x_pooled
        for block in self.global_blocks:
            x_att_global = block(x_att_global)

        x_att_global = x_att_global.view(
            B, -1, H_pad // self.window_size, W_pad // self.window_size
        )
        x_att_global = F.interpolate(
            x_att_global, scale_factor=self.window_size
        ).permute(0, 2, 3, 1)

        if pad_r > 0 or pad_b > 0:
            x_att_global = x_att_global[:, :H, :W, :].contiguous().view(B, -1, C)
        else:
            x_att_global = x_att_global.contiguous().view(B, -1, C)

        return keep_indices, soft_mask, x_score, x_att_global

    # ==================================================================
    #  Window entropy
    # ==================================================================
    def _compute_window_entropy(self, x, hw_shape):
        """Compute per-window entropy of features (detached).

        Used for INC = |H_after - H_before|.

        Args:
            x:        (B, L, C)  features
            hw_shape: (H, W)     spatial shape

        Returns:
            entropy:  (B*N,)  per-window entropy, detached
        """
        B, L, C = x.shape
        H, W = hw_shape
        x_2d = x.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x_padded = F.pad(x_2d, (0, 0, 0, pad_r, 0, pad_b))

        x_windows = self.window_partition(x_padded)  # (B*N, ws, ws, C)
        total_windows = x_windows.shape[0]
        N = total_windows // B

        x_windows = x_windows.view(B, N, self.window_size * self.window_size, C)
        # Per-window: mean(entropy over tokens)
        attn = F.softmax(x_windows, dim=-1)
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean(dim=-1)
        return entropy.view(-1).detach()  # (B*N,)

    # ==================================================================
    #  INC rescoring (after each block)
    # ==================================================================
    def _inc_rescore(self, inc_scores, B, C_b, stage_idx, block_idx):
        """INC-based window rescoring for the NEXT block.

        Pipeline:
            inc_scores (B*N,) → normalize → histogram →
            ThresholdPredictor(INC) → τ → soft_mask → keep_indices → gate_loss

        Args:
            inc_scores:  (B*N,)  |ΔH| for ALL windows
            B:           batch size
            C_b:         (B, 1)  image complexity
            stage_idx:   int
            block_idx:   int (the block that JUST ran, not the next one)

        Returns:
            keep_indices:  (B, K_new)  refined indices for next block
            inc_gate_loss: scalar or None
        """
        N_total = inc_scores.shape[0] // B
        inc_2d = inc_scores.view(B, N_total)  # (B, N)

        # 1. Min-Max normalize → [0, 1]
        inc_norm = min_max_normalize(inc_2d)  # (B, N)

        # 2. Soft histogram (distribution shape of INC scores)
        soft_hist = compute_soft_histogram(inc_norm.view(-1), B, K=16)  # (B, 16)

        # 3. INC ThresholdPredictor → τ_inc
        tau_inc = self._entropy_inc_predictor(
            soft_hist, C_b, stage_idx, block_idx
        )  # (B, 1)

        # 4. Soft mask
        tau_tiled = tau_inc.squeeze(-1).unsqueeze(-1)  # (B, 1) → (B, N)
        soft_mask_inc = torch.sigmoid((inc_norm - tau_tiled) / self._entropy_temperature)
        soft_mask_inc = torch.clamp(soft_mask_inc, min=1e-6, max=1 - 1e-6)

        # 5. Hard mask (no fallback needed — at least 1 window will have INC > 0)
        hard_mask = inc_norm > tau_tiled

        # Fallback: keep at least the most-changed window
        fallback = torch.zeros_like(hard_mask)
        fallback[torch.arange(B, device=hard_mask.device), inc_2d.argmax(dim=1)] = True
        hard_mask = hard_mask | fallback

        # 6. Build keep_indices (same K_max strategy as KL)
        K_per_image = hard_mask.sum(dim=1)
        K_max = K_per_image.max().item()
        _, sorted_idx = soft_mask_inc.sort(dim=1, descending=True)
        keep_indices = sorted_idx[:, :K_max]  # (B, K_max)

        # 7. INC Gate loss + keep ratio
        inc_gate_loss = None
        inc_keep_ratio = None
        if self._entropy_lambda_inc > 0:
            inc_gate_loss, _, inc_act_ratio = compute_gate_loss(
                soft_mask_inc.view(-1),
                inc_norm.view(-1),
                self._entropy_lambda_inc,
            )
            inc_keep_ratio = inc_act_ratio

        return keep_indices, inc_gate_loss, inc_keep_ratio

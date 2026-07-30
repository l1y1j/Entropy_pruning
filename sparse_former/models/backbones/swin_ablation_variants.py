"""Ablation variants of ThresholdPredictor and SwinTransformerV3.

Usage in config::

    backbone=dict(
        type='SwinTransformerV3_NoDist',   # pick the variant
        ...
    )
"""

import torch
import torch.nn as nn
from mmdet.registry import MODELS
from .swin_baseline_v3 import ThresholdPredictor, SwinTransformerV3


# ==============================================================================
# Predictor variants
# ==============================================================================

class PredictorNoDist(ThresholdPredictor):
    """Drop the soft-histogram input — the MLP only sees image complexity C_b."""

    def __init__(self, hidden_dim=64):
        super().__init__(hidden_dim=hidden_dim)
        self.stat_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),   # C_b only
            nn.ReLU(),
        )

    def forward(self, soft_hist, C_b, stage_idx, block_idx):
        if not isinstance(stage_idx, torch.Tensor):
            stage_idx = torch.tensor([stage_idx], device=C_b.device, dtype=torch.long)
            if C_b.shape[0] > 1:
                stage_idx = stage_idx.expand(C_b.shape[0])
        if not isinstance(block_idx, torch.Tensor):
            block_idx = torch.tensor([block_idx], device=C_b.device, dtype=torch.long)
            if C_b.shape[0] > 1:
                block_idx = block_idx.expand(C_b.shape[0])
        h_stat = self.stat_encoder(C_b)                      # (B, 64)  C_b only
        h_stage = self.stage_emb(stage_idx)                  # (B, 16)
        h_block = self.block_emb(block_idx)                  # (B, 16)
        h = torch.cat([h_stat, h_stage, h_block], dim=1)
        return self.net(h)                                    # (B, 1) ∈ [0, 1]


class PredictorNoCb(ThresholdPredictor):
    """Drop the image-complexity input — the MLP only sees the soft histogram."""

    def __init__(self, hidden_dim=64):
        super().__init__(hidden_dim=hidden_dim)
        self.stat_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim),  # soft_hist only
            nn.ReLU(),
        )

    def forward(self, soft_hist, C_b, stage_idx, block_idx):
        if not isinstance(stage_idx, torch.Tensor):
            stage_idx = torch.tensor([stage_idx], device=soft_hist.device, dtype=torch.long)
            if soft_hist.shape[0] > 1:
                stage_idx = stage_idx.expand(soft_hist.shape[0])
        if not isinstance(block_idx, torch.Tensor):
            block_idx = torch.tensor([block_idx], device=soft_hist.device, dtype=torch.long)
            if soft_hist.shape[0] > 1:
                block_idx = block_idx.expand(soft_hist.shape[0])
        h_stat = self.stat_encoder(soft_hist)                # (B, 64)  soft_hist only
        h_stage = self.stage_emb(stage_idx)                  # (B, 16)
        h_block = self.block_emb(block_idx)                  # (B, 16)
        h = torch.cat([h_stat, h_stage, h_block], dim=1)
        return self.net(h)                                    # (B, 1) ∈ [0, 1]


class PredictorNoPos(ThresholdPredictor):
    """Drop stage / block position embeddings.

    The fusion network receives only the statistics branch (64-dim).
    """

    def __init__(self, hidden_dim=64):
        super().__init__(hidden_dim=hidden_dim)
        del self.stage_emb
        del self.block_emb
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, soft_hist, C_b, stage_idx, block_idx):
        # replicate the parent logic but without pos-emb concat
        if not isinstance(stage_idx, torch.Tensor):
            stage_idx = torch.tensor([stage_idx], device=soft_hist.device, dtype=torch.long)
            if soft_hist.shape[0] > 1:
                stage_idx = stage_idx.expand(soft_hist.shape[0])
        if not isinstance(block_idx, torch.Tensor):
            block_idx = torch.tensor([block_idx], device=soft_hist.device, dtype=torch.long)
            if soft_hist.shape[0] > 1:
                block_idx = block_idx.expand(soft_hist.shape[0])

        stats = torch.cat([soft_hist, C_b], dim=1)  # (B, 17)
        h_stat = self.stat_encoder(stats)            # (B, hidden_dim)
        return self.net(h_stat)                       # no pos-emb concat


class PredictorFixedRatio(nn.Module):
    """Fixed top-K threshold — ignores all inputs, reads scores from ``_scores``.

    The caller (a ``SwinBlockV3_FixedRatio``) sets ``self._kl_scores`` or
    ``self._inc_scores`` on the predictor *before* calling ``forward()``.
    The forward method ranks the stored scores and returns the k-th largest
    as the threshold τ.
    """

    def __init__(self, keep_ratio=0.3):
        super().__init__()
        self.keep_ratio = keep_ratio
        self._kl_scores = None
        self._inc_scores = None

    def forward(self, soft_hist, C_b, stage_idx, block_idx):
        # Determine which score tensor was set last
        scores = self._kl_scores if self._kl_scores is not None else self._inc_scores
        if scores is None:
            return torch.tensor([[0.5]], device=soft_hist.device)
        B, N = scores.shape
        k = max(1, int(self.keep_ratio * N))
        tau = scores.topk(k, dim=1)[0][:, -1:]  # (B, 1)
        return tau.detach()  # no grad through top-k threshold


# ==============================================================================
# Fixed-Ratio Block (only change: top-k threshold instead of MLP predictor)
# ==============================================================================

from .swin_baseline_v3 import SwinBlockV3, SwinBlockSequenceV3
import torch.nn.functional as F


class SwinBlockV3_FixedRatio(SwinBlockV3):
    """Same as SwinBlockV3, but threshold = k-th largest score (fixed ratio)."""

    def __init__(self, *args, fixed_keep_ratio=0.3, **kwargs):
        kwargs.pop('fixed_keep_ratio', None)  # not accepted by parent
        super().__init__(*args, **kwargs)
        self.fixed_keep_ratio = fixed_keep_ratio

    # ---- override threshold calls in _forward_kl ------------------------------
    def _forward_kl(self, x, hw_shape, C_b=None):
        """Copy of parent._forward_kl with predictor call → top-k."""
        B, L, C = x.shape
        H, W = hw_shape
        x = x.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b), mode='replicate')
        H_pad, W_pad = x.shape[1], x.shape[2]
        shifted_x = (
            x if self.shift_size == 0
            else torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        )
        x_windows = self._window_partition(shifted_x)
        total_windows = x_windows.shape[0]
        N_win = total_windows // B
        from .swin_baseline_v3 import compute_window_relative_entropy, _collect_kl_scores, compute_soft_histogram
        window_scores = compute_window_relative_entropy(x_windows, B, self.window_size, H=H, W=W)
        window_scores_reshaped = window_scores.view(B, -1)
        _collect_kl_scores(window_scores_reshaped, self.stage_idx, self.block_idx)
        scores_min = window_scores_reshaped.min(dim=1, keepdim=True)[0]
        scores_max = window_scores_reshaped.max(dim=1, keepdim=True)[0]
        kl_norm_scores = (window_scores_reshaped - scores_min) / (scores_max - scores_min + 1e-8)
        # ---- fixed-ratio threshold (only changed part) ----
        k = max(1, int(self.fixed_keep_ratio * N_win))
        kl_threshold = kl_norm_scores.topk(k, dim=1)[0][:, -1:]  # (B, 1)
        # ---- rest is identical to parent ----
        soft_hist = compute_soft_histogram(kl_norm_scores.view(-1), B, K=16)
        m_mask_kl = torch.sigmoid((kl_norm_scores - kl_threshold.unsqueeze(-1)) / self.temperature)
        kl_keep_mask = kl_norm_scores > kl_threshold
        kl_keep_idx_flat = torch.nonzero(kl_keep_mask.view(-1)).squeeze(-1)
        x_kl = x_windows.view(-1, self.window_size * self.window_size, C)[kl_keep_idx_flat]
        m_mask_kept = m_mask_kl.view(-1)[kl_keep_idx_flat].unsqueeze(-1).unsqueeze(-1)
        x_kl = x_kl * m_mask_kept
        identity_attn = x_kl
        x_kl = self.norm1(x_kl)
        x_kl = self.attn.w_msa(x_kl)
        x_kl = identity_attn + x_kl
        cur_entropy_local = self._compute_entropy(x_kl)
        x_windows_new = x_windows.clone()
        x_kl_reshaped = x_kl.view(-1, self.window_size, self.window_size, C)
        x_windows_new[kl_keep_idx_flat] = x_kl_reshaped
        full_entropy = torch.zeros(total_windows, device=x.device)
        full_entropy[kl_keep_idx_flat] = cur_entropy_local.detach()
        attn_windows = x_windows_new.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, H_pad, W_pad)
        x = (shifted_x if self.shift_size == 0
             else torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)))
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        block_gate_loss = None
        from .swin_baseline_v3 import compute_gate_loss, collect_threshold
        if self.use_learnable_gate and self.kl_predictor is not None:
            kl_loss, kl_info, kl_reg = compute_gate_loss(m_mask_kl.view(-1), kl_norm_scores.view(-1), self.lambda_kl)
            self.kl_gate_loss = kl_loss
            for b in range(B):
                collect_threshold(kl_threshold[b].item(), self.stage_idx, self.block_idx, 'kl')
            block_gate_loss = (kl_loss, None, kl_info, kl_reg, None, None)
        self._keep_stats = {'kl_keep': kl_keep_idx_flat.shape[0],
                            'final_keep': kl_keep_idx_flat.shape[0],
                            'total': total_windows, 'kl_tau': kl_threshold.mean().item()}
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return (x, full_entropy, x_windows, x_windows_new, window_scores_reshaped,
                (H_pad, W_pad), self.embed_dims, kl_keep_idx_flat, block_gate_loss)

    # ---- override threshold calls in _forward_kl_inc --------------------------
    def _forward_kl_inc(self, x, hw_shape, entropy_cache=None,
                        prev_aligned_entropy=None, prev_kl_keep_idx=None, C_b=None):
        """Copy of parent._forward_kl_inc with predictor calls → top-k."""
        if self.inc_ratio is None:
            return self._forward_kl(x, hw_shape)
        B, L, C = x.shape
        H, W = hw_shape
        x = x.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b), mode='replicate')
        H_pad, W_pad = x.shape[1], x.shape[2]
        shifted_x = (x if self.shift_size == 0
                     else torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)))
        x_windows = self._window_partition(shifted_x)
        total_windows = x_windows.shape[0]
        N_win = total_windows // B
        if self.kl_ratio is None:
            base_result = self._forward_base_with_pad(x, hw_shape, pad_r, pad_b)
            return base_result + (None,) * (9 - len(base_result))
        from .swin_baseline_v3 import (compute_window_relative_entropy, _collect_kl_scores,
                                        compute_soft_histogram, canonical_to_shifted,
                                        _collect_inc_scores, compute_gate_loss, collect_threshold)
        window_scores = compute_window_relative_entropy(x_windows, B, self.window_size, H=H, W=W)
        window_scores_reshaped = window_scores.view(B, -1)
        _collect_kl_scores(window_scores_reshaped, self.stage_idx, self.block_idx)
        scores_min = window_scores_reshaped.min(dim=1, keepdim=True)[0]
        scores_max = window_scores_reshaped.max(dim=1, keepdim=True)[0]
        kl_norm_scores = (window_scores_reshaped - scores_min) / (scores_max - scores_min + 1e-8)
        # ---- KL fixed-ratio ----
        k_kl = max(1, int(self.fixed_keep_ratio * N_win))
        kl_threshold = kl_norm_scores.topk(k_kl, dim=1)[0][:, -1:]
        soft_hist = compute_soft_histogram(kl_norm_scores.view(-1), B, K=16)
        tau_tiled = kl_threshold.squeeze(-1).unsqueeze(-1)
        m_mask_kl = torch.sigmoid((kl_norm_scores - tau_tiled) / self.temperature)
        m_mask_kl = torch.clamp(m_mask_kl, min=1e-6, max=1 - 1e-6)
        kl_keep_mask = kl_norm_scores > kl_threshold
        fallback = torch.zeros_like(kl_keep_mask)
        fallback[torch.arange(B, device=x.device), window_scores_reshaped.argmax(dim=1)] = True
        kl_keep_mask = kl_keep_mask | fallback
        kl_keep_idx_flat = torch.nonzero(kl_keep_mask.view(-1)).squeeze(-1)
        x_kl = x_windows.view(-1, self.window_size * self.window_size, C)[kl_keep_idx_flat]
        m_mask_kept = m_mask_kl.view(-1)[kl_keep_idx_flat].unsqueeze(-1).unsqueeze(-1)
        x_kl = x_kl * m_mask_kept
        identity_attn = x_kl
        x_kl = self.norm1(x_kl)
        x_kl = self.attn.w_msa(x_kl)
        x_kl = identity_attn + x_kl
        cur_entropy = self._compute_entropy(x_kl)
        x_windows_flat = x_windows.view(-1, self.window_size * self.window_size, C)
        full_entropy_before_kl = self._compute_entropy(x_windows_flat)
        x_windows_new = x_windows.clone()
        x_kl_reshaped = x_kl.view(-1, self.window_size, self.window_size, C)
        x_windows_new[kl_keep_idx_flat] = x_kl_reshaped
        # INC
        inc_scores_survived = torch.zeros_like(cur_entropy)
        if entropy_cache is not None and entropy_cache.shape[0] == total_windows:
            cache_src = (canonical_to_shifted(entropy_cache, B, H_pad, W_pad, self.window_size, self.shift_size)
                         if self.shift_size > 0 else entropy_cache)
            inc_scores_survived = torch.abs(cur_entropy - cache_src[kl_keep_idx_flat].detach())
        elif prev_aligned_entropy is not None and prev_aligned_entropy.shape[0] == total_windows:
            aligned_src = (canonical_to_shifted(prev_aligned_entropy, B, H_pad, W_pad, self.window_size, self.shift_size)
                           if self.shift_size > 0 else prev_aligned_entropy)
            inc_scores_survived = torch.abs(cur_entropy - aligned_src[kl_keep_idx_flat].detach())
        inc_scores_full = torch.zeros(B, N_win, device=x.device)
        inc_scores_full.view(-1)[kl_keep_idx_flat] = inc_scores_survived
        _collect_inc_scores(inc_scores_full, self.stage_idx, self.block_idx)
        inc_for_max = torch.where(kl_keep_mask, inc_scores_full, torch.tensor(-1e9, device=x.device))
        inc_for_min = torch.where(kl_keep_mask, inc_scores_full, torch.tensor(1e9, device=x.device))
        inc_max = inc_for_max.amax(dim=1, keepdim=True)
        inc_min = inc_for_min.amin(dim=1, keepdim=True)
        inc_range = (inc_max - inc_min).clamp(min=1e-6)
        inc_norm_scores = (inc_scores_full - inc_min) / inc_range
        inc_norm_scores = torch.where(kl_keep_mask, inc_norm_scores, torch.tensor(0.0, device=x.device))
        # ---- INC fixed-ratio ----
        k_inc = max(1, int(self.fixed_keep_ratio * N_win))
        inc_threshold = inc_norm_scores.topk(k_inc, dim=1)[0][:, -1:]
        soft_hist = compute_soft_histogram(inc_norm_scores.view(-1), B, K=16)
        tau_tiled = inc_threshold.squeeze(-1).unsqueeze(-1)
        m_mask_inc = torch.sigmoid((inc_norm_scores - tau_tiled) / self.temperature)
        m_mask_inc = torch.clamp(m_mask_inc, min=1e-6, max=1 - 1e-6)
        m_mask_inc = torch.where(kl_keep_mask, m_mask_inc, torch.tensor(0.0, device=x.device))
        inc_keep_mask = (inc_norm_scores > inc_threshold) & kl_keep_mask
        inc_scores_for_fb = inc_scores_full.clone()
        inc_scores_for_fb[~kl_keep_mask] = -1.0
        inc_fallback = torch.zeros_like(inc_keep_mask)
        inc_fallback[torch.arange(B, device=x.device), inc_scores_for_fb.argmax(dim=1)] = True
        inc_keep_mask = inc_keep_mask | inc_fallback
        if self.enable_selection_vis:
            self.selection_vis = {
                'kl_mask': kl_keep_mask[0].detach().cpu(), 'inc_mask': inc_keep_mask[0].detach().cpu(),
                'kl_norm': kl_norm_scores[0].detach().cpu(), 'inc_norm': inc_norm_scores[0].detach().cpu(),
                'H': H, 'W': W, 'H_pad': H_pad, 'W_pad': W_pad,
                'window_size': self.window_size, 'shift_size': self.shift_size,
            }
        final_keep_idx_flat = torch.nonzero(inc_keep_mask.view(-1)).squeeze(-1)
        x_ffn = x_windows_new.view(-1, self.window_size * self.window_size, C)[final_keep_idx_flat]
        m_mask_inc_kept = m_mask_inc.view(-1)[final_keep_idx_flat].unsqueeze(-1).unsqueeze(-1)
        x_ffn = x_ffn * m_mask_inc_kept
        identity_ffn = x_ffn
        x_ffn = self.norm2(x_ffn)
        x_ffn = self.ffn(x_ffn, identity=identity_ffn)
        x_ffn_reshaped = x_ffn.view(-1, self.window_size, self.window_size, C)
        x_windows_new.view(-1, self.window_size, self.window_size, C)[final_keep_idx_flat] = x_ffn_reshaped
        attn_windows = x_windows_new.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, H_pad, W_pad)
        x = (shifted_x if self.shift_size == 0
             else torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)))
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        block_gate_loss = None
        if self.use_learnable_gate:
            kl_loss, kl_info, kl_reg = compute_gate_loss(m_mask_kl.view(-1), kl_norm_scores.view(-1), self.lambda_kl)
            self.kl_gate_loss = kl_loss
            inc_loss = inc_info = inc_reg = None
            if inc_scores_full.numel() > 0:
                inc_loss, inc_info, inc_reg = compute_gate_loss(m_mask_inc.view(-1), inc_norm_scores.view(-1), self.lambda_inc, valid_mask=kl_keep_mask.view(-1))
                self.inc_gate_loss = inc_loss
            for b in range(B):
                collect_threshold(kl_threshold[b].item(), self.stage_idx, self.block_idx, 'kl')
                collect_threshold(inc_threshold[b].item(), self.stage_idx, self.block_idx, 'inc')
            block_gate_loss = (kl_loss, inc_loss, kl_info, kl_reg, inc_info, inc_reg)
        self._keep_stats = {'kl_keep': kl_keep_idx_flat.shape[0],
                            'final_keep': final_keep_idx_flat.shape[0],
                            'total': total_windows, 'kl_tau': kl_threshold.mean().item(),
                            'inc_tau': inc_threshold.mean().item()}
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return (x, full_entropy_before_kl, x_windows, x_windows_new, window_scores_reshaped,
                (H_pad, W_pad), self.embed_dims, kl_keep_idx_flat, block_gate_loss)


class SwinBlockSequenceV3_FixedRatio(SwinBlockSequenceV3):
    """Sequence that uses FixedRatio blocks."""


# ==============================================================================
# Backbone subclasses  (each only replaces predictor creation)
# ==============================================================================

def _reassign_predictors(backbone):
    """Push backbone-level predictors down to every pruning block."""
    for stage in backbone.stages:
        for block in stage.blocks:
            if hasattr(block, 'kl_predictor'):
                block.kl_predictor = backbone.kl_predictor
            if hasattr(block, 'inc_predictor'):
                block.inc_predictor = backbone.inc_predictor


@MODELS.register_module()
class SwinTransformerV3_NoDist(SwinTransformerV3):
    """Ablation: no soft-histogram (score distribution) input."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.use_learnable_gate:
            self.kl_predictor = PredictorNoDist()
            self.inc_predictor = PredictorNoDist()
            _reassign_predictors(self)


@MODELS.register_module()
class SwinTransformerV3_NoCb(SwinTransformerV3):
    """Ablation: no image-complexity (C_b) input."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.use_learnable_gate:
            self.kl_predictor = PredictorNoCb()
            self.inc_predictor = PredictorNoCb()
            _reassign_predictors(self)


@MODELS.register_module()
class SwinTransformerV3_NoPos(SwinTransformerV3):
    """Ablation: no stage / block position embedding."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.use_learnable_gate:
            self.kl_predictor = PredictorNoPos()
            self.inc_predictor = PredictorNoPos()
            _reassign_predictors(self)


@MODELS.register_module()
class SwinTransformerV3_FixedRatio(SwinTransformerV3):
    """Ablation: fixed top-K ratio (no learned threshold).

    Config must supply ``fixed_keep_ratio`` in backbone dict, e.g.::

        backbone=dict(
            type='SwinTransformerV3_FixedRatio',
            fixed_keep_ratio=0.3,
            ...
        )
    """

    def __init__(self, fixed_keep_ratio=0.3, **kwargs):
        self._fixed_keep_ratio = fixed_keep_ratio
        # Disable learnable gate — we'll use fixed ratio blocks instead
        kwargs['use_learnable_gate'] = False
        super().__init__(**kwargs)
        self.use_learnable_gate = True  # re-enable for gate loss collection
        self.kl_predictor = PredictorFixedRatio(keep_ratio=fixed_keep_ratio)
        self.inc_predictor = PredictorFixedRatio(keep_ratio=fixed_keep_ratio)
        self.kl_predictor_gate_loss = torch.zeros(1)
        self.inc_predictor_gate_loss = torch.zeros(1)
        # Build with FixedRatio blocks
        self._replace_blocks_fixed_ratio()

    def _replace_blocks_fixed_ratio(self):
        """Rebuild stages with ``SwinBlockV3_FixedRatio`` blocks."""
        for si in range(len(self.stages)):
            stage = self.stages[si]
            first_block = stage.blocks[0]
            embed_dims = first_block.attn.w_msa.embed_dims
            num_heads = first_block.attn.w_msa.num_heads
            ff_channels = first_block.ffn.layers[1].out_features
            ws = first_block.attn.w_msa.window_size[0]
            drop_paths = []
            for b in stage.blocks:
                dp = 0.0
                for child in b.attn.children():
                    if hasattr(child, 'drop_prob'): dp = child.drop_prob; break
                drop_paths.append(dp)

            kl_cfg = self.stage_config.get(si, {})
            inc_cfg = self.inc_stage_config.get(si, {})
            kl_blocks_set = set(kl_cfg.get('blocks', []))
            inc_blocks_set = set(inc_cfg.get('blocks', []))
            kl_r = kl_cfg.get('ratio', [])
            inc_r = inc_cfg.get('inc_ratio', [])
            if isinstance(kl_r, (int, float)): kl_r = [kl_r]
            if isinstance(inc_r, (int, float)): inc_r = [inc_r]

            depth = len(stage.blocks)
            blocks = nn.ModuleList()
            for i in range(depth):
                kl_ratio = None
                if i in kl_blocks_set:
                    idx = list(kl_blocks_set).index(i) if i in kl_blocks_set else -1
                    # get ratio by position in config's sorted list
                    sorted_kl = sorted(kl_blocks_set)
                    pos = sorted_kl.index(i) if i in sorted_kl else -1
                    kl_ratio = kl_r[pos] if pos >= 0 and pos < len(kl_r) else (kl_r[-1] if kl_r else None)
                inc_ratio = None
                if i in inc_blocks_set:
                    sorted_inc = sorted(inc_blocks_set)
                    pos = sorted_inc.index(i) if i in sorted_inc else -1
                    inc_ratio = inc_r[pos] if pos >= 0 and pos < len(inc_r) else (inc_r[-1] if inc_r else None)

                block = SwinBlockV3_FixedRatio(
                    embed_dims=embed_dims, num_heads=num_heads,
                    feedforward_channels=ff_channels, window_size=ws,
                    shift=False if i % 2 == 0 else True,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=drop_paths[i],
                    kl_ratio=kl_ratio, inc_ratio=inc_ratio,
                    strategy='kl_inc', stage_idx=si, block_idx=i,
                    kl_predictor=self.kl_predictor, inc_predictor=self.inc_predictor,
                    use_learnable_gate=True, temperature=self.temperature,
                    lambda_kl=self.lambda_kl, lambda_inc=self.lambda_inc,
                    fixed_keep_ratio=self._fixed_keep_ratio,
                )
                block.norm1 = stage.blocks[i].norm1
                block.norm2 = stage.blocks[i].norm2
                block.attn = stage.blocks[i].attn
                block.ffn = stage.blocks[i].ffn
                blocks.append(block)

            new_seq = SwinBlockSequenceV3_FixedRatio(
                embed_dims=embed_dims, num_heads=num_heads,
                feedforward_channels=ff_channels, depth=depth, window_size=ws,
                block_kl_ratios=[None] * depth, block_inc_ratios=[None] * depth,
                strategy='kl_inc', stage_idx=si,
                kl_predictor=self.kl_predictor, inc_predictor=self.inc_predictor,
                use_learnable_gate=True, temperature=self.temperature,
                lambda_kl=self.lambda_kl, lambda_inc=self.lambda_inc,
                downsample=stage.downsample,
            )
            new_seq.blocks = blocks
            new_seq.downsample = stage.downsample
            self.stages[si] = new_seq

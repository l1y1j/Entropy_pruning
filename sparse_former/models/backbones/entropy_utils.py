"""Entropy-based scoring utilities for sparse computation.

Pure functions extracted and adapted from swin_baseline_v3.py.
No dependencies on other project modules — only torch + torch.nn.functional.
"""

import torch
import torch.nn.functional as F


# ===========================================================================
# 1. KL Divergence Score
# ===========================================================================
def compute_window_relative_entropy(x_windows, B, window_size=7):
    """Compute per-window KL divergence against global distribution.

    For each window: compute softmax distribution over channels (after mean-pooling
    tokens), then measure KL(window || global_mean).  High KL = window is "unusual"
    = more informative = should be kept.

    Args:
        x_windows: (B*N, ws, ws, C)  window-partitioned features
        B:         batch size
        window_size: window spatial size (default 7)

    Returns:
        kl_scores: (B*N,)  KL divergence score per window
    """
    total_windows, ws_h, ws_w, C = x_windows.shape
    N = total_windows // B

    # (B*N, ws, ws, C) → (B, N, ws*ws, C)
    x_windows = x_windows.view(B, N, window_size * window_size, C)
    # mean over tokens → (B, N, C) → softmax over channels
    local_dist = F.softmax(x_windows.mean(dim=2), dim=-1)       # (B, N, C)
    # global distribution = mean over windows
    global_dist = local_dist.mean(dim=1, keepdim=True)          # (B, 1, C)
    # KL(window || global) summed over channels
    kl = (local_dist * torch.log((local_dist + 1e-8) / (global_dist + 1e-8))).sum(dim=-1)

    return kl.view(-1)  # (B*N,)


# ===========================================================================
# 2. Variance Score
# ===========================================================================
def compute_window_variance(x_windows, B, window_size=7):
    """Compute per-window variance score (sparseformer formula).

    Follows the same pattern as block_entropy.py's variance mode:
      1. Mean-pool tokens within each window -> (B, N_win, C)
      2. Softmax over channels -> channel probability distribution
      3. Variance over channels -> variance score per window

    High variance = peaked channel distribution = complex local texture.
    This is a purely local statistic with no global context, so it tends to
    flag repetitive but textured backgrounds as "important".

    Args:
        x_windows: (B*N, ws, ws, C)  window-partitioned features
        B:         batch size
        window_size: window spatial size (default 7)

    Returns:
        var_scores: (B*N,)  variance score per window
    """
    total_windows, ws_h, ws_w, C = x_windows.shape
    N = total_windows // B

    # (B*N, ws, ws, C) -> (B, N, ws*ws, C)
    x = x_windows.view(B, N, window_size * window_size, C)
    # mean over tokens -> (B, N, C) -> softmax over channels
    window_dist = F.softmax(x.mean(dim=2), dim=-1)       # (B, N, C)
    # variance of channel distribution per window
    var_scores = window_dist.var(dim=-1)                  # (B, N)

    return var_scores.view(-1)  # (B*N,)


# ===========================================================================
# 3. Soft Histogram
# ===========================================================================
def compute_soft_histogram(scores_flat, B, K=16, sigma=0.08):
    """Map window scores to a differentiable soft histogram.

    Uses Gaussian kernels to softly assign each window to K bins.
    The resulting (B, K) vector captures the shape of the score distribution,
    which the ThresholdPredictor uses to set an appropriate threshold.

    Args:
        scores_flat: (B*N,)  scores already normalized to [0, 1]
        B:           batch size
        K:           number of histogram bins (default 16)
        sigma:       Gaussian kernel width (default 0.08)

    Returns:
        hist: (B, K)  soft histogram, sum ≈ 1
    """
    scores_2d = scores_flat.detach().view(B, -1)  # (B, N)

    # bin centers: (j+0.5)/K for j=0..K-1  →  [0.03125, ..., 0.96875]
    centers = torch.linspace(0.5 / K, 1.0 - 0.5 / K, K, device=scores_2d.device)

    # (B, N, 1) - (K,)  →  (B, N, K)
    dist = scores_2d.unsqueeze(-1) - centers.view(1, 1, K)

    # Gaussian kernel
    w = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    # normalize each window's contribution across bins
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

    # average over all windows → distribution shape
    hist = w.mean(dim=1)  # (B, K), sum ≈ 1

    return hist


# ===========================================================================
# 4. Gate Loss (log-barrier)
# ===========================================================================
def compute_gate_loss(m_mask, scores, lambda_reg, valid_mask=None):
    """Log-barrier gate loss for entropy pruning.

        L = -log(R_info) + lambda_reg * R_act

    - -log(R_info):  information barrier — prevents over-pruning.
      When R_info → 0, loss → ∞  (hard barrier).
    - lambda_reg * R_act:  sparsity penalty — encourages lower activation.

    Gradient adaptive behaviour:
      - High R_info (information-rich):  mild gradient, allows pruning exploration.
      - Low R_info (information-poor):  steep gradient, forces retention.

    Args:
        m_mask:     soft mask, shape (N,)
        scores:     normalized scores, shape (N,), already detached
        lambda_reg: sparsity penalty coefficient (sole hyperparameter)
        valid_mask: boolean mask of valid windows, shape (N,) or None

    Returns:
        loss:       scalar gate loss
        info_ratio: information retention ratio (monitoring, higher=better)
        act_ratio:  activation ratio (monitoring, lower=more pruning)
    """
    scores = scores.detach()

    if valid_mask is None:
        valid_mask = torch.ones_like(m_mask, dtype=torch.bool)

    N_valid = valid_mask.sum()
    if N_valid == 0:
        zero = torch.tensor(0.0, device=scores.device)
        return zero, zero, zero

    # Information retention ratio:  Σ(m_i * S_i) / Σ(S_i)
    reserved_sum = (m_mask * scores * valid_mask.float()).sum()
    total_sum = (scores * valid_mask.float()).sum()

    if not torch.isfinite(total_sum) or total_sum.abs() < 1e-8:
        total_sum = torch.tensor(1e-8, device=scores.device)

    info_ratio = reserved_sum / (total_sum + 1e-8)

    # Activation ratio:  mean(m_i) over valid windows
    act_ratio = (m_mask * valid_mask.float()).sum() / N_valid

    # Log-barrier loss
    info_barrier = -torch.log(info_ratio.clamp(min=1e-8))
    sparsity_term = lambda_reg * act_ratio
    loss = info_barrier + sparsity_term

    if not torch.isfinite(loss):
        loss = torch.tensor(0.0, device=scores.device, requires_grad=True)

    return loss, info_ratio.detach(), act_ratio.detach()


# ===========================================================================
# 5. Image Complexity C_b
# ===========================================================================
def compute_erb_from_conv(x_conv, window_size=7):
    """Compute image-level complexity C_b from raw Conv output (pre-LayerNorm).

    LN normalizes magnitudes, erasing the per-region energy differences that
    indicate spatial complexity. We intercept the Conv output before LN and
    compute window-energy entropy as a proxy for image complexity.

    C_b = H(window_energies) / log(N_windows)  ∈  [0, 1]

    - C_b ≈ 0:  most energy concentrated in few windows → simple image
    - C_b ≈ 1:  energy uniformly distributed → complex image

    Args:
        x_conv:     (B, C, H, W)  raw Conv2d output from patch_embed.projection
        window_size: window size for energy aggregation (default 7)

    Returns:
        C_b:  (B, 1)  image complexity, detached
    """
    B, C, H, W = x_conv.shape

    # Pad H/W to multiples of window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x_conv = F.pad(x_conv, (0, pad_w, 0, pad_h))

    H_pad, W_pad = H + pad_h, W + pad_w
    H_w, W_w = H_pad // window_size, W_pad // window_size
    N_w = H_w * W_w  # number of windows

    # Partition into windows: (B, C, H_pad, W_pad) → (B, N_w, ws*ws, C)
    x_conv = x_conv.view(B, C, H_w, window_size, W_w, window_size)
    x_conv = x_conv.permute(0, 2, 4, 3, 5, 1).contiguous()
    x_windows = x_conv.view(B, N_w, window_size * window_size, C)

    # Per-window energy: mean L2-norm-squared of tokens
    token_norm_sq = (x_windows ** 2).sum(dim=-1)           # (B, N_w, M)
    E_i = token_norm_sq.mean(dim=-1)                        # (B, N_w)

    # Energy distribution
    p_i = E_i / (E_i.sum(dim=-1, keepdim=True) + 1e-8)     # (B, N_w)

    # Entropy of energy distribution
    H_ent = -(p_i * torch.log(p_i + 1e-8)).sum(dim=-1)     # (B,)

    # Normalize:  C_b = H / H_max  ∈  [0, 1]
    H_max = torch.log(torch.tensor(N_w, dtype=torch.float32, device=x_conv.device))
    C_b = (H_ent / H_max).unsqueeze(-1)                     # (B, 1)

    return C_b.detach()


# ===========================================================================
# 6. Min-Max Normalization
# ===========================================================================
def min_max_normalize(scores_2d):
    """Min-Max normalize to [0, 1] along the window dimension.

    Args:
        scores_2d: (B, N)  raw score matrix

    Returns:
        norm_scores: (B, N)  normalized scores ∈ [0, 1]
    """
    s_min = scores_2d.min(dim=1, keepdim=True)[0]
    s_max = scores_2d.max(dim=1, keepdim=True)[0]
    return (scores_2d - s_min) / (s_max - s_min + 1e-8)

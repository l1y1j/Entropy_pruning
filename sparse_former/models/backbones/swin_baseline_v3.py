import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from mmcv.cnn import build_norm_layer
from mmdet.registry import MODELS
from mmdet.models.backbones.swin import (
    SwinTransformer,
    SwinBlock,
    SwinBlockSequence,
    ShiftWindowMSA,
    WindowMSA,
)


# KL/INC 分数收集器（只收集前10张图）
_kl_img_counter = 0
_kl_img_limit = 10
_kl_scores_collector = []

_inc_img_counter = 0
_inc_img_limit = 10
_inc_scores_collector = []

_current_epoch = 0
_enabled_collections = set()  # {(stage_idx, block_idx, 'kl'), ...}

# 存储每个stage/block的threshold
_kl_threshold_collector = {}  # {(stage_idx, block_idx): [threshold_values]}
_inc_threshold_collector = {}

# KL/INC 分数导出路径（默认为同级目录，可通过 set_export_base_path 设置）
_export_base_path = None


def set_export_base_path(path):
    """设置 KL/INC 分数导出的基础路径（通常设为 work_dir）"""
    global _export_base_path
    _export_base_path = path


def _collect_kl_scores(window_scores, stage_idx, block_idx):
    """收集KL分数用于后续分析"""
    global _kl_img_counter, _kl_img_limit, _kl_scores_collector, _current_epoch
    key = (stage_idx, block_idx, "kl")
    if key not in _enabled_collections:
        return
    if _kl_img_counter < _kl_img_limit:
        _kl_scores_collector.append(
            {
                "data": window_scores.detach().cpu().numpy(),
                "shape": window_scores.shape,
                "stage_idx": stage_idx,
                "block_idx": block_idx,
                "epoch": _current_epoch,
                "strategy": "kl",
            }
        )
        _kl_img_counter += 1

    if _kl_img_counter >= _kl_img_limit and len(_kl_scores_collector) > 0:
        _export_kl_scores()


def _collect_inc_scores(inc_scores, stage_idx, block_idx):
    """收集INC分数用于后续分析"""
    global _inc_img_counter, _inc_img_limit, _inc_scores_collector, _current_epoch
    key = (stage_idx, block_idx, "inc")
    if key not in _enabled_collections:
        return
    if _inc_img_counter < _inc_img_limit:
        _inc_scores_collector.append(
            {
                "data": inc_scores.detach().cpu().numpy(),
                "shape": inc_scores.shape,
                "stage_idx": stage_idx,
                "block_idx": block_idx,
                "epoch": _current_epoch,
                "strategy": "inc",
            }
        )
        _inc_img_counter += 1

    if _inc_img_counter >= _inc_img_limit and len(_inc_scores_collector) > 0:
        _export_inc_scores()


def collect_threshold(threshold_value, stage_idx, block_idx, strategy):
    """收集每个stage/block的threshold用于后续分析"""
    global _current_epoch
    key = (stage_idx, block_idx, strategy)
    if key not in _enabled_collections:
        return
    if strategy == "kl":
        if key not in _kl_threshold_collector:
            _kl_threshold_collector[key] = []
        _kl_threshold_collector[key].append(
            {
                "threshold": float(threshold_value),
                "stage_idx": stage_idx,
                "block_idx": block_idx,
                "epoch": _current_epoch,
            }
        )
    elif strategy == "inc":
        if key not in _inc_threshold_collector:
            _inc_threshold_collector[key] = []
        _inc_threshold_collector[key].append(
            {
                "threshold": float(threshold_value),
                "stage_idx": stage_idx,
                "block_idx": block_idx,
                "epoch": _current_epoch,
            }
        )


def _get_export_dir():
    """获取导出目录路径"""
    base = _export_base_path if _export_base_path else os.path.dirname(__file__)
    return os.path.join(base, "kl_scores_export", f"epoch_{_current_epoch:03d}")


def _export_kl_scores():
    """导出收集的KL分数到文件"""
    global _kl_scores_collector, _current_epoch, _kl_threshold_collector
    export_dir = _get_export_dir()
    if _kl_scores_collector:
        os.makedirs(export_dir, exist_ok=True)
        save_path = os.path.join(export_dir, "kl_scores.pkl")

        import pickle

        with open(save_path, "wb") as f:
            pickle.dump(_kl_scores_collector, f)

        _kl_scores_collector.clear()

    # 导出 KL thresholds
    if _kl_threshold_collector:
        os.makedirs(export_dir, exist_ok=True)
        save_path = os.path.join(export_dir, "kl_thresholds.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(_kl_threshold_collector, f)
        _kl_threshold_collector.clear()


def _export_inc_scores():
    """导出收集的INC分数到文件"""
    global _inc_scores_collector, _current_epoch, _inc_threshold_collector
    export_dir = _get_export_dir()
    if _inc_scores_collector:
        os.makedirs(export_dir, exist_ok=True)
        save_path = os.path.join(export_dir, "inc_scores.pkl")

        import pickle

        with open(save_path, "wb") as f:
            pickle.dump(_inc_scores_collector, f)

        _inc_scores_collector.clear()

    # 导出 INC thresholds
    if _inc_threshold_collector:
        os.makedirs(export_dir, exist_ok=True)
        save_path = os.path.join(export_dir, "inc_thresholds.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(_inc_threshold_collector, f)
        _inc_threshold_collector.clear()


def set_epoch(epoch):
    """供外部调用设置当前epoch"""
    global _current_epoch
    _current_epoch = epoch


def register_enabled_collections(enabled_set):
    """供外部调用注册需要收集的stage/block组合"""
    global _enabled_collections
    _enabled_collections = enabled_set


def reset_collectors():
    """重置计数器，在每个epoch开始时调用"""
    global \
        _kl_img_counter, \
        _inc_img_counter, \
        _kl_threshold_collector, \
        _inc_threshold_collector
    _kl_img_counter = 0
    _inc_img_counter = 0
    _kl_threshold_collector.clear()
    _inc_threshold_collector.clear()


def compute_window_relative_entropy(
    x_windows: torch.Tensor, B: int, window_size: int = 7
) -> torch.Tensor:
    """Compute KL divergence for each window

    Args:
        x_windows: (total_windows, window_size, window_size, C)
        B: batch size
        window_size: window size

    Returns:
        kl: (B * N_win,) KL score for each window
    """
    total_windows, _, _, C = x_windows.shape
    N_win = total_windows // B

    x_windows = x_windows.view(B, N_win, window_size * window_size, C)
    local_dist = F.softmax(x_windows.mean(dim=2), dim=-1)
    global_dist = local_dist.mean(dim=1, keepdim=True)
    # kl = (local_dist * torch.log(local_dist / (global_dist + 1e-8))).sum(dim=-1)
    kl = (local_dist * torch.log((local_dist + 1e-8) / (global_dist + 1e-8))).sum(
        dim=-1
    )

    return kl.view(-1)


# ========== 可学习门控阈值相关 ==========


class ThresholdPredictor(nn.Module):
    """预测KL或INC的门控阈值 - 改进版使用Embedding位置编码"""

    def __init__(self, hidden_dim=64, num_stages=4, num_blocks=8):
        super().__init__()
        # 统计量直接线性映射（不再用LayerNorm）
        self.stat_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
        )
        # 强位置编码 - Embedding替代归一化标量
        self.stage_emb = nn.Embedding(num_stages, 16)
        self.block_emb = nn.Embedding(num_blocks, 16)
        # 融合网络
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim + 32),
            nn.ReLU(),
            nn.Linear(hidden_dim + 32, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            # 无 Sigmoid：输出 τ_raw ∈ (-∞, +∞)，sigmoid 移至外部与 stage_policy 合并
        )

    def forward(
        self, stats_mean, stats_std, stats_p50, stats_max, stage_idx, block_idx
    ):
        """
        Args:
            stats_mean: (B, 1) 分数均值
            stats_std: (B, 1) 分数标准差
            stats_p50: (B, 1) 分数中位数
            stats_max: (B, 1) 分数最大值
            stage_idx: int 或 (B,) 当前stage索引
            block_idx: int 或 (B,) 当前block索引

        Returns:
            threshold: (B, 1) 预测的阈值τ
        """
        # 处理 stage_idx 和 block_idx 类型（可能是Python int或tensor）
        if not isinstance(stage_idx, torch.Tensor):
            stage_idx = torch.tensor([stage_idx], device=stats_mean.device, dtype=torch.long)
            if stats_mean.shape[0] > 1:
                stage_idx = stage_idx.expand(stats_mean.shape[0])
        if not isinstance(block_idx, torch.Tensor):
            block_idx = torch.tensor([block_idx], device=stats_mean.device, dtype=torch.long)
            if stats_mean.shape[0] > 1:
                block_idx = block_idx.expand(stats_mean.shape[0])

        # 统计量直接concat，不做LayerNorm
        stats = torch.cat([stats_mean, stats_std, stats_p50, stats_max], dim=1)  # (B, 4)
        h_stat = self.stat_encoder(stats)  # (B, hidden_dim)

        # Embedding位置编码
        h_stage = self.stage_emb(stage_idx)  # (B, 16)
        h_block = self.block_emb(block_idx)   # (B, 16)

        # 融合
        h = torch.cat([h_stat, h_stage, h_block], dim=1)
        return self.net(h)  # (B, 1)


# ==========================================================================
# 全局调试计数器：用于监控 mask 均值变化
# ==========================================================================
_mask_debug_counter = 0


def compute_soft_mask(
    scores: torch.Tensor, threshold: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """计算软掩码 m_i = sigmoid((S_i - tau) / T)

    Args:
        scores: (N,) 窗口分数
        threshold: (B, 1) 预测的阈值
        temperature: float 温度参数

    Returns:
        m_mask: (N,) 软掩码，值在0~1之间
    """
    threshold_expanded = threshold.squeeze(-1)  # (B,)
    B = threshold_expanded.shape[0]
    N = scores.shape[0] // B
    threshold_tiled = threshold_expanded.repeat(N)  # (B*N,)
    m_mask = torch.sigmoid((scores - threshold_tiled) / temperature)

    # 数值检查：确保不包含 NaN/Inf
    if not torch.isfinite(m_mask).all():
        m_mask = torch.where(
            torch.isfinite(m_mask), m_mask, torch.ones_like(m_mask) * 0.5
        )

    # 裁剪到有效范围
    m_mask = torch.clamp(m_mask, min=1e-6, max=1 - 1e-6)

    # 调试日志：每50次调用打印一次，仅限前500次
    global _mask_debug_counter
    if _mask_debug_counter % 50 == 0 and _mask_debug_counter < 500:
        print(
            f"DEBUG [Iter {_mask_debug_counter}]: Mask Mean = {m_mask.mean().item():.4f}, Threshold Mean = {threshold.mean().item():.4f}"
        )
    _mask_debug_counter += 1

    return m_mask


def collect_stats(
    scores: torch.Tensor, batch_size: int, mask: torch.Tensor = None
) -> tuple:
    # 核心修正：彻底阻断梯度，防止 sqrt(0) 爆炸和对抗性特征坍塌
    scores = scores.detach()

    B = batch_size
    device = scores.device
    N = int(scores.shape[0]) // int(B)
    scores_per_batch = scores.view(B, N)

    if mask is not None:
        mask = mask.to(device)
        mask_bool = mask.detach().bool()
        mask_float = mask_bool.float()

        count = mask_float.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (scores_per_batch * mask_float).sum(dim=1, keepdim=True) / count

        var = ((scores_per_batch * mask_float) - mean * mask_float).pow(2).sum(
            dim=1, keepdim=True
        ) / count
        var = var.clamp(min=1e-8)
        std = var.sqrt()

        scores_for_sort = scores_per_batch.clone()
        scores_for_sort[~mask_bool] = 1e9
        scores_sorted = scores_for_sort.sort(dim=1)[0]

        max_n = torch.tensor(N - 1, dtype=torch.long, device=device)
        mid_idx = (count // 2).clamp(max=max_n).long()
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        p50 = scores_sorted[batch_idx, mid_idx]

        scores_for_max = scores_per_batch.clone()
        scores_for_max[~mask_bool] = -1e9
        max_val = scores_for_max.max(dim=1, keepdim=True)[0]

        return mean, std, p50, max_val

    # Fallback: 所有窗口都有效，用原生方法（同样受益于 detach）
    mean = scores_per_batch.mean(dim=1, keepdim=True)
    std = scores_per_batch.std(dim=1, keepdim=True)
    p50 = scores_per_batch.quantile(0.5, dim=1, keepdim=True)
    max_val = scores_per_batch.max(dim=1, keepdim=True)[0]
    return mean, std, p50, max_val


def compute_erank(scores_2d: torch.Tensor) -> torch.Tensor:
    """从 window scores 计算图像级复杂度 C_b（梯度安全阻断）

    C_b = eRank / N ∈ [1/N, 1]，衡量窗口分数分布的均匀程度。
    值越大 → 分布越均匀 → 图像信息越分散 → 图像越"难"。

    ⚠️ 必须 detach：C_b 是纯环境变量，不允许梯度通过 scores 回传，
    否则网络会通过拉平所有窗口分数来"作弊"降低剪枝惩罚。

    Args:
        scores_2d: (B, N) window scores（KL 或 INC），已 reshape 为 2D

    Returns:
        C_b: (B, 1) 图像复杂度，已 detach
    """
    scores_2d = scores_2d.detach()  # 入口阻断，双保险
    B, N = scores_2d.shape
    probs = F.softmax(scores_2d, dim=-1)
    H = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (B,)
    e_rank = torch.exp(H)  # (B,)
    C_b = (e_rank / N).unsqueeze(-1)  # (B, 1)
    return torch.clamp(C_b, 0.0, 1.0)  # clamp 防极端值


# ==========================================================================
# tau debug: 记录 tau_raw / stage_policy / C_b / bias / τ 的数值关系
# ==========================================================================
_TAU_DEBUG_COUNTER = 0
_TAU_DEBUG_MAX = 3000


def _write_tau_debug(strategy, stage, block, tau_raw, tau, C_b, stage_policy_val):
    global _TAU_DEBUG_COUNTER
    import os

    bias = stage_policy_val * C_b  # (B, 1)，核心量：stage_policy 对 τ 的实际贡献

    tau_raw_abs = tau_raw.abs().mean().item()
    bias_abs = bias.abs().mean().item()
    ratio = bias_abs / (tau_raw_abs + 1e-6)

    path = os.path.join(_export_base_path if _export_base_path else '/tmp', 'tau_debug.csv')
    if _TAU_DEBUG_COUNTER == 0:
        with open(path, 'w') as f:
            f.write(
                "iter,strategy,stage,block,"
                "tau_raw_mean,tau_raw_std,tau_raw_abs,tau_raw_min,tau_raw_max,"
                "stage_policy,"
                "Cb_mean,Cb_std,"
                "bias_mean,bias_std,bias_abs,"
                "tau_mean,tau_std,tau_min,tau_max,"
                "ratio\n"
            )

    with open(path, 'a') as f:
        f.write(
            f"{_TAU_DEBUG_COUNTER},{strategy},{stage},{block},"
            f"{tau_raw.mean().item():.6f},{tau_raw.std().item():.6f},"
            f"{tau_raw_abs:.6f},{tau_raw.min().item():.6f},{tau_raw.max().item():.6f},"
            f"{stage_policy_val.item():.6f},"
            f"{C_b.mean().item():.6f},{C_b.std().item():.6f},"
            f"{bias.mean().item():.6f},{bias.std().item():.6f},{bias_abs:.6f},"
            f"{tau.mean().item():.6f},{tau.std().item():.6f},"
            f"{tau.min().item():.6f},{tau.max().item():.6f},"
            f"{ratio:.6f}\n"
        )

    _TAU_DEBUG_COUNTER += 1


def compute_gate_loss(
    m_mask: torch.Tensor,
    scores: torch.Tensor,
    lambda_reg: float,
    valid_mask: torch.Tensor = None,
) -> tuple:
    """Log-barrier gate loss for entropy pruning.

    L = -log(R_info) + lambda_reg * R_act

    梯度自适应：
      - R_info 高时（信息富裕）：dL/dR = -1/R_info 温和，允许剪枝探索
      - R_info 低时（信息贫穷）：dL/dR = -1/R_info 暴增，形成不可逾越的屏障

    Args:
        m_mask:     软掩码, shape (N,)
        scores:     归一化分数, shape (N,), 已 detach
        lambda_reg: 算力惩罚系数（唯一超参数）
        valid_mask: 有效窗口掩码

    Returns:
        loss:       标量 gate loss
        info_ratio: 信息保留率（用于监控，值越大越好）
        act_ratio:  激活率（用于监控，值越小=剪枝越多）
    """
    scores = scores.detach()

    if valid_mask is None:
        valid_mask = torch.ones_like(m_mask, dtype=torch.bool)

    N_valid = valid_mask.sum()
    if N_valid == 0:
        zero = torch.tensor(0.0, device=scores.device)
        return zero, zero, zero

    # 信息保留率: sum(m_i * S_i) / sum(S_i)
    reserved_sum = (m_mask * scores * valid_mask.float()).sum()
    total_sum = (scores * valid_mask.float()).sum()

    if not torch.isfinite(total_sum) or total_sum.abs() < 1e-8:
        total_sum = torch.tensor(1e-8, device=scores.device)

    info_ratio = reserved_sum / (total_sum + 1e-8)

    # 激活率: mean(m_i)
    act_ratio = (m_mask * valid_mask.float()).sum() / N_valid

    # === 对数屏障 Loss ===
    info_barrier = -torch.log(info_ratio.clamp(min=1e-8))
    sparsity_term = lambda_reg * act_ratio
    loss = info_barrier + sparsity_term

    if not torch.isfinite(loss):
        loss = torch.tensor(0.0, device=scores.device, requires_grad=True)

    return loss, info_ratio.detach(), act_ratio.detach()


class CrossStageInfo:
    """跨stage传递的对齐信息"""

    def __init__(self, aligned_windows, aligned_entropy, kl_scores, channel, hw_shape):
        self.aligned_windows = aligned_windows
        self.aligned_entropy = aligned_entropy
        self.kl_scores = kl_scores
        self.channel = channel
        self.hw_shape = hw_shape


def kl_weighted_pool_and_align(
    prev_windows: torch.Tensor,
    prev_kl_scores: torch.Tensor,
    prev_full_entropy: torch.Tensor,
    prev_channel: int,
    target_channel: int,
    target_hw_shape: tuple,
    target_window_size: int,
    prev_hw_shape: tuple,
    prev_window_size: int,
) -> tuple:
    """跨stage特征对齐：恢复2D空间结构后做标准下采样，再重新窗口化

    Args:
        prev_windows: (B*N_prev, window_size, window_size, C_prev) 前一stage的窗口特征（完整窗口）
        prev_kl_scores: (B*N_prev,) 前一stage的KL分数
        prev_full_entropy: (B*N_prev,) 前一stage完整窗口的entropy（KL剪枝前计算）
        prev_channel: int 前一stage的通道数
        target_channel: int 目标通道数
        target_hw_shape: tuple 目标分辨率 (H, W)
        target_window_size: int 目标窗口大小
        prev_hw_shape: tuple 前一stage的分辨率 (H, W)
        prev_window_size: int 前一stage的窗口大小

    Returns:
        aligned_windows: 对齐后的窗口特征 (B*N_target, window_size, window_size, target_channel)
        aligned_entropy: 对齐后的entropy (B*N_target,)
    """
    B = prev_kl_scores.shape[0] if prev_kl_scores.numel() > 0 else 1
    if B == 0 or prev_full_entropy.numel() == 0:
        B = prev_windows.shape[0] // (
            int(np.ceil(prev_hw_shape[0] / prev_window_size))
            * int(np.ceil(prev_hw_shape[1] / prev_window_size))
        )

    if B == 0:
        B = 1

    total_prev_windows = prev_windows.shape[0]
    N_prev = total_prev_windows // B

    H_prev = int(np.ceil(prev_hw_shape[0] / prev_window_size))
    W_prev = int(np.ceil(prev_hw_shape[1] / prev_window_size))
    C_prev = prev_windows.shape[3]

    H_target, W_target = target_hw_shape

    prev_windows_flat = prev_windows.view(B, N_prev, -1, C_prev)

    kl_weights = F.softmax(prev_kl_scores.view(B, N_prev), dim=-1)

    # Step 1: Restore 2D spatial structure
    # prev_windows: (B*N_prev, ws, ws, C) → (B, H_prev, W_prev, ws, ws, C) → (B, H_prev*ws, W_prev*ws, C)
    prev_windows_2d = prev_windows.view(
        B, H_prev, W_prev, prev_window_size, prev_window_size, C_prev
    )
    prev_feat_2d = prev_windows_2d.permute(0, 5, 1, 3, 2, 4).contiguous()
    prev_feat_2d = prev_feat_2d.view(
        B, C_prev, H_prev * prev_window_size, W_prev * prev_window_size
    )

    # Step 2: Apply 2D downsampling with KL-weighted pooling
    # For each 2x2 spatial neighborhood, compute KL-weighted average
    scale_factor = 2
    H_down = int(np.ceil(H_prev / scale_factor))
    W_down = int(np.ceil(W_prev / scale_factor))

    downsampled_windows_list = []
    downsampled_entropy_list = []

    for b in range(B):
        feat_b = prev_feat_2d[b]  # (C_prev, H_prev*ws, W_prev*ws)
        weights_b = kl_weights[b]  # (N_prev,)

        down_windows_b = []
        down_entropy_b = []

        for h in range(H_down):
            for w in range(W_down):
                # 2x2 spatial neighborhood in 2D feature map
                # Each window corresponds to (h*2, w*2), (h*2, w*2+1), (h*2+1, w*2), (h*2+1, w*2+1) in original
                h_base = h * scale_factor
                w_base = w * scale_factor

                # Collect 2x2=4 windows' KL weights and features
                indices = []
                for dh in range(scale_factor):
                    for dw in range(scale_factor):
                        win_h = h_base + dh
                        win_w = w_base + dw
                        if win_h < H_prev and win_w < W_prev:
                            idx = win_h * W_prev + win_w
                            indices.append(idx)

                if len(indices) == 0:
                    down_windows_b.append(
                        torch.zeros(
                            target_window_size,
                            target_window_size,
                            C_prev,
                            device=feat_b.device,
                            dtype=feat_b.dtype,
                        )
                    )
                    down_entropy_b.append(torch.zeros(1, device=feat_b.device))
                    continue

                # KL-weighted pooling of 2x2 windows
                weights_2x2 = weights_b[indices]  # (num_neighbors,)
                weights_2x2_norm = F.softmax(weights_2x2, dim=0)

                # Aggregate features from 2x2 windows
                feat_2x2 = []
                for idx in indices:
                    flat_idx = b * N_prev + idx
                    win_feat = prev_windows[flat_idx]  # (ws, ws, C_prev)
                    feat_2x2.append(win_feat)

                feat_2x2 = torch.stack(
                    feat_2x2, dim=0
                )  # (num_neighbors, ws, ws, C_prev)
                weights_2x2_norm = weights_2x2_norm.view(-1, 1, 1, 1)

                pooled_feat = (feat_2x2 * weights_2x2_norm).sum(
                    dim=0
                )  # (ws, ws, C_prev)
                down_windows_b.append(pooled_feat)

                # KL-weighted pooling of entropy
                entropy_2x2 = prev_full_entropy.view(B, N_prev)[
                    b, indices
                ]  # (num_neighbors,)
                pooled_ent = (entropy_2x2 * F.softmax(weights_2x2, dim=0)).sum()
                down_entropy_b.append(pooled_ent)

        down_windows_b = torch.stack(
            down_windows_b, dim=0
        )  # (H_down*W_down, ws, ws, C_prev)
        down_entropy_b = torch.stack(down_entropy_b, dim=0)  # (H_down*W_down,)

        downsampled_windows_list.append(down_windows_b)
        downsampled_entropy_list.append(down_entropy_b)

    aligned_windows = torch.cat(
        downsampled_windows_list, dim=0
    )  # (B*N_target, ws, ws, C_prev)
    aligned_entropy = torch.cat(downsampled_entropy_list, dim=0)  # (B*N_target,)

    # Step 3: Channel alignment if needed
    if prev_channel != target_channel:
        aligned_windows_flat = aligned_windows.view(
            B * H_down * W_down, target_window_size * target_window_size, prev_channel
        )
        aligned_windows_flat = aligned_windows_flat.permute(0, 2, 1)

        conv = nn.Conv1d(prev_channel, target_channel, kernel_size=1).to(
            aligned_windows.device
        )
        aligned_windows_flat = conv(aligned_windows_flat)
        aligned_windows_flat = aligned_windows_flat.permute(0, 2, 1)

        aligned_windows = aligned_windows_flat.view(
            B * H_down * W_down, target_window_size, target_window_size, target_channel
        )

    return aligned_windows, aligned_entropy


def compute_aligned_entropy_from_windows(
    x_windows: torch.Tensor, B: int, window_size: int = 7
) -> torch.Tensor:
    """从对齐后的窗口特征计算entropy

    Args:
        x_windows: (B*N, window_size, window_size, C)
        B: batch size
        window_size: window size

    Returns:
        entropy: (B*N,) entropy for each window
    """
    total_windows = x_windows.shape[0]
    N_win = total_windows // B
    x_flat = x_windows.view(B, N_win, window_size * window_size, -1)
    x_flat = x_flat.view(-1, window_size * window_size, x_windows.shape[-1])
    attn = F.softmax(x_flat, dim=-1)
    entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean(dim=1)
    return entropy.view(B, N_win).view(-1)


class SwinBlockV3(nn.Module):
    """Swin Block with optional KL pruning (Cross-Layer Design)"""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        window_size: int = 7,
        shift: bool = False,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: dict = dict(type="GELU"),
        norm_cfg: dict = dict(type="LN"),
        with_cp: bool = False,
        kl_ratio: float = None,
        inc_ratio: float = None,
        strategy: str = None,
        stage_idx: int = 0,
        block_idx: int = 0,
        kl_predictor: nn.Module = None,
        inc_predictor: nn.Module = None,
        use_learnable_gate: bool = False,
        temperature: float = 1.0,
        lambda_kl: float = 0.1,
        lambda_inc: float = 0.1,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.with_cp = with_cp
        self.kl_ratio = kl_ratio
        self.inc_ratio = inc_ratio
        self.strategy = strategy
        self.stage_idx = stage_idx
        self.block_idx = block_idx
        self.kl_predictor = kl_predictor
        self.inc_predictor = inc_predictor
        self.use_learnable_gate = use_learnable_gate
        self.temperature = temperature
        self.lambda_kl = lambda_kl
        self.lambda_inc = lambda_inc
        self.kl_gate_loss = None
        self.inc_gate_loss = None

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=self.shift_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type="DropPath", drop_prob=drop_path_rate),
            init_cfg=None,
        )

        from mmcv.cnn.bricks.transformer import FFN

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type="DropPath", drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None,
        )

    def forward(
        self,
        x: torch.Tensor,
        hw_shape: tuple,
        entropy_cache: torch.Tensor = None,
        prev_aligned_entropy: torch.Tensor = None,
        prev_kl_keep_idx: torch.Tensor = None,
        C_b: torch.Tensor = None,
    ) -> tuple:
        """Forward function

        Args:
            x: input features
            hw_shape: spatial shape (H, W)
            entropy_cache: entropy from previous W-MSA in same stage for INC comparison
            prev_aligned_entropy: entropy from previous stage for cross-stage INC comparison
            prev_kl_keep_idx: KL keep indices from previous stage
            C_b: image-level complexity from Stage 0 (reused across all stages)

        Returns:
            x: output features
            entropy_or_tuple: entropy value, or tuple with additional info for cross-stage
        """
        can_prune_kl = self.kl_ratio is not None and self.kl_ratio < 1.0
        can_prune_inc = self.inc_ratio is not None and self.inc_ratio < 1.0

        if self.shift_size > 0:
            return (
                self._forward_base(x, hw_shape),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        if not can_prune_kl and not can_prune_inc:
            return (
                self._forward_base(x, hw_shape),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        if self.strategy is not None:
            if self.strategy == "kl_inc" and can_prune_kl and can_prune_inc:
                return self._forward_kl_inc(
                    x, hw_shape, entropy_cache, prev_aligned_entropy, prev_kl_keep_idx, C_b=C_b
                )
            elif self.strategy in ["kl", "kl_inc"] and can_prune_kl:
                return self._forward_kl(x, hw_shape, C_b=C_b)
            elif self.strategy in ["inc", "kl_inc"] and can_prune_inc:
                return self._forward_inc(
                    x, hw_shape, entropy_cache, prev_aligned_entropy, prev_kl_keep_idx, C_b=C_b
                )

        return (
            self._forward_base(x, hw_shape),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def _forward_base(self, x: torch.Tensor, hw_shape: tuple) -> torch.Tensor:
        """Base forward without pruning"""
        B, L, C = x.shape
        H, W = hw_shape

        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)

        return x

    def _forward_kl(self, x: torch.Tensor, hw_shape: tuple, C_b: torch.Tensor = None) -> tuple:
        """KL pruning forward with dynamic threshold + soft mask injection"""
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, f"Input size mismatch: {L} vs {H}*{W}"

        x = x.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]

        shifted_x = (
            x
            if self.shift_size == 0
            else torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        )
        x_windows = self._window_partition(shifted_x)
        total_windows = x_windows.shape[0]
        N_win = total_windows // B

        if self.shift_size > 0 or self.kl_ratio is None:
            base_result = self._forward_base_with_pad(x, hw_shape, pad_r, pad_b)
            return base_result, None, None, None, None, None, None, None

        # ==========================================
        # 阶段一：KL 动态阈值剪枝
        # ==========================================
        window_scores = compute_window_relative_entropy(x_windows, B, self.window_size)
        window_scores_reshaped = window_scores.view(B, -1)
        _collect_kl_scores(window_scores_reshaped, self.stage_idx, self.block_idx)

        # ----- Step 1: MLP 预测 τ_raw (无 sigmoid) -----
        scores_flat = window_scores_reshaped.view(-1)
        stats_mean, stats_std, stats_p50, stats_max = collect_stats(scores_flat, B)
        tau_raw = self.kl_predictor(
            stats_mean, stats_std, stats_p50, stats_max, self.stage_idx, self.block_idx
        )  # (B, 1) raw logit

        # 图像级复杂度 C_b（Stage 0 计算，后续 stage 复用）
        if C_b is None:
            C_b = compute_erank(window_scores_reshaped)  # (B, 1)，首次计算
        self._C_b = C_b.detach()  # 存储供 stage sequence 传递给下游 stage

        # stage_policy 调制: τ = sigmoid(τ_raw + stage_policy)
        tau = torch.sigmoid(tau_raw + self.kl_predictor.stage_policy[self.stage_idx])  # (B, 1)

        # ----- Step 2: Min-Max 归一化 (用于比较) -----
        scores_min = window_scores_reshaped.min(dim=1, keepdim=True)[0]
        scores_max = window_scores_reshaped.max(dim=1, keepdim=True)[0]
        kl_norm_scores = (window_scores_reshaped - scores_min) / (
            scores_max - scores_min + 1e-8
        )

        # ----- Step 3: 软掩码计算 -----
        tau_tiled = tau.squeeze(-1).unsqueeze(
            -1
        )  # (B, 1) -> (B, N_win)
        m_mask_kl = torch.sigmoid(
            (kl_norm_scores - tau_tiled) / self.temperature
        )
        # m_mask_kl = torch.clamp(m_mask_kl, min=1e-6, max=1 - 1e-6)

        # ----- Step 4: 硬掩码 + Fallback -----
        kl_keep_mask = kl_norm_scores > tau  # (B, N_win)
        # fallback: 每张图至少保留分数最高的窗口
        # fallback = torch.zeros_like(kl_keep_mask)
        # fallback[
        #     torch.arange(B, device=x.device), window_scores_reshaped.argmax(dim=1)
        # ] = True
        # kl_keep_mask = kl_keep_mask | fallback
        kl_keep_idx_flat = torch.nonzero(kl_keep_mask.view(-1)).squeeze(-1)

        # ----- Step 5: 软掩码注入特征 -----
        x_kl = x_windows.view(-1, self.window_size * self.window_size, C)[
            kl_keep_idx_flat
        ]
        m_mask_kept = (
            m_mask_kl.view(-1)[kl_keep_idx_flat].unsqueeze(-1).unsqueeze(-1)
        )  # (N_kl, 1, 1)
        x_kl = x_kl * m_mask_kept  # 软掩码乘到特征上

        # ----- Step 6: Attention 计算 -----
        identity_attn = x_kl
        x_kl = self.norm1(x_kl)
        x_kl = self.attn.w_msa(x_kl)
        x_kl = identity_attn + x_kl

        # ----- Step 7: FFN 计算 -----
        identity_ffn = x_kl
        x_kl = self.norm2(x_kl)
        x_kl = self.ffn(x_kl)
        x_kl = identity_ffn + x_kl

        # ----- Step 8: 特征写回 -----
        cur_entropy_local = self._compute_entropy(x_kl)
        x_windows_new = x_windows.clone()
        x_kl_reshaped = x_kl.view(-1, self.window_size, self.window_size, C)
        x_windows_new[kl_keep_idx_flat] = x_kl_reshaped

        full_entropy = torch.zeros(total_windows, device=x.device)
        full_entropy[kl_keep_idx_flat] = cur_entropy_local.detach()

        # ----- Step 9: Patch 重构 -----
        attn_windows = x_windows_new.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, H_pad, W_pad)
        x = (
            shifted_x
            if self.shift_size == 0
            else torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        )
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # ----- Step 10: Gate Loss 计算 -----
        block_gate_loss = None
        if self.use_learnable_gate and self.kl_predictor is not None:
            kl_loss, kl_info, kl_reg = compute_gate_loss(
                m_mask_kl.view(-1), kl_norm_scores.view(-1), self.lambda_kl
            )
            self.kl_gate_loss = kl_loss
            for b in range(B):
                collect_threshold(
                    tau[b].item(), self.stage_idx, self.block_idx, "kl"
                )
            block_gate_loss = (kl_loss, None, kl_info, kl_reg, None, None)

        # 数值检查
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # return x, full_entropy, x_windows, x_windows_new, window_scores, (H_pad, W_pad), self.embed_dims, kl_keep_idx_flat, block_gate_loss
        return (
            x,
            full_entropy,
            x_windows,
            x_windows_new,
            window_scores_reshaped,
            (H_pad, W_pad),
            self.embed_dims,
            kl_keep_idx_flat,
            block_gate_loss,
        )

    def _forward_inc(
        self,
        x: torch.Tensor,
        hw_shape: tuple,
        entropy_cache: torch.Tensor = None,
        prev_aligned_entropy: torch.Tensor = None,
        prev_kl_keep_idx: torch.Tensor = None,
        C_b: torch.Tensor = None,
    ) -> tuple:
        """Incremental pruning forward with dynamic threshold + soft mask injection

        Args:
            entropy_cache: entropy from previous W-MSA in same stage
            prev_aligned_entropy: entropy from previous stage (aligned)
        """
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, f"Input size mismatch: {L} vs {H}*{W}"

        x = x.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]

        shifted_x = (
            x
            if self.shift_size == 0
            else torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        )
        x_windows = self._window_partition(shifted_x)
        total_windows = x_windows.shape[0]
        N_win = total_windows // B

        if self.shift_size > 0 or self.inc_ratio is None:
            base_result = self._forward_base_with_pad(x, hw_shape, pad_r, pad_b)
            return base_result, None, None, None, None, None, None, None

        # ==========================================
        # 阶段一：W-MSA 计算
        # ==========================================
        x_windows_flat = x_windows.view(-1, self.window_size * self.window_size, C)
        identity_attn = x_windows_flat
        x_after_attn = self.norm1(x_windows_flat)
        x_after_attn = self.attn.w_msa(x_after_attn)
        x_after_attn = identity_attn + x_after_attn

        cur_entropy = self._compute_entropy(x_after_attn)

        # ==========================================
        # 阶段二：INC 动态阈值剪枝
        # ==========================================
        inc_scores = torch.zeros_like(cur_entropy)

        if entropy_cache is not None and entropy_cache.shape[0] == total_windows:
            inc_scores = torch.abs(cur_entropy - entropy_cache.detach())
        elif (
            prev_aligned_entropy is not None
            and prev_aligned_entropy.shape[0] == total_windows
        ):
            cross_stage_scores = torch.abs(cur_entropy - prev_aligned_entropy.detach())
            inc_scores = cross_stage_scores

        inc_scores_reshaped = inc_scores.view(B, -1)
        _collect_inc_scores(inc_scores_reshaped, self.stage_idx, self.block_idx)

        # ----- Step 1: MLP 预测 τ_raw (无 sigmoid) -----
        scores_flat = inc_scores_reshaped.view(-1)
        stats_mean, stats_std, stats_p50, stats_max = collect_stats(scores_flat, B)
        tau_raw = self.inc_predictor(
            stats_mean, stats_std, stats_p50, stats_max, self.stage_idx, self.block_idx
        )  # (B, 1) raw logit

        # 图像级复杂度 C_b（Stage 0 计算，后续 stage 复用）
        if C_b is None:
            C_b = compute_erank(inc_scores_reshaped)  # (B, 1)，首次计算
        self._C_b = C_b.detach()  # 存储供 stage sequence 传递给下游 stage

        # stage_policy 调制: τ = sigmoid(τ_raw + stage_policy)
        tau = torch.sigmoid(tau_raw + self.inc_predictor.stage_policy[self.stage_idx])  # (B, 1)

        # ----- Step 2: Min-Max 归一化 -----
        scores_min = inc_scores_reshaped.min(dim=1, keepdim=True)[0]
        scores_max = inc_scores_reshaped.max(dim=1, keepdim=True)[0]
        inc_norm_scores = (inc_scores_reshaped - scores_min) / (
            scores_max - scores_min + 1e-8
        )

        # ----- Step 3: 软掩码计算 -----
        tau_tiled = tau.squeeze(-1).unsqueeze(
            -1
        )  # (B, 1) -> (B, N_win)
        m_mask_inc = torch.sigmoid(
            (inc_norm_scores - tau_tiled) / self.temperature
        )
        m_mask_inc = torch.clamp(m_mask_inc, min=1e-6, max=1 - 1e-6)

        # ----- Step 4: 硬掩码 + Fallback -----
        inc_keep_mask = inc_norm_scores > tau  # (B, N_win)
        fallback = torch.zeros_like(inc_keep_mask)
        fallback[
            torch.arange(B, device=x.device), inc_scores_reshaped.argmax(dim=1)
        ] = True
        inc_keep_mask = inc_keep_mask | fallback
        inc_keep_idx_flat = torch.nonzero(inc_keep_mask.view(-1)).squeeze(-1)

        # ----- Step 5: 软掩码注入特征 -----
        x_ffn_input = x_after_attn[inc_keep_idx_flat]
        m_mask_kept = m_mask_inc.view(-1)[inc_keep_idx_flat].unsqueeze(-1).unsqueeze(-1)
        x_ffn_input = x_ffn_input * m_mask_kept

        # ----- Step 6: FFN 计算 -----
        identity_ffn = x_ffn_input
        x_ffn_input = self.norm2(x_ffn_input)
        x_ffn_input = self.ffn(x_ffn_input)
        x_ffn_input = identity_ffn + x_ffn_input

        # ----- Step 7: 特征写回 -----
        x_windows_new = x_windows.clone()
        x_windows_new_reshaped = x_ffn_input.view(
            -1, self.window_size, self.window_size, C
        )
        x_windows_new[inc_keep_idx_flat] = x_windows_new_reshaped

        # ----- Step 8: Patch 重构 -----
        attn_windows = x_windows_new.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, H_pad, W_pad)
        x = (
            shifted_x
            if self.shift_size == 0
            else torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        )
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # ----- Step 9: Gate Loss 计算 -----
        block_gate_loss = None
        if self.use_learnable_gate and self.inc_predictor is not None:
            inc_loss, inc_info, inc_reg = compute_gate_loss(
                m_mask_inc.view(-1), inc_norm_scores.view(-1), self.lambda_inc
            )
            self.inc_gate_loss = inc_loss
            for b in range(B):
                collect_threshold(
                    tau[b].item(), self.stage_idx, self.block_idx, "inc"
                )
            block_gate_loss = (None, inc_loss, None, None, inc_info, inc_reg)

        # 数值检查
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            x,
            cur_entropy.detach(),
            x_windows,
            None,
            hw_shape,
            self.embed_dims,
            block_gate_loss,
        )

    def _forward_kl_inc(
        self,
        x: torch.Tensor,
        hw_shape: tuple,
        entropy_cache: torch.Tensor = None,
        prev_aligned_entropy: torch.Tensor = None,
        prev_kl_keep_idx: torch.Tensor = None,
        C_b: torch.Tensor = None,
    ) -> tuple:
        """KL + INC 串联动态阈值剪枝 with soft mask injection

        KL stage: dynamic threshold → soft mask → Attention
        INC stage: dynamic threshold on KL-surviving → soft mask → FFN
        """
        if self.inc_ratio is None:
            return self._forward_kl(x, hw_shape)

        B, L, C = x.shape
        H, W = hw_shape
        x = x.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]

        shifted_x = (
            x
            if self.shift_size == 0
            else torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        )
        x_windows = self._window_partition(shifted_x)
        total_windows = x_windows.shape[0]
        N_win = total_windows // B

        if self.shift_size > 0 or self.kl_ratio is None:
            base_result = self._forward_base_with_pad(x, hw_shape, pad_r, pad_b)
            return base_result, None, None, None, None, None, None, None

        # ==========================================
        # 阶段一：KL 动态阈值剪枝 (Attention Path)
        # ==========================================
        window_scores = compute_window_relative_entropy(x_windows, B, self.window_size)
        window_scores_reshaped = window_scores.view(B, -1)
        _collect_kl_scores(window_scores_reshaped, self.stage_idx, self.block_idx)

        # ----- KL: MLP 预测 τ_raw (无 sigmoid) -----
        scores_flat = window_scores_reshaped.view(-1)
        stats_mean, stats_std, stats_p50, stats_max = collect_stats(scores_flat, B)
        tau_raw = self.kl_predictor(
            stats_mean, stats_std, stats_p50, stats_max, self.stage_idx, self.block_idx
        )  # (B, 1) raw logit

        # 图像级复杂度 C_b（Stage 0 计算，后续 stage 复用）
        if C_b is None:
            C_b = compute_erank(window_scores_reshaped)  # (B, 1)，首次计算
        self._C_b = C_b.detach()  # 存储供 stage sequence 传递给下游 stage

        # stage_policy 调制: τ = sigmoid(τ_raw + stage_policy)
        tau = torch.sigmoid(tau_raw + self.kl_predictor.stage_policy[self.stage_idx])  # (B, 1)

        if _TAU_DEBUG_COUNTER < _TAU_DEBUG_MAX:
            _write_tau_debug('kl', self.stage_idx, self.block_idx,
                             tau_raw, tau, C_b,
                             self.kl_predictor.stage_policy[self.stage_idx])

        # ----- KL: Min-Max 归一化 -----
        scores_min = window_scores_reshaped.min(dim=1, keepdim=True)[0]
        scores_max = window_scores_reshaped.max(dim=1, keepdim=True)[0]
        kl_norm_scores = (window_scores_reshaped - scores_min) / (
            scores_max - scores_min + 1e-8
        )

        # ----- KL: 软掩码计算 -----
        tau_tiled = tau.squeeze(-1).unsqueeze(
            -1
        )  # (B, 1) -> (B, N_win)
        m_mask_kl = torch.sigmoid(
            (kl_norm_scores - tau_tiled) / self.temperature
        )
        m_mask_kl = torch.clamp(m_mask_kl, min=1e-6, max=1 - 1e-6)

        # ----- KL: 硬掩码 + Fallback -----
        kl_keep_mask = kl_norm_scores > tau  # (B, N_win)
        fallback = torch.zeros_like(kl_keep_mask)
        fallback[
            torch.arange(B, device=x.device), window_scores_reshaped.argmax(dim=1)
        ] = True
        kl_keep_mask = kl_keep_mask | fallback
        kl_keep_idx_flat = torch.nonzero(kl_keep_mask.view(-1)).squeeze(-1)

        # ----- KL: 软掩码注入 + Attention -----
        x_kl = x_windows.view(-1, self.window_size * self.window_size, C)[
            kl_keep_idx_flat
        ]
        m_mask_kept = m_mask_kl.view(-1)[kl_keep_idx_flat].unsqueeze(-1).unsqueeze(-1)
        x_kl = x_kl * m_mask_kept

        identity_attn = x_kl
        x_kl = self.norm1(x_kl)
        x_kl = self.attn.w_msa(x_kl)
        x_kl = identity_attn + x_kl

        # 计算 KL 存活窗口的 entropy (用于后续 INC 计算)
        cur_entropy = self._compute_entropy(x_kl)

        # 备份全部窗口的 entropy (用于跨层/跨阶段传递)
        x_windows_flat = x_windows.view(-1, self.window_size * self.window_size, C)
        full_entropy_before_kl = self._compute_entropy(x_windows_flat)

        # ----- KL: 写回到全局 x_windows_new -----
        x_windows_new = x_windows.clone()
        x_kl_reshaped = x_kl.view(-1, self.window_size, self.window_size, C)
        x_windows_new[kl_keep_idx_flat] = x_kl_reshaped

        # ==========================================
        # 阶段二：INC 动态阈值剪枝 (FFN Path)
        # ==========================================
        # ----- INC: 计算存活窗口的 INC 分数 -----
        inc_scores_survived = torch.zeros_like(cur_entropy)

        if entropy_cache is not None and entropy_cache.shape[0] == total_windows:
            inc_scores_survived = torch.abs(
                cur_entropy - entropy_cache[kl_keep_idx_flat].detach()
            )
        elif (
            prev_aligned_entropy is not None
            and prev_aligned_entropy.shape[0] == total_windows
        ):
            cross_stage_scores = torch.abs(
                cur_entropy - prev_aligned_entropy[kl_keep_idx_flat].detach()
            )
            inc_scores_survived = cross_stage_scores

        # ----- INC: Scatter 到全局张量 (B, N_win) -----
        # KL 剪掉的窗口位置为 0，保持 (B, N_win) 形状，可被 B 整除
        inc_scores_full = torch.zeros(B, N_win, device=x.device)
        inc_scores_full.view(-1)[kl_keep_idx_flat] = inc_scores_survived
        _collect_inc_scores(inc_scores_full, self.stage_idx, self.block_idx)

        # ----- INC: MLP 预测 τ_raw (无 sigmoid) -----
        # 必须用 inc_scores_full.view(-1) 而不是 inc_scores_survived.view(-1)
        # 因为后者长度 = N_kl (动态)，无法 view(B, N_win)
        raw_inc_scores_flat = inc_scores_full.view(-1)
        stats_mean, stats_std, stats_p50, stats_max = collect_stats(
            raw_inc_scores_flat, B, mask=kl_keep_mask
        )
        tau_raw = self.inc_predictor(
            stats_mean, stats_std, stats_p50, stats_max, self.stage_idx, self.block_idx
        )  # (B, 1) raw logit

        # 图像级复杂度 C_b（复用 KL 路径计算结果，不重复计算）

        # stage_policy 调制: τ = sigmoid(τ_raw + stage_policy)
        tau = torch.sigmoid(tau_raw + self.inc_predictor.stage_policy[self.stage_idx])  # (B, 1)

        if _TAU_DEBUG_COUNTER < _TAU_DEBUG_MAX:
            _write_tau_debug('inc', self.stage_idx, self.block_idx,
                             tau_raw, tau, C_b,
                             self.inc_predictor.stage_policy[self.stage_idx])

        # ----- INC: Min-Max 归一化 (纯 GPU 加速版，无 CPU 同步) -----
        # 用极值填充被排除的窗口，安全地按 Batch 计算极值
        inc_for_max = torch.where(
            kl_keep_mask, inc_scores_full, torch.tensor(-1e9, device=x.device)
        )
        inc_for_min = torch.where(
            kl_keep_mask, inc_scores_full, torch.tensor(1e9, device=x.device)
        )

        inc_max = inc_for_max.amax(dim=1, keepdim=True)
        inc_min = inc_for_min.amin(dim=1, keepdim=True)

        # Min-Max normalization
        inc_range = (inc_max - inc_min).clamp(min=1e-6)
        inc_norm_scores = (inc_scores_full - inc_min) / inc_range

        # 核心修复：被 KL 剪掉的窗口，归一化分数强制设为 0.0 (不影响 Loss 的 sum)
        inc_norm_scores = torch.where(
            kl_keep_mask, inc_norm_scores, torch.tensor(0.0, device=x.device)
        )

        # ----- INC: 软掩码计算 -----
        tau_tiled = tau.squeeze(-1).unsqueeze(
            -1
        )  # (B, 1) -> (B, N_win)
        m_mask_inc = torch.sigmoid(
            (inc_norm_scores - tau_tiled) / self.temperature
        )
        m_mask_inc = torch.clamp(m_mask_inc, min=1e-6, max=1 - 1e-6)

        # 核心修复：被 KL 剪掉的窗口，软掩码也强制设为 0.0
        m_mask_inc = torch.where(
            kl_keep_mask, m_mask_inc, torch.tensor(0.0, device=x.device)
        )

        # ----- INC: 硬掩码 + Fallback (INC 必须是 KL 的子集) -----
        inc_keep_mask = (inc_norm_scores > tau) & kl_keep_mask
        inc_fallback = torch.zeros_like(inc_keep_mask)
        inc_fallback[
            torch.arange(B, device=x.device), inc_scores_full.argmax(dim=1)
        ] = True
        inc_keep_mask = inc_keep_mask | (inc_fallback & kl_keep_mask)

        # ----- INC: 软掩码注入 + FFN -----
        final_keep_idx_flat = torch.nonzero(inc_keep_mask.view(-1)).squeeze(-1)

        x_ffn = x_windows_new.view(-1, self.window_size * self.window_size, C)[
            final_keep_idx_flat
        ]
        m_mask_inc_kept = (
            m_mask_inc.view(-1)[final_keep_idx_flat].unsqueeze(-1).unsqueeze(-1)
        )
        x_ffn = x_ffn * m_mask_inc_kept

        identity_ffn = x_ffn
        x_ffn = self.norm2(x_ffn)
        x_ffn = self.ffn(x_ffn)
        x_ffn = identity_ffn + x_ffn

        # ----- INC: 写回到全局 -----
        x_ffn_reshaped = x_ffn.view(-1, self.window_size, self.window_size, C)
        x_windows_new.view(-1, self.window_size, self.window_size, C)[
            final_keep_idx_flat
        ] = x_ffn_reshaped

        # ==========================================
        # 阶段三：Patch 重构
        # ==========================================
        attn_windows = x_windows_new.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, H_pad, W_pad)
        x = (
            shifted_x
            if self.shift_size == 0
            else torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        )
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        full_entropy = full_entropy_before_kl

        # ==========================================
        # 阶段四：Gate Loss 计算
        # ==========================================
        block_gate_loss = None
        if self.use_learnable_gate:
            kl_loss = None
            inc_loss = None
            kl_info = None
            kl_reg = None
            inc_info = None
            inc_reg = None

            if self.kl_predictor is not None:
                kl_loss, kl_info, kl_reg = compute_gate_loss(
                    m_mask_kl.view(-1), kl_norm_scores.view(-1), self.lambda_kl
                )
                self.kl_gate_loss = kl_loss
                for b in range(B):
                    collect_threshold(
                        tau[b].item(), self.stage_idx, self.block_idx, "kl"
                    )

            if self.inc_predictor is not None and inc_scores_full.numel() > 0:
                inc_loss, inc_info, inc_reg = compute_gate_loss(
                    m_mask_inc.view(-1),
                    inc_norm_scores.view(-1),
                    self.lambda_inc,
                    valid_mask=kl_keep_mask.view(-1),
                )
                self.inc_gate_loss = inc_loss
                for b in range(B):
                    collect_threshold(
                        tau[b].item(), self.stage_idx, self.block_idx, "inc"
                    )

            block_gate_loss = (kl_loss, inc_loss, kl_info, kl_reg, inc_info, inc_reg)

        # 数值检查
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # return x, full_entropy, x_windows, x_windows_new, window_scores, (H_pad, W_pad), self.embed_dims, kl_keep_idx_flat, block_gate_loss
        return (
            x,
            full_entropy,
            x_windows,
            x_windows_new,
            window_scores_reshaped,
            (H_pad, W_pad),
            self.embed_dims,
            kl_keep_idx_flat,
            block_gate_loss,
        )

    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    def _window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(
            B, H // window_size, W // window_size, window_size, window_size, -1
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def _forward_base_with_pad(
        self, x: torch.Tensor, hw_shape: tuple, pad_r: int, pad_b: int
    ) -> torch.Tensor:
        """Base forward with padding handling"""
        B, H, W, C = x.shape
        shifted_x = (
            x
            if self.shift_size == 0
            else torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        )
        x_windows = self._window_partition(shifted_x)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        identity = x_windows
        x_windows = self.norm1(x_windows)
        x_windows = self.attn.w_msa(x_windows)
        x_windows = identity + x_windows
        identity = x_windows
        x_windows = self.norm2(x_windows)
        x_windows = self.ffn(x_windows, identity=identity)
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(x_windows, x.shape[1], x.shape[2])
        x = (
            shifted_x
            if self.shift_size == 0
            else torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        )
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        return x

    def _compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute entropy from attention features"""
        B_N, area, C = x.shape
        attn = F.softmax(x, dim=-1)
        entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean(dim=1)
        return entropy


class SwinBlockSequenceV3(nn.Module):
    """Swin Block Sequence with KL config

    Args:
        embed_dims: feature dimension
        num_heads: number of attention heads
        feedforward_channels: FFN hidden dimension
        depth: number of blocks
        window_size: window size
        qkv_bias: whether to use bias in qkv
        qk_scale: qk scale
        drop_rate: dropout rate
        attn_drop_rate: attention dropout rate
        drop_path_rate: stochastic depth rate
        downsample: downsample module
        act_cfg: activation config
        norm_cfg: normalization config
        with_cp: use checkpoint
        block_kl_ratios: list of KL ratios for each block
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        depth: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        downsample: nn.Module = None,
        act_cfg: dict = dict(type="GELU"),
        norm_cfg: dict = dict(type="LN"),
        with_cp: bool = False,
        block_kl_ratios: list = None,
        block_inc_ratios: list = None,
        strategy: str = None,
        stage_idx: int = 0,
        kl_predictor: nn.Module = None,
        inc_predictor: nn.Module = None,
        use_learnable_gate: bool = False,
        temperature: float = 1.0,
        lambda_kl: float = 0.1,
        lambda_inc: float = 0.1,
    ):
        super().__init__()

        if block_kl_ratios is None:
            block_kl_ratios = [None] * depth
        elif isinstance(block_kl_ratios, (int, float)):
            block_kl_ratios = [block_kl_ratios] * depth

        if block_inc_ratios is None:
            block_inc_ratios = [None] * depth
        elif isinstance(block_inc_ratios, (int, float)):
            block_inc_ratios = [block_inc_ratios] * depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            kl_ratio = block_kl_ratios[i] if i < len(block_kl_ratios) else None
            inc_ratio = block_inc_ratios[i] if i < len(block_inc_ratios) else None

            block = SwinBlockV3(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate[i]
                if isinstance(drop_path_rate, list)
                else drop_path_rate,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                kl_ratio=kl_ratio,
                inc_ratio=inc_ratio,
                strategy=strategy,
                stage_idx=stage_idx,
                block_idx=i,
                kl_predictor=kl_predictor,
                inc_predictor=inc_predictor,
                use_learnable_gate=use_learnable_gate,
                temperature=temperature,
                lambda_kl=lambda_kl,
                lambda_inc=lambda_inc,
            )
            self.blocks.append(block)

        self.downsample = downsample

    def forward(
        self,
        x: torch.Tensor,
        hw_shape: tuple,
        prev_cross_stage_info: dict = None,
        next_stage_info: dict = None,
    ):
        """Forward function with cross-stage INC comparison support

        Args:
            x: (B, L, C) input features
            hw_shape: (H, W) spatial shape
            prev_cross_stage_info: dict with keys:
                - 'prev_aligned_entropy': entropy from previous stage
                - 'prev_windows': window features from previous stage
                - 'prev_kl_scores': KL scores from previous stage
                - 'prev_channel': channel from previous stage
            next_stage_info: dict for next stage (used for cross-stage alignment in this stage):
                - 'channel': next stage channel
                - 'hw_shape': next stage input spatial shape
                - 'window_size': next stage window size

        Returns:
            x_down: output features after downsample
            down_hw_shape: new spatial shape
            x: output features before downsample
            hw_shape: original spatial shape
            cross_stage_info: dict for next stage (aligned entropy, windows, kl_scores, channel, hw_shape)
        """
        entropy_cache = None
        prev_entropy = None

        prev_aligned_entropy = None
        prev_kl_keep_idx = None
        C_b = None
        if prev_cross_stage_info is not None:
            prev_aligned_entropy = prev_cross_stage_info.get(
                "prev_aligned_entropy", None
            )
            prev_kl_keep_idx = prev_cross_stage_info.get("prev_kl_keep_idx", None)
            C_b = prev_cross_stage_info.get("C_b", None)  # 从上游 stage 获取全局 C_b

        last_wmsa_info = None

        for i, block in enumerate(self.blocks):
            result = block(
                x, hw_shape, entropy_cache, prev_aligned_entropy, prev_kl_keep_idx, C_b=C_b
            )

            # 第一个 W-MSA block 会计算 C_b 并存入 block._C_b
            if C_b is None and hasattr(block, '_C_b') and block._C_b is not None:
                C_b = block._C_b

            if isinstance(result, tuple) and len(result) >= 8:
                x = result[0]
                new_entropy = result[1]
                windows = result[2]
                kl_scores = result[4]  # window_scores 在索引4位置
                block_hw_shape = result[5]
                channel = result[6]
                kl_keep_idx = result[7] if len(result) > 7 else None

                block_gate_loss = result[8] if len(result) > 8 else None
                if block_gate_loss is not None:
                    if not hasattr(self, "gate_losses"):
                        self.gate_losses = []
                    self.gate_losses.append(block_gate_loss)

                if block.shift_size == 0:
                    prev_entropy = new_entropy
                    entropy_cache = prev_entropy
                    prev_kl_keep_idx = kl_keep_idx

                    if windows is not None and kl_scores is not None:
                        last_wmsa_info = {
                            "windows": windows,
                            "kl_scores": kl_scores,
                            "kl_keep_idx": kl_keep_idx,
                            "hw_shape": block_hw_shape,
                            "channel": channel,
                            "entropy": new_entropy,
                            "window_size": block.window_size,
                        }
            else:
                x = result[0] if isinstance(result, tuple) else result
                prev_aligned_entropy = None
                prev_kl_keep_idx = None

        cross_stage_info = None
        if last_wmsa_info is not None:
            base_cross_stage_info = {
                "prev_aligned_entropy": last_wmsa_info["entropy"],
                "prev_windows": last_wmsa_info["windows"],
                "prev_kl_scores": last_wmsa_info["kl_scores"],
                "prev_kl_keep_idx": last_wmsa_info["kl_keep_idx"],
                "prev_channel": last_wmsa_info["channel"],
                "prev_hw_shape": last_wmsa_info["hw_shape"],
                "window_size": self.blocks[0].window_size,
                "C_b": C_b,  # Stage 0 计算的全局图像复杂度，全 stage 复用
            }

            if next_stage_info is not None:
                aligned_windows, aligned_entropy = kl_weighted_pool_and_align(
                    prev_windows=last_wmsa_info["windows"],
                    prev_kl_scores=last_wmsa_info["kl_scores"],
                    prev_full_entropy=last_wmsa_info["entropy"],
                    prev_channel=last_wmsa_info["channel"],
                    target_channel=next_stage_info["channel"],
                    target_hw_shape=next_stage_info["hw_shape"],
                    target_window_size=next_stage_info["window_size"],
                    prev_hw_shape=last_wmsa_info["hw_shape"],
                    prev_window_size=last_wmsa_info.get("window_size", 7),
                )
                cross_stage_info = {
                    "prev_aligned_entropy": aligned_entropy,
                    "prev_windows": aligned_windows,
                    "prev_kl_scores": last_wmsa_info["kl_scores"],
                    "prev_kl_keep_idx": last_wmsa_info["kl_keep_idx"],
                    "prev_channel": next_stage_info["channel"],
                    "prev_hw_shape": next_stage_info["hw_shape"],
                    "window_size": next_stage_info["window_size"],
                    "C_b": C_b,  # Stage 0 计算的全局图像复杂度，全 stage 复用
                }
            else:
                cross_stage_info = base_cross_stage_info

        if self.downsample is not None:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape, cross_stage_info
        else:
            return x, hw_shape, x, hw_shape, cross_stage_info


@MODELS.register_module()
class SwinTransformerV3(SwinTransformer):
    """Swin Transformer V3 with optional KL pruning

    Simplified implementation with two strategies:
    - 'base': No pruning, same as original Swin
    - 'kl': KL-based window pruning

    Config:
        backbone=dict(
            type='SwinTransformerV3',
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,

            # Strategy configuration
            strategy='kl',    # 'base' or 'kl'
            stage_config={
                0: {'blocks': [0], 'ratio': 0.8},
                1: {'blocks': [0], 'ratio': 0.8},
                2: {'blocks': [0, 2, 4], 'ratio': [0.8, 0.6, 0.4]},
                3: {'blocks': [0], 'ratio': 0.8},
            }
        )

    Args:
        pretrain_img_size: pretrain image size
        in_channels: input channels
        embed_dims: embedding dimension
        patch_size: patch size
        window_size: window size
        mlp_ratio: MLP ratio
        depths: depth of each stage
        num_heads: number of heads
        strides: stride for each stage
        out_indices: output indices
        qkv_bias: whether to use bias in qkv
        qk_scale: qk scale
        patch_norm: use patch norm
        drop_rate: dropout rate
        attn_drop_rate: attention dropout rate
        drop_path_rate: stochastic depth rate
        use_abs_pos_embed: use absolute position embedding
        act_cfg: activation config
        norm_cfg: normalization config
        with_cp: use checkpoint
        pretrained: pretrained checkpoint
        convert_weights: convert weights
        frozen_stages: frozen stages
        init_cfg: init config
    strategy: 'base', 'kl' or 'inc'
    stage_config: stage configuration for KL
    inc_stage_config: stage configuration for incremental
    """

    def __init__(
        self,
        pretrain_img_size: int = 224,
        in_channels: int = 3,
        embed_dims: int = 96,
        patch_size: int = 4,
        window_size: int = 7,
        mlp_ratio: int = 4,
        depths: tuple = (2, 2, 6, 2),
        num_heads: tuple = (3, 6, 12, 24),
        strides: tuple = (4, 2, 2, 2),
        out_indices: tuple = (0, 1, 2, 3),
        qkv_bias: bool = True,
        qk_scale: float = None,
        patch_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_abs_pos_embed: bool = False,
        act_cfg: dict = dict(type="GELU"),
        norm_cfg: dict = dict(type="LN"),
        with_cp: bool = False,
        pretrained: str = None,
        convert_weights: bool = False,
        frozen_stages: int = -1,
        init_cfg: dict = None,
        strategy: str = "base",
        stage_config: dict = None,
        inc_stage_config: dict = None,
        use_learnable_gate: bool = False,
        temperature: float = 1.0,
        lambda_kl: float = 0.1,
        lambda_inc: float = 0.1,
    ):
        self.strategy = strategy
        self.stage_config = stage_config or {}
        self.inc_stage_config = inc_stage_config or {}
        self.use_learnable_gate = use_learnable_gate
        self.temperature = temperature
        self.lambda_kl = lambda_kl
        self.lambda_inc = lambda_inc

        super().__init__(
            pretrain_img_size=pretrain_img_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            patch_size=patch_size,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            depths=depths,
            num_heads=num_heads,
            strides=strides,
            out_indices=out_indices,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            patch_norm=patch_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_abs_pos_embed=use_abs_pos_embed,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            with_cp=with_cp,
            pretrained=pretrained,
            convert_weights=convert_weights,
            frozen_stages=frozen_stages,
            init_cfg=init_cfg,
        )

        if use_learnable_gate:
            self.kl_predictor = ThresholdPredictor()
            self.inc_predictor = ThresholdPredictor()
            # 将 stage_policy 直接挂载到 predictor 上，零侵入传参
            num_stages = len(depths)
            self.kl_predictor.stage_policy = nn.Parameter(torch.zeros(num_stages))
            self.inc_predictor.stage_policy = nn.Parameter(torch.zeros(num_stages))
            self.register_buffer("kl_predictor_gate_loss", torch.zeros(1))
            self.register_buffer("inc_predictor_gate_loss", torch.zeros(1))
            self.register_buffer("kl_predictor_gate_info", torch.zeros(1))
            self.register_buffer("kl_predictor_gate_reg", torch.zeros(1))
            self.register_buffer("inc_predictor_gate_info", torch.zeros(1))
            self.register_buffer("inc_predictor_gate_reg", torch.zeros(1))
        else:
            self.kl_predictor = None
            self.inc_predictor = None

        if strategy == "kl":
            self._replace_blocks_with_kl()
        elif strategy == "inc":
            self._replace_blocks_with_inc()
        elif strategy == "kl_inc":
            self._replace_blocks_with_kl_inc()

    def _replace_blocks_with_kl(self):
        """Replace blocks with KL-enabled blocks"""
        from mmdet.models.layers import PatchMerging

        default_act_cfg = dict(type="GELU")
        default_norm_cfg = dict(type="LN")

        num_layers = len(self.stages)

        for stage_idx in range(num_layers):
            stage = self.stages[stage_idx]

            # Get stage config
            stage_cfg = self.stage_config.get(stage_idx, {})
            stage_blocks = stage_cfg.get("blocks", [])
            stage_ratio = stage_cfg.get("ratio", [])

            # Convert single value to list
            if isinstance(stage_ratio, (int, float)):
                stage_ratio = [stage_ratio]

            if not stage_blocks:
                continue

            depth = len(stage.blocks)

            # Get first block to extract dims
            first_block = stage.blocks[0]
            embed_dims = first_block.attn.w_msa.embed_dims
            num_heads = first_block.attn.w_msa.num_heads
            feedforward_channels = first_block.ffn.layers[1].out_features
            window_size = first_block.attn.w_msa.window_size[0]
            qkv_bias = first_block.attn.w_msa.qkv.bias is not None
            qk_scale = first_block.attn.w_msa.scale
            with_cp = first_block.with_cp
            drop_rate = 0.0
            attn_drop_rate = 0.0
            for child in first_block.attn.w_msa.children():
                if hasattr(child, "drop_prob"):
                    attn_drop_rate = child.drop_prob
                    break
            drop_path_rates = []
            for b in stage.blocks:
                dp = 0.0
                for child in b.attn.children():
                    if hasattr(child, "drop_prob"):
                        dp = child.drop_prob
                        break
                drop_path_rates.append(dp)

            # Collect KL ratios for each block
            block_kl_ratios = []
            for block_idx in range(depth):
                if block_idx in stage_blocks:
                    # Find ratio for this block
                    idx_in_blocks = stage_blocks.index(block_idx)
                    kl_ratio = (
                        stage_ratio[idx_in_blocks]
                        if idx_in_blocks < len(stage_ratio)
                        else stage_ratio[-1]
                    )
                    block_kl_ratios.append(kl_ratio)
                else:
                    block_kl_ratios.append(None)

            # Create new block sequence
            new_stage = SwinBlockSequenceV3(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                depth=depth,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates,
                downsample=stage.downsample,
                act_cfg=default_act_cfg,
                norm_cfg=default_norm_cfg,
                with_cp=with_cp,
                block_kl_ratios=block_kl_ratios,
                strategy="kl",
                stage_idx=stage_idx,
                kl_predictor=self.kl_predictor,
                inc_predictor=self.inc_predictor,
                use_learnable_gate=self.use_learnable_gate,
                temperature=self.temperature,
                lambda_kl=self.lambda_kl,
                lambda_inc=self.lambda_inc,
            )

            # Copy blocks' parameters
            for old_block, new_block in zip(stage.blocks, new_stage.blocks):
                new_block.norm1 = old_block.norm1
                new_block.norm2 = old_block.norm2
                new_block.attn = old_block.attn
                new_block.ffn = old_block.ffn

            new_stage.downsample = stage.downsample

            self.stages[stage_idx] = new_stage

    def _replace_blocks_with_inc(self):
        """Replace blocks with incremental-enabled blocks"""
        default_act_cfg = dict(type="GELU")
        default_norm_cfg = dict(type="LN")

        num_layers = len(self.stages)

        for stage_idx in range(num_layers):
            stage = self.stages[stage_idx]

            stage_cfg = self.inc_stage_config.get(stage_idx, {})
            stage_blocks = stage_cfg.get("blocks", [])
            stage_inc_ratio = stage_cfg.get("inc_ratio", [])

            if isinstance(stage_inc_ratio, (int, float)):
                stage_inc_ratio = [stage_inc_ratio]

            if not stage_blocks:
                continue

            depth = len(stage.blocks)

            first_block = stage.blocks[0]
            embed_dims = first_block.attn.w_msa.embed_dims
            num_heads = first_block.attn.w_msa.num_heads
            feedforward_channels = first_block.ffn.layers[1].out_features
            window_size = first_block.attn.w_msa.window_size[0]
            qkv_bias = first_block.attn.w_msa.qkv.bias is not None
            qk_scale = first_block.attn.w_msa.scale
            with_cp = first_block.with_cp
            drop_rate = 0.0
            attn_drop_rate = 0.0
            for child in first_block.attn.w_msa.children():
                if hasattr(child, "drop_prob"):
                    attn_drop_rate = child.drop_prob
                    break
            drop_path_rates = []
            for b in stage.blocks:
                dp = 0.0
                for child in b.attn.children():
                    if hasattr(child, "drop_prob"):
                        dp = child.drop_prob
                        break
                drop_path_rates.append(dp)

            block_inc_ratios = []
            for block_idx in range(depth):
                if block_idx in stage_blocks:
                    idx_in_blocks = stage_blocks.index(block_idx)
                    inc_ratio = (
                        stage_inc_ratio[idx_in_blocks]
                        if idx_in_blocks < len(stage_inc_ratio)
                        else stage_inc_ratio[-1]
                    )
                    block_inc_ratios.append(inc_ratio)
                else:
                    block_inc_ratios.append(None)

            new_stage = SwinBlockSequenceV3(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                depth=depth,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates,
                downsample=stage.downsample,
                act_cfg=default_act_cfg,
                norm_cfg=default_norm_cfg,
                with_cp=with_cp,
                block_inc_ratios=block_inc_ratios,
                strategy="inc",
                stage_idx=stage_idx,
                kl_predictor=self.kl_predictor,
                inc_predictor=self.inc_predictor,
                use_learnable_gate=self.use_learnable_gate,
                temperature=self.temperature,
                lambda_kl=self.lambda_kl,
                lambda_inc=self.lambda_inc,
            )

            for old_block, new_block in zip(stage.blocks, new_stage.blocks):
                new_block.norm1 = old_block.norm1
                new_block.norm2 = old_block.norm2
                new_block.attn = old_block.attn
                new_block.ffn = old_block.ffn

            new_stage.downsample = stage.downsample

            self.stages[stage_idx] = new_stage

    def _replace_blocks_with_kl_inc(self):
        """Replace blocks with both KL and INC enabled"""
        default_act_cfg = dict(type="GELU")
        default_norm_cfg = dict(type="LN")

        num_layers = len(self.stages)

        for stage_idx in range(num_layers):
            stage = self.stages[stage_idx]

            kl_stage_cfg = self.stage_config.get(stage_idx, {})
            kl_blocks = kl_stage_cfg.get("blocks", [])
            kl_ratios = kl_stage_cfg.get("ratio", [])

            inc_stage_cfg = self.inc_stage_config.get(stage_idx, {})
            inc_blocks = inc_stage_cfg.get("blocks", [])
            inc_ratios = inc_stage_cfg.get("inc_ratio", [])

            if isinstance(kl_ratios, (int, float)):
                kl_ratios = [kl_ratios]
            if isinstance(inc_ratios, (int, float)):
                inc_ratios = [inc_ratios]

            depth = len(stage.blocks)

            first_block = stage.blocks[0]
            embed_dims = first_block.attn.w_msa.embed_dims
            num_heads = first_block.attn.w_msa.num_heads
            feedforward_channels = first_block.ffn.layers[1].out_features
            window_size = first_block.attn.w_msa.window_size[0]
            qkv_bias = first_block.attn.w_msa.qkv.bias is not None
            qk_scale = first_block.attn.w_msa.scale
            with_cp = first_block.with_cp
            drop_rate = 0.0
            attn_drop_rate = 0.0
            for child in first_block.attn.w_msa.children():
                if hasattr(child, "drop_prob"):
                    attn_drop_rate = child.drop_prob
                    break
            drop_path_rates = []
            for b in stage.blocks:
                dp = 0.0
                for child in b.attn.children():
                    if hasattr(child, "drop_prob"):
                        dp = child.drop_prob
                        break
                drop_path_rates.append(dp)

            block_kl_ratios = []
            block_inc_ratios = []
            for block_idx in range(depth):
                kl_ratio = None
                inc_ratio = None
                if block_idx in kl_blocks:
                    idx_in_kl = kl_blocks.index(block_idx)
                    kl_ratio = (
                        kl_ratios[idx_in_kl]
                        if idx_in_kl < len(kl_ratios)
                        else kl_ratios[-1]
                    )
                if block_idx in inc_blocks:
                    idx_in_inc = inc_blocks.index(block_idx)
                    inc_ratio = (
                        inc_ratios[idx_in_inc]
                        if idx_in_inc < len(inc_ratios)
                        else inc_ratios[-1]
                    )
                block_kl_ratios.append(kl_ratio)
                block_inc_ratios.append(inc_ratio)

            new_stage = SwinBlockSequenceV3(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                depth=depth,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates,
                downsample=stage.downsample,
                act_cfg=default_act_cfg,
                norm_cfg=default_norm_cfg,
                with_cp=with_cp,
                block_kl_ratios=block_kl_ratios,
                block_inc_ratios=block_inc_ratios,
                strategy="kl_inc",
                stage_idx=stage_idx,
                kl_predictor=self.kl_predictor,
                inc_predictor=self.inc_predictor,
                use_learnable_gate=self.use_learnable_gate,
                temperature=self.temperature,
                lambda_kl=self.lambda_kl,
                lambda_inc=self.lambda_inc,
            )

            for old_block, new_block in zip(stage.blocks, new_stage.blocks):
                new_block.norm1 = old_block.norm1
                new_block.norm2 = old_block.norm2
                new_block.attn = old_block.attn
                new_block.ffn = old_block.ffn

            new_stage.downsample = stage.downsample

            self.stages[stage_idx] = new_stage

    def get_strategy_config(self) -> dict:
        """Get current strategy configuration"""
        return {
            "strategy": self.strategy,
            "stage_config": self.stage_config,
            "inc_stage_config": self.inc_stage_config,
        }

    def forward(self, x):
        """Forward with cross-stage INC comparison support

        Args:
            x: input images

        Returns:
            outs: list of feature maps
        """
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        cross_stage_info = None

        outs = []

        if self.use_learnable_gate:
            total_kl_gate_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            total_inc_gate_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            total_kl_gate_info = torch.tensor(0.0, device=x.device)
            total_kl_gate_reg = torch.tensor(0.0, device=x.device)
            total_inc_gate_info = torch.tensor(0.0, device=x.device)
            total_inc_gate_reg = torch.tensor(0.0, device=x.device)
            total_kl_count = 0
            total_inc_count = 0

        for i, stage in enumerate(self.stages):
            if self.use_learnable_gate and hasattr(stage, "gate_losses"):
                stage.gate_losses = []

            B = x.shape[0]
            input_H = hw_shape[0]
            input_W = hw_shape[1]
            input_C = x.shape[2]
            window_size = (
                stage.blocks[0].window_size
                if hasattr(stage.blocks[0], "window_size")
                else 7
            )
            input_windows = (input_H // window_size) * (input_W // window_size) * B

            next_stage_info = None
            if i < len(self.stages) - 1:
                next_stage = self.stages[i + 1]
                next_channel = (
                    next_stage.blocks[0].embed_dims
                    if hasattr(next_stage.blocks[0], "embed_dims")
                    else next_stage.blocks[0].attn.w_msa.embed_dims
                )
                next_window_size = (
                    next_stage.blocks[0].window_size
                    if hasattr(next_stage.blocks[0], "window_size")
                    else 7
                )
                next_hw_shape = (hw_shape[0] // 2, hw_shape[1] // 2)
                next_stage_info = {
                    "channel": next_channel,
                    "hw_shape": next_hw_shape,
                    "window_size": next_window_size,
                }

            x, hw_shape, out, out_hw_shape, cross_stage_info = stage(
                x, hw_shape, cross_stage_info, next_stage_info
            )

            if (
                self.use_learnable_gate
                and hasattr(stage, "gate_losses")
                and stage.gate_losses
            ):
                for item in stage.gate_losses:
                    (
                        b_kl_loss,
                        b_inc_loss,
                        b_kl_info,
                        b_kl_reg,
                        b_inc_info,
                        b_inc_reg,
                    ) = item
                    if b_kl_loss is not None:
                        total_kl_gate_loss = total_kl_gate_loss + b_kl_loss
                        total_kl_gate_info = total_kl_gate_info + b_kl_info
                        total_kl_gate_reg = total_kl_gate_reg + b_kl_reg
                        total_kl_count += 1
                    if b_inc_loss is not None:
                        total_inc_gate_loss = total_inc_gate_loss + b_inc_loss
                        total_inc_gate_info = total_inc_gate_info + b_inc_info
                        total_inc_gate_reg = total_inc_gate_reg + b_inc_reg
                        total_inc_count += 1
                stage.gate_losses.clear()

            kl_keep = 0
            inc_keep = 0
            block = stage.blocks[0]
            kl_ratio = getattr(block, "kl_ratio", None)
            inc_ratio = getattr(block, "inc_ratio", None)

            if cross_stage_info is not None:
                kl_keep_idx = cross_stage_info.get("prev_kl_keep_idx")
                if kl_keep_idx is not None:
                    kl_keep = kl_keep_idx.shape[0]
                    if kl_ratio and inc_ratio:
                        inc_keep = max(1, int(kl_keep * inc_ratio))

            kl_ratio_str = f"{kl_ratio:.2f}" if kl_ratio else "None"
            inc_ratio_str = f"{inc_ratio:.2f}" if inc_ratio else "None"
            # if torch.distributed.get_rank() == 0:
            #     print(f"[Stage {i}] Input: H={input_H}, W={input_W}, C={input_C}, windows={input_windows} (ws={window_size}) | KL: {kl_keep} ({kl_ratio_str}) | INC: {inc_keep} ({inc_ratio_str}) | Output: H={hw_shape[0]}, W={hw_shape[1]}, C={input_C}")

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, self.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)

        if self.use_learnable_gate:
            if total_kl_count > 1:
                total_kl_gate_loss = total_kl_gate_loss / total_kl_count
                total_kl_gate_info = total_kl_gate_info / total_kl_count
                total_kl_gate_reg = total_kl_gate_reg / total_kl_count
            if total_inc_count > 1:
                total_inc_gate_loss = total_inc_gate_loss / total_inc_count
                total_inc_gate_info = total_inc_gate_info / total_inc_count
                total_inc_gate_reg = total_inc_gate_reg / total_inc_count
            self.kl_predictor_gate_loss = total_kl_gate_loss
            self.kl_predictor_gate_info = total_kl_gate_info
            self.kl_predictor_gate_reg = total_kl_gate_reg
            self.inc_predictor_gate_loss = total_inc_gate_loss
            self.inc_predictor_gate_info = total_inc_gate_info
            self.inc_predictor_gate_reg = total_inc_gate_reg

        return outs

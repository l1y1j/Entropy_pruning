import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from mmcv.cnn import build_norm_layer
from mmdet.registry import MODELS
from mmdet.models.backbones.swin import (
    SwinTransformer, SwinBlock, SwinBlockSequence,
    ShiftWindowMSA, WindowMSA
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


def _collect_kl_scores(window_scores, stage_idx, block_idx):
    """收集KL分数用于后续分析"""
    global _kl_img_counter, _kl_img_limit, _kl_scores_collector, _current_epoch
    key = (stage_idx, block_idx, 'kl')
    if key not in _enabled_collections:
        return
    if _kl_img_counter < _kl_img_limit:
        _kl_scores_collector.append({
            'data': window_scores.detach().cpu().numpy(),
            'shape': window_scores.shape,
            'stage_idx': stage_idx,
            'block_idx': block_idx,
            'epoch': _current_epoch,
            'strategy': 'kl'
        })
        _kl_img_counter += 1

    if _kl_img_counter >= _kl_img_limit and len(_kl_scores_collector) > 0:
        _export_kl_scores()


def _collect_inc_scores(inc_scores, stage_idx, block_idx):
    """收集INC分数用于后续分析"""
    global _inc_img_counter, _inc_img_limit, _inc_scores_collector, _current_epoch
    key = (stage_idx, block_idx, 'inc')
    if key not in _enabled_collections:
        return
    if _inc_img_counter < _inc_img_limit:
        _inc_scores_collector.append({
            'data': inc_scores.detach().cpu().numpy(),
            'shape': inc_scores.shape,
            'stage_idx': stage_idx,
            'block_idx': block_idx,
            'epoch': _current_epoch,
            'strategy': 'inc'
        })
        _inc_img_counter += 1

    if _inc_img_counter >= _inc_img_limit and len(_inc_scores_collector) > 0:
        _export_inc_scores()


def collect_threshold(threshold_value, stage_idx, block_idx, strategy):
    """收集每个stage/block的threshold用于后续分析"""
    global _current_epoch
    key = (stage_idx, block_idx, strategy)
    if key not in _enabled_collections:
        return
    if strategy == 'kl':
        if key not in _kl_threshold_collector:
            _kl_threshold_collector[key] = []
        _kl_threshold_collector[key].append({
            'threshold': float(threshold_value),
            'stage_idx': stage_idx,
            'block_idx': block_idx,
            'epoch': _current_epoch
        })
    elif strategy == 'inc':
        if key not in _inc_threshold_collector:
            _inc_threshold_collector[key] = []
        _inc_threshold_collector[key].append({
            'threshold': float(threshold_value),
            'stage_idx': stage_idx,
            'block_idx': block_idx,
            'epoch': _current_epoch
        })


def _export_kl_scores():
    """导出收集的KL分数到文件"""
    global _kl_scores_collector, _current_epoch, _kl_threshold_collector
    if _kl_scores_collector:
        export_dir = os.path.join(os.path.dirname(__file__), 'kl_scores_export', f'epoch_{_current_epoch:03d}')
        os.makedirs(export_dir, exist_ok=True)
        save_path = os.path.join(export_dir, 'kl_scores.pkl')

        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(_kl_scores_collector, f)

        _kl_scores_collector.clear()

    # 导出 KL thresholds
    if _kl_threshold_collector:
        export_dir = os.path.join(os.path.dirname(__file__), 'kl_scores_export', f'epoch_{_current_epoch:03d}')
        os.makedirs(export_dir, exist_ok=True)
        save_path = os.path.join(export_dir, 'kl_thresholds.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(_kl_threshold_collector, f)
        _kl_threshold_collector.clear()


def _export_inc_scores():
    """导出收集的INC分数到文件"""
    global _inc_scores_collector, _current_epoch, _inc_threshold_collector
    if _inc_scores_collector:
        export_dir = os.path.join(os.path.dirname(__file__), 'kl_scores_export', f'epoch_{_current_epoch:03d}')
        os.makedirs(export_dir, exist_ok=True)
        save_path = os.path.join(export_dir, 'inc_scores.pkl')

        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(_inc_scores_collector, f)

        _inc_scores_collector.clear()

    # 导出 INC thresholds
    if _inc_threshold_collector:
        export_dir = os.path.join(os.path.dirname(__file__), 'kl_scores_export', f'epoch_{_current_epoch:03d}')
        os.makedirs(export_dir, exist_ok=True)
        save_path = os.path.join(export_dir, 'inc_thresholds.pkl')
        with open(save_path, 'wb') as f:
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
    global _kl_img_counter, _inc_img_counter, _kl_threshold_collector, _inc_threshold_collector
    _kl_img_counter = 0
    _inc_img_counter = 0
    _kl_threshold_collector.clear()
    _inc_threshold_collector.clear()


def compute_window_relative_entropy(x_windows: torch.Tensor, B: int, window_size: int = 7) -> torch.Tensor:
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
    kl = (local_dist * torch.log((local_dist + 1e-8) / (global_dist + 1e-8))).sum(dim=-1)
    
    return kl.view(-1)


# ========== 可学习门控阈值相关 ==========

class ThresholdPredictor(nn.Module):
    """预测KL或INC的门控阈值"""

    def __init__(self, input_dim=6, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0~1之间的阈值
        )

    def forward(self, stats_mean, stats_std, stats_p50, stats_max, stage_idx, block_idx):
        """
        Args:
            stats_mean: (B, 1) 分数均值
            stats_std: (B, 1) 分数标准差
            stats_p50: (B, 1) 分数中位数
            stats_max: (B, 1) 分数最大值
            stage_idx: int 当前stage索引
            block_idx: int 当前block索引

        Returns:
            threshold: (B, 1) 预测的阈值τ
        """
        B = stats_mean.shape[0]
        mlp_input = torch.cat([
            stats_mean, stats_std, stats_p50, stats_max,
            torch.full((B, 1), stage_idx / 3.0, device=stats_mean.device),
            torch.full((B, 1), block_idx / 6.0, device=stats_mean.device)
        ], dim=1)
        return self.net(mlp_input)  # (B, 1)


def compute_soft_mask(scores: torch.Tensor, threshold: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
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
        m_mask = torch.where(torch.isfinite(m_mask), m_mask, torch.ones_like(m_mask) * 0.5)
    
    # 裁剪到有效范围
    m_mask = torch.clamp(m_mask, min=1e-6, max=1 - 1e-6)
    
    return m_mask


def collect_stats(scores: torch.Tensor, batch_size: int) -> tuple:
    """收集分数的4维统计量

    Args:
        scores: (N,) 窗口分数 flatten 为 (B*N,)
        batch_size: 实际的batch size

    Returns:
        (mean, std, p50, max): 各维度张量，形状 (B, 1) 以匹配 MLP 输入
    """
    B = batch_size
    N = scores.shape[0] // B
    scores_per_batch = scores.view(B, N)
    mean = scores_per_batch.mean(dim=1, keepdim=True)  # (B, 1)
    std = scores_per_batch.std(dim=1, keepdim=True)    # (B, 1)
    p50 = scores_per_batch.quantile(0.5, dim=1, keepdim=True)  # (B, 1)
    max_val = scores_per_batch.max(dim=1, keepdim=True)[0]  # (B, 1)
    return mean, std, p50, max_val


def compute_gate_loss(m_mask: torch.Tensor, scores: torch.Tensor, lambda_reg: float) -> torch.Tensor:
    """计算Gate Loss

    L = (1 - sum(m_i * S_i) / sum(S_i))^2 + lambda * sum(m_i) / N

    Args:
        m_mask: (N,) 软掩码
        scores: (N,) 窗口分数
        lambda_reg: 正则化系数

    Returns:
        loss: 标量
    """
    N = scores.shape[0]
    reserved_sum = (m_mask * scores).sum()
    total_sum = scores.sum()
    
    # 数值保护：确保 total_sum 是有限的
    if not torch.isfinite(total_sum):
        total_sum = torch.tensor(1e-8, device=scores.device)
    
    info_ratio = reserved_sum / (total_sum + 1e-8)
    activation_ratio = m_mask.sum() / N
    
    loss = (1 - info_ratio) ** 2 + lambda_reg * activation_ratio
    
    # 确保 loss 是有限的
    if not torch.isfinite(loss):
        loss = torch.tensor(0.0, device=scores.device, requires_grad=True)
    
    return loss


class CrossStageInfo:
    """跨stage传递的对齐信息"""
    def __init__(self, aligned_windows, aligned_entropy, kl_scores, channel, hw_shape):
        self.aligned_windows = aligned_windows
        self.aligned_entropy = aligned_entropy
        self.kl_scores = kl_scores
        self.channel = channel
        self.hw_shape = hw_shape


def kl_weighted_pool_and_align(prev_windows: torch.Tensor, prev_kl_scores: torch.Tensor,
                                prev_full_entropy: torch.Tensor,
                                prev_channel: int, target_channel: int,
                                target_hw_shape: tuple, target_window_size: int,
                                prev_hw_shape: tuple, prev_window_size: int) -> tuple:
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
        B = prev_windows.shape[0] // (int(np.ceil(prev_hw_shape[0] / prev_window_size)) * int(np.ceil(prev_hw_shape[1] / prev_window_size)))

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
    prev_windows_2d = prev_windows.view(B, H_prev, W_prev, prev_window_size, prev_window_size, C_prev)
    prev_feat_2d = prev_windows_2d.permute(0, 5, 1, 3, 2, 4).contiguous()
    prev_feat_2d = prev_feat_2d.view(B, C_prev, H_prev * prev_window_size, W_prev * prev_window_size)

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
                    down_windows_b.append(torch.zeros(target_window_size, target_window_size, C_prev, device=feat_b.device, dtype=feat_b.dtype))
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

                feat_2x2 = torch.stack(feat_2x2, dim=0)  # (num_neighbors, ws, ws, C_prev)
                weights_2x2_norm = weights_2x2_norm.view(-1, 1, 1, 1)

                pooled_feat = (feat_2x2 * weights_2x2_norm).sum(dim=0)  # (ws, ws, C_prev)
                down_windows_b.append(pooled_feat)

                # KL-weighted pooling of entropy
                entropy_2x2 = prev_full_entropy.view(B, N_prev)[b, indices]  # (num_neighbors,)
                pooled_ent = (entropy_2x2 * F.softmax(weights_2x2, dim=0)).sum()
                down_entropy_b.append(pooled_ent)

        down_windows_b = torch.stack(down_windows_b, dim=0)  # (H_down*W_down, ws, ws, C_prev)
        down_entropy_b = torch.stack(down_entropy_b, dim=0)  # (H_down*W_down,)

        downsampled_windows_list.append(down_windows_b)
        downsampled_entropy_list.append(down_entropy_b)

    aligned_windows = torch.cat(downsampled_windows_list, dim=0)  # (B*N_target, ws, ws, C_prev)
    aligned_entropy = torch.cat(downsampled_entropy_list, dim=0)  # (B*N_target,)

    # Step 3: Channel alignment if needed
    if prev_channel != target_channel:
        aligned_windows_flat = aligned_windows.view(B * H_down * W_down, target_window_size * target_window_size, prev_channel)
        aligned_windows_flat = aligned_windows_flat.permute(0, 2, 1)

        conv = nn.Conv1d(prev_channel, target_channel, kernel_size=1).to(aligned_windows.device)
        aligned_windows_flat = conv(aligned_windows_flat)
        aligned_windows_flat = aligned_windows_flat.permute(0, 2, 1)

        aligned_windows = aligned_windows_flat.view(B * H_down * W_down, target_window_size, target_window_size, target_channel)

    return aligned_windows, aligned_entropy


def compute_aligned_entropy_from_windows(x_windows: torch.Tensor, B: int, window_size: int = 7) -> torch.Tensor:
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
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        act_cfg: dict = dict(type='GELU'),
        norm_cfg: dict = dict(type='LN'),
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
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)
        
        from mmcv.cnn.bricks.transformer import FFN
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)
    
    def forward(self, x: torch.Tensor, hw_shape: tuple, 
                entropy_cache: torch.Tensor = None, 
                prev_aligned_entropy: torch.Tensor = None,
                prev_kl_keep_idx: torch.Tensor = None) -> tuple:
        """Forward function
        
        Args:
            x: input features
            hw_shape: spatial shape (H, W)
            entropy_cache: entropy from previous W-MSA in same stage for INC comparison
            prev_aligned_entropy: entropy from previous stage for cross-stage INC comparison
            prev_kl_keep_idx: KL keep indices from previous stage
        
        Returns:
            x: output features
            entropy_or_tuple: entropy value, or tuple with additional info for cross-stage
        """
        can_prune_kl = self.kl_ratio is not None and self.kl_ratio < 1.0
        can_prune_inc = self.inc_ratio is not None and self.inc_ratio < 1.0
        
        if self.shift_size > 0:
            return self._forward_base(x, hw_shape), None, None, None, None, None, None, None, None
        
        if not can_prune_kl and not can_prune_inc:
            return self._forward_base(x, hw_shape), None, None, None, None, None, None, None, None
        
        if self.strategy is not None:
            if self.strategy == 'kl_inc' and can_prune_kl and can_prune_inc:
                return self._forward_kl_inc(x, hw_shape, entropy_cache, prev_aligned_entropy, prev_kl_keep_idx)
            elif self.strategy in ['kl', 'kl_inc'] and can_prune_kl:
                return self._forward_kl(x, hw_shape)
            elif self.strategy in ['inc', 'kl_inc'] and can_prune_inc:
                return self._forward_inc(x, hw_shape, entropy_cache, prev_aligned_entropy, prev_kl_keep_idx)
        
        return self._forward_base(x, hw_shape), None, None, None, None, None, None, None, None
    
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
    
    def _forward_kl(self, x: torch.Tensor, hw_shape: tuple) -> tuple:
        """KL pruning forward"""
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, f'Input size mismatch: {L} vs {H}*{W}'
        
        x = x.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]
        
        shifted_x = x if self.shift_size == 0 else torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = self._window_partition(shifted_x)
        total_windows = x_windows.shape[0]
        N_win = total_windows // B
        
        if self.shift_size > 0 or self.kl_ratio is None:
            base_result = self._forward_base_with_pad(x, hw_shape, pad_r, pad_b)
            return base_result, None, None, None, None, None, None, None
        
        window_scores = compute_window_relative_entropy(x_windows, B, self.window_size)
        window_scores = window_scores.view(B, -1)
        _collect_kl_scores(window_scores, self.stage_idx, self.block_idx)

        k = max(1, int(N_win * self.kl_ratio))
        _, keep_idx = torch.topk(window_scores, k=k, dim=1)
        keep_idx = keep_idx.sort(dim=1)[0]
        
        batch_offsets = (torch.arange(B, device=x.device) * N_win).unsqueeze(1)
        keep_idx_flat = (keep_idx + batch_offsets).view(-1)
        
        x_to_attn = x_windows[keep_idx_flat]
        x_to_attn = x_to_attn.view(-1, self.window_size * self.window_size, C)
        
        identity_attn = x_to_attn
        x_after_attn = self.norm1(x_to_attn)
        x_after_attn = self.attn.w_msa(x_after_attn)
        x_after_attn = identity_attn + x_after_attn
        
        cur_entropy_local = self._compute_entropy(x_after_attn)
        full_entropy = torch.zeros(total_windows, device=x.device)
        full_entropy[keep_idx_flat] = cur_entropy_local.detach()
        
        identity_ffn = x_after_attn
        x_after_ffn = self.norm2(x_after_attn)
        x_after_ffn = self.ffn(x_after_ffn)
        x_after_ffn = identity_ffn + x_after_ffn
        
        x_windows_new = x_windows.clone()
        x_ffn_reshaped = x_after_ffn.view(-1, self.window_size, self.window_size, C)
        x_windows_new[keep_idx_flat] = x_ffn_reshaped
        
        attn_windows = x_windows_new.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, H_pad, W_pad)
        
        x = shifted_x if self.shift_size == 0 else torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        
        block_gate_loss = None
        if self.use_learnable_gate and self.kl_predictor is not None:
            scores_flat = window_scores.view(-1)
            stats_mean, stats_std, stats_p50, stats_max = collect_stats(scores_flat, B)
            threshold = self.kl_predictor(stats_mean, stats_std, stats_p50, stats_max,
                                          self.stage_idx, self.block_idx)
            m_mask = compute_soft_mask(scores_flat, threshold, self.temperature)
            kl_loss = compute_gate_loss(m_mask, scores_flat, self.lambda_kl)
            self.kl_gate_loss = kl_loss
            # threshold shape is (B, 1), collect for each batch element
            for b in range(B):
                collect_threshold(threshold[b].item(), self.stage_idx, self.block_idx, 'kl')
            block_gate_loss = (kl_loss, None)
        
        # 数值检查：确保输出不包含 NaN/Inf
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        return x, full_entropy, x_windows, x_windows, window_scores, (H_pad, W_pad), self.embed_dims, keep_idx_flat, block_gate_loss

    def _forward_inc(self, x: torch.Tensor, hw_shape: tuple, 
                  entropy_cache: torch.Tensor = None, 
                  prev_aligned_entropy: torch.Tensor = None,
                  prev_kl_keep_idx: torch.Tensor = None) -> tuple:
        """Incremental pruning forward with cross-layer and cross-stage comparison
        
        Args:
            entropy_cache: entropy from previous W-MSA in same stage
            prev_aligned_entropy: entropy from previous stage (aligned)
        """
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, f'Input size mismatch: {L} vs {H}*{W}'
        
        x = x.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]
        
        shifted_x = x if self.shift_size == 0 else torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = self._window_partition(shifted_x)
        total_windows = x_windows.shape[0]
        N_win = total_windows // B
        
        if self.shift_size > 0 or self.inc_ratio is None:
            base_result = self._forward_base_with_pad(x, hw_shape, pad_r, pad_b)
            return base_result, None, None, None, None, None, None, None
        
        x_windows_flat = x_windows.view(-1, self.window_size * self.window_size, C)
        identity_attn = x_windows_flat
        x_after_attn = self.norm1(x_windows_flat)
        x_after_attn = self.attn.w_msa(x_after_attn)
        x_after_attn = identity_attn + x_after_attn
        
        cur_entropy = self._compute_entropy(x_after_attn)
        
        inc_scores = torch.zeros_like(cur_entropy)

        if entropy_cache is not None and entropy_cache.shape[0] == total_windows:
            # Intra-stage comparison (same stage, no downsampling needed)
            inc_scores = torch.abs(cur_entropy - entropy_cache.detach())
        elif prev_aligned_entropy is not None and prev_aligned_entropy.shape[0] == total_windows:
            # Cross-stage comparison (different stage, needs downsampling)
            # After fixing issues 1 & 2, window counts should match exactly
            cross_stage_scores = torch.abs(cur_entropy - prev_aligned_entropy.detach())
            inc_scores = cross_stage_scores
        
        inc_scores = inc_scores.view(B, -1)
        _collect_inc_scores(inc_scores, self.stage_idx, self.block_idx)

        k = max(1, int(N_win * self.inc_ratio))
        _, inc_keep_idx = torch.topk(inc_scores, k=k, dim=1)
        inc_keep_idx = inc_keep_idx.sort(dim=1)[0]
        
        batch_offsets = (torch.arange(B, device=x.device) * N_win).unsqueeze(1)
        inc_keep_idx_flat = (inc_keep_idx + batch_offsets).view(-1)
        
        x_ffn_input = x_after_attn[inc_keep_idx_flat]
        identity_ffn = x_ffn_input
        x_ffn_input = self.norm2(x_ffn_input)
        x_ffn_input = self.ffn(x_ffn_input)
        x_ffn_input = identity_ffn + x_ffn_input
        
        x_windows_new = x_windows.clone()
        x_windows_new_reshaped = x_ffn_input.view(-1, self.window_size, self.window_size, C)
        x_windows_new[inc_keep_idx_flat] = x_windows_new_reshaped
        
        attn_windows = x_windows_new.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, H_pad, W_pad)
        
        x = shifted_x if self.shift_size == 0 else torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        
        block_gate_loss = None
        if self.use_learnable_gate and self.inc_predictor is not None:
            inc_scores_flat = inc_scores.view(-1)
            stats_mean, stats_std, stats_p50, stats_max = collect_stats(inc_scores_flat, B)
            threshold = self.inc_predictor(stats_mean, stats_std, stats_p50, stats_max,
                                          self.stage_idx, self.block_idx)
            m_mask = compute_soft_mask(inc_scores_flat, threshold, self.temperature)
            inc_loss = compute_gate_loss(m_mask, inc_scores_flat, self.lambda_inc)
            self.inc_gate_loss = inc_loss
            for b in range(B):
                collect_threshold(threshold[b].item(), self.stage_idx, self.block_idx, 'inc')
            block_gate_loss = (None, inc_loss)
        
        # 数值检查：确保输出不包含 NaN/Inf
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        return x, cur_entropy.detach(), x_windows, None, hw_shape, self.embed_dims, block_gate_loss

    def _forward_kl_inc(self, x: torch.Tensor, hw_shape: tuple, 
                     entropy_cache: torch.Tensor = None,
                     prev_aligned_entropy: torch.Tensor = None,
                     prev_kl_keep_idx: torch.Tensor = None) -> tuple:
        """KL + INC 串联筛选 with cross-layer and cross-stage comparison"""
        if self.inc_ratio is None:
            return self._forward_kl(x, hw_shape)
        
        B, L, C = x.shape
        H, W = hw_shape
        x = x.view(B, H, W, C)
        
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]
        
        shifted_x = x if self.shift_size == 0 else torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = self._window_partition(shifted_x)
        total_windows = x_windows.shape[0]
        N_win = total_windows // B
        batch_offsets = (torch.arange(B, device=x.device) * N_win).unsqueeze(1)
        
        if self.shift_size > 0 or self.kl_ratio is None:
            base_result = self._forward_base_with_pad(x, hw_shape, pad_r, pad_b)
            return base_result, None, None, None, None, None, None, None
        
        window_scores = compute_window_relative_entropy(x_windows, B, self.window_size)
        window_scores = window_scores.view(B, -1)
        _collect_kl_scores(window_scores, self.stage_idx, self.block_idx)

        k_kl = max(1, int(N_win * self.kl_ratio))
        _, kl_keep_idx = torch.topk(window_scores, k=k_kl, dim=1)
        kl_keep_idx = kl_keep_idx.sort(dim=1)[0]
        
        kl_keep_idx_flat = (kl_keep_idx + batch_offsets).view(-1)
        
        x_windows_flat = x_windows.view(-1, self.window_size * self.window_size, C)
        full_entropy_before_kl = self._compute_entropy(x_windows_flat)
        
        x_kl = x_windows[kl_keep_idx_flat]
        x_kl = x_kl.view(-1, self.window_size * self.window_size, C)
        identity_attn = x_kl
        x_kl = self.norm1(x_kl)
        x_kl = self.attn.w_msa(x_kl)
        x_kl = identity_attn + x_kl
        
        cur_entropy = self._compute_entropy(x_kl)
        
        inc_scores = torch.zeros_like(cur_entropy)

        if entropy_cache is not None and entropy_cache.shape[0] == total_windows:
            # Intra-stage comparison (same stage, no downsampling needed)
            inc_scores = torch.abs(cur_entropy - entropy_cache[kl_keep_idx_flat].detach())
        elif prev_aligned_entropy is not None and prev_aligned_entropy.shape[0] == total_windows:
            # Cross-stage comparison (different stage, needs downsampling)
            # After fixing issues 1 & 2, window counts should match exactly
            cross_stage_scores = torch.abs(cur_entropy - prev_aligned_entropy[kl_keep_idx_flat].detach())
            inc_scores = cross_stage_scores
        
        inc_scores = inc_scores.view(B, -1)
        _collect_inc_scores(inc_scores, self.stage_idx, self.block_idx)

        k_inc = max(1, int(k_kl * self.inc_ratio))
        _, inc_keep_idx = torch.topk(inc_scores, k=k_inc, dim=1)
        inc_keep_idx = inc_keep_idx.sort(dim=1)[0]
        
        kl_batch_offsets = (torch.arange(B, device=x.device) * k_kl).unsqueeze(1)
        inc_keep_idx_flat = (inc_keep_idx + kl_batch_offsets).view(-1)
        
        x_ffn_input = x_kl[inc_keep_idx_flat]
        identity_ffn = x_ffn_input
        x_ffn_input = self.norm2(x_ffn_input)
        x_ffn_input = self.ffn(x_ffn_input)
        x_ffn_input = identity_ffn + x_ffn_input
        
        x_windows_new = x_windows.clone()
        x_kl_reshaped = x_kl.view(-1, self.window_size, self.window_size, C)
        x_windows_new[kl_keep_idx_flat] = x_kl_reshaped
        
        x_ffn_reshaped = x_ffn_input.view(-1, self.window_size, self.window_size, C)
        x_windows_new[kl_keep_idx_flat[inc_keep_idx_flat]] = x_ffn_reshaped
        
        attn_windows = x_windows_new.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, H_pad, W_pad)
        
        x = shifted_x if self.shift_size == 0 else torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        
        full_entropy = full_entropy_before_kl
        
        block_gate_loss = None
        if self.use_learnable_gate:
            # KL 门控
            if self.kl_predictor is not None:
                kl_scores_flat = window_scores.view(-1)
                stats_mean, stats_std, stats_p50, stats_max = collect_stats(kl_scores_flat, B)
                threshold = self.kl_predictor(stats_mean, stats_std, stats_p50, stats_max,
                                              self.stage_idx, self.block_idx)
                m_mask = compute_soft_mask(kl_scores_flat, threshold, self.temperature)
                kl_loss = compute_gate_loss(m_mask, kl_scores_flat, self.lambda_kl)
                self.kl_gate_loss = kl_loss
                for b in range(B):
                    collect_threshold(threshold[b].item(), self.stage_idx, self.block_idx, 'kl')
            else:
                kl_loss = None

            # INC 门控
            if self.inc_predictor is not None and inc_scores.numel() > 0:
                inc_scores_flat = inc_scores.view(-1)
                stats_mean, stats_std, stats_p50, stats_max = collect_stats(inc_scores_flat, B)
                threshold = self.inc_predictor(stats_mean, stats_std, stats_p50, stats_max,
                                              self.stage_idx, self.block_idx)
                m_mask = compute_soft_mask(inc_scores_flat, threshold, self.temperature)
                inc_loss = compute_gate_loss(m_mask, inc_scores_flat, self.lambda_inc)
                self.inc_gate_loss = inc_loss
                for b in range(B):
                    collect_threshold(threshold[b].item(), self.stage_idx, self.block_idx, 'inc')
            else:
                inc_loss = None
            
            block_gate_loss = (kl_loss, inc_loss)
        
        # 数值检查：确保输出不包含 NaN/Inf
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        return x, full_entropy, x_windows, x_windows, window_scores, (H_pad, W_pad), self.embed_dims, kl_keep_idx_flat, block_gate_loss

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
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def _forward_base_with_pad(self, x: torch.Tensor, hw_shape: tuple, pad_r: int, pad_b: int) -> torch.Tensor:
        """Base forward with padding handling"""
        B, H, W, C = x.shape
        shifted_x = x if self.shift_size == 0 else torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
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
        x = shifted_x if self.shift_size == 0 else torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
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
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        downsample: nn.Module = None,
        act_cfg: dict = dict(type='GELU'),
        norm_cfg: dict = dict(type='LN'),
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
                drop_path_rate=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
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
    
    def forward(self, x: torch.Tensor, hw_shape: tuple, 
                prev_cross_stage_info: dict = None,
                next_stage_info: dict = None):
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
        if prev_cross_stage_info is not None:
            prev_aligned_entropy = prev_cross_stage_info.get('prev_aligned_entropy', None)
            prev_kl_keep_idx = prev_cross_stage_info.get('prev_kl_keep_idx', None)
        
        last_wmsa_info = None
        
        for i, block in enumerate(self.blocks):
            result = block(x, hw_shape, entropy_cache, prev_aligned_entropy, prev_kl_keep_idx)
            
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
                    if not hasattr(self, 'gate_losses'):
                        self.gate_losses = []
                    self.gate_losses.append(block_gate_loss)
                
                if block.shift_size == 0:
                    prev_entropy = new_entropy
                    entropy_cache = prev_entropy
                    prev_kl_keep_idx = kl_keep_idx
                    
                    if windows is not None and kl_scores is not None:
                        last_wmsa_info = {
                            'windows': windows,
                            'kl_scores': kl_scores,
                            'kl_keep_idx': kl_keep_idx,
                            'hw_shape': block_hw_shape,
                            'channel': channel,
                            'entropy': new_entropy,
                            'window_size': block.window_size
                        }
            else:
                x = result[0] if isinstance(result, tuple) else result
                prev_aligned_entropy = None
                prev_kl_keep_idx = None
        
        cross_stage_info = None
        if last_wmsa_info is not None:
            base_cross_stage_info = {
                'prev_aligned_entropy': last_wmsa_info['entropy'],
                'prev_windows': last_wmsa_info['windows'],
                'prev_kl_scores': last_wmsa_info['kl_scores'],
                'prev_kl_keep_idx': last_wmsa_info['kl_keep_idx'],
                'prev_channel': last_wmsa_info['channel'],
                'prev_hw_shape': last_wmsa_info['hw_shape'],
                'window_size': self.blocks[0].window_size
            }
            
            if next_stage_info is not None:
                aligned_windows, aligned_entropy = kl_weighted_pool_and_align(
                    prev_windows=last_wmsa_info['windows'],
                    prev_kl_scores=last_wmsa_info['kl_scores'],
                    prev_full_entropy=last_wmsa_info['entropy'],
                    prev_channel=last_wmsa_info['channel'],
                    target_channel=next_stage_info['channel'],
                    target_hw_shape=next_stage_info['hw_shape'],
                    target_window_size=next_stage_info['window_size'],
                    prev_hw_shape=last_wmsa_info['hw_shape'],
                    prev_window_size=last_wmsa_info.get('window_size', 7)
                )
                cross_stage_info = {
                    'prev_aligned_entropy': aligned_entropy,
                    'prev_windows': aligned_windows,
                    'prev_kl_scores': last_wmsa_info['kl_scores'],
                    'prev_kl_keep_idx': last_wmsa_info['kl_keep_idx'],
                    'prev_channel': next_stage_info['channel'],
                    'prev_hw_shape': next_stage_info['hw_shape'],
                    'window_size': next_stage_info['window_size']
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
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        use_abs_pos_embed: bool = False,
        act_cfg: dict = dict(type='GELU'),
        norm_cfg: dict = dict(type='LN'),
        with_cp: bool = False,
        pretrained: str = None,
        convert_weights: bool = False,
        frozen_stages: int = -1,
        init_cfg: dict = None,
        strategy: str = 'base',
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
            self.register_buffer('kl_predictor_gate_loss', torch.zeros(1))
            self.register_buffer('inc_predictor_gate_loss', torch.zeros(1))
        else:
            self.kl_predictor = None
            self.inc_predictor = None
        
        if strategy == 'kl':
            self._replace_blocks_with_kl()
        elif strategy == 'inc':
            self._replace_blocks_with_inc()
        elif strategy == 'kl_inc':
            self._replace_blocks_with_kl_inc()
    
    def _replace_blocks_with_kl(self):
        """Replace blocks with KL-enabled blocks"""
        from mmdet.models.layers import PatchMerging
        
        default_act_cfg = dict(type='GELU')
        default_norm_cfg = dict(type='LN')
        
        num_layers = len(self.stages)
        
        for stage_idx in range(num_layers):
            stage = self.stages[stage_idx]
            
            # Get stage config
            stage_cfg = self.stage_config.get(stage_idx, {})
            stage_blocks = stage_cfg.get('blocks', [])
            stage_ratio = stage_cfg.get('ratio', [])
            
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
            drop_rate = 0.
            attn_drop_rate = 0.
            for child in first_block.attn.w_msa.children():
                if hasattr(child, 'drop_prob'):
                    attn_drop_rate = child.drop_prob
                    break
            drop_path_rates = []
            for b in stage.blocks:
                dp = 0.
                for child in b.attn.children():
                    if hasattr(child, 'drop_prob'):
                        dp = child.drop_prob
                        break
                drop_path_rates.append(dp)
            
            # Collect KL ratios for each block
            block_kl_ratios = []
            for block_idx in range(depth):
                if block_idx in stage_blocks:
                    # Find ratio for this block
                    idx_in_blocks = stage_blocks.index(block_idx)
                    kl_ratio = stage_ratio[idx_in_blocks] if idx_in_blocks < len(stage_ratio) else stage_ratio[-1]
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
                strategy='kl',
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
        default_act_cfg = dict(type='GELU')
        default_norm_cfg = dict(type='LN')
        
        num_layers = len(self.stages)
        
        for stage_idx in range(num_layers):
            stage = self.stages[stage_idx]
            
            stage_cfg = self.inc_stage_config.get(stage_idx, {})
            stage_blocks = stage_cfg.get('blocks', [])
            stage_inc_ratio = stage_cfg.get('inc_ratio', [])
            
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
            drop_rate = 0.
            attn_drop_rate = 0.
            for child in first_block.attn.w_msa.children():
                if hasattr(child, 'drop_prob'):
                    attn_drop_rate = child.drop_prob
                    break
            drop_path_rates = []
            for b in stage.blocks:
                dp = 0.
                for child in b.attn.children():
                    if hasattr(child, 'drop_prob'):
                        dp = child.drop_prob
                        break
                drop_path_rates.append(dp)
            
            block_inc_ratios = []
            for block_idx in range(depth):
                if block_idx in stage_blocks:
                    idx_in_blocks = stage_blocks.index(block_idx)
                    inc_ratio = stage_inc_ratio[idx_in_blocks] if idx_in_blocks < len(stage_inc_ratio) else stage_inc_ratio[-1]
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
                strategy='inc',
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
        default_act_cfg = dict(type='GELU')
        default_norm_cfg = dict(type='LN')
        
        num_layers = len(self.stages)
        
        for stage_idx in range(num_layers):
            stage = self.stages[stage_idx]
            
            kl_stage_cfg = self.stage_config.get(stage_idx, {})
            kl_blocks = kl_stage_cfg.get('blocks', [])
            kl_ratios = kl_stage_cfg.get('ratio', [])
            
            inc_stage_cfg = self.inc_stage_config.get(stage_idx, {})
            inc_blocks = inc_stage_cfg.get('blocks', [])
            inc_ratios = inc_stage_cfg.get('inc_ratio', [])
            
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
            drop_rate = 0.
            attn_drop_rate = 0.
            for child in first_block.attn.w_msa.children():
                if hasattr(child, 'drop_prob'):
                    attn_drop_rate = child.drop_prob
                    break
            drop_path_rates = []
            for b in stage.blocks:
                dp = 0.
                for child in b.attn.children():
                    if hasattr(child, 'drop_prob'):
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
                    kl_ratio = kl_ratios[idx_in_kl] if idx_in_kl < len(kl_ratios) else kl_ratios[-1]
                if block_idx in inc_blocks:
                    idx_in_inc = inc_blocks.index(block_idx)
                    inc_ratio = inc_ratios[idx_in_inc] if idx_in_inc < len(inc_ratios) else inc_ratios[-1]
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
                strategy='kl_inc',
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
            'strategy': self.strategy,
            'stage_config': self.stage_config,
            'inc_stage_config': self.inc_stage_config,
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
        
        for i, stage in enumerate(self.stages):
            if self.use_learnable_gate and hasattr(stage, 'gate_losses'):
                stage.gate_losses = []
            
            B = x.shape[0]
            input_H = hw_shape[0]
            input_W = hw_shape[1]
            input_C = x.shape[2]
            window_size = stage.blocks[0].window_size if hasattr(stage.blocks[0], 'window_size') else 7
            input_windows = (input_H // window_size) * (input_W // window_size) * B
            
            next_stage_info = None
            if i < len(self.stages) - 1:
                next_stage = self.stages[i + 1]
                next_channel = next_stage.blocks[0].embed_dims if hasattr(next_stage.blocks[0], 'embed_dims') else next_stage.blocks[0].attn.w_msa.embed_dims
                next_window_size = next_stage.blocks[0].window_size if hasattr(next_stage.blocks[0], 'window_size') else 7
                next_hw_shape = (hw_shape[0] // 2, hw_shape[1] // 2)
                next_stage_info = {
                    'channel': next_channel,
                    'hw_shape': next_hw_shape,
                    'window_size': next_window_size
                }
            
            x, hw_shape, out, out_hw_shape, cross_stage_info = stage(x, hw_shape, cross_stage_info, next_stage_info)
            
            if self.use_learnable_gate and hasattr(stage, 'gate_losses') and stage.gate_losses:
                for b_kl_loss, b_inc_loss in stage.gate_losses:
                    if b_kl_loss is not None:
                        total_kl_gate_loss = total_kl_gate_loss + b_kl_loss
                    if b_inc_loss is not None:
                        total_inc_gate_loss = total_inc_gate_loss + b_inc_loss
                stage.gate_losses.clear()
            
            kl_keep = 0
            inc_keep = 0
            block = stage.blocks[0]
            kl_ratio = getattr(block, 'kl_ratio', None)
            inc_ratio = getattr(block, 'inc_ratio', None)
            
            if cross_stage_info is not None:
                kl_keep_idx = cross_stage_info.get('prev_kl_keep_idx')
                if kl_keep_idx is not None:
                    kl_keep = kl_keep_idx.shape[0]
                    if kl_ratio and inc_ratio:
                        inc_keep = max(1, int(kl_keep * inc_ratio))
            
            kl_ratio_str = f"{kl_ratio:.2f}" if kl_ratio else "None"
            inc_ratio_str = f"{inc_ratio:.2f}" if inc_ratio else "None"
            # if torch.distributed.get_rank() == 0:
            #     print(f"[Stage {i}] Input: H={input_H}, W={input_W}, C={input_C}, windows={input_windows} (ws={window_size}) | KL: {kl_keep} ({kl_ratio_str}) | INC: {inc_keep} ({inc_ratio_str}) | Output: H={hw_shape[0]}, W={hw_shape[1]}, C={input_C}")
            
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
        
        if self.use_learnable_gate:
            dummy_grad = 0.0 * sum(p.sum() for p in self.parameters() if p.requires_grad)
            self.kl_predictor_gate_loss = total_kl_gate_loss + dummy_grad
            self.inc_predictor_gate_loss = total_inc_gate_loss + dummy_grad
        
        return outs
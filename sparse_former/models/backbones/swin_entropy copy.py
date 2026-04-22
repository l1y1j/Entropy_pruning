import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.models.backbones.swin import SwinTransformer, SwinBlock, SwinBlockSequence
import numpy as np
from PIL import Image

from typing import List
from sparse_former.utils.block_entropy import (
    compute_window_relative_entropy,
    compute_attention_entropy_from_features
)


class SwinBlockEntropy(SwinBlock):
    """Swin Block with Entropy-based Window Pruning

    支持三种策略:
    - 'origin': 不做任何剪枝，和原始SwinBlock一样
    - 'kl': 单次KL筛选，所有保留窗口都过Attn+FFN
    - 'kl_incremental': 两次筛选，KL过滤 → Attn → 增量筛选决定谁过FFN

    剪枝只会在 shift_size == 0 的block上生效（对应Swin的W-MSA阶段）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_entropy_params()

        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        if drop_path_rate > 0:
            from mmdet.models.utils import build_dropout
            self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        else:
            self.drop_path = None

    def _init_entropy_params(self):
        """统一初始化熵剪枝相关参数"""
        self.enable_entropy = False
        self.entropy_strategy = 'origin'  # 'origin', 'kl', 'kl_incremental'
        # kl策略参数
        self.kl_ratio = None  # 用于 kl 和 kl_incremental 的KL筛选阶段
        # kl_incremental 额外参数
        self.increment_ratio = None  # 用于 kl_incremental 的增量筛选阶段
        # 用于跨block传递熵值（kl_incremental策略）
        self._attn_entropy_cache = None

    def set_entropy_config(self, strategy, kl_ratio=None, increment_ratio=None):
        """统一设置熵剪枝配置

        Args:
            strategy: 策略名称 - 'origin', 'kl', 'kl_incremental', 'incremental_only'
            kl_ratio: KL筛选的保留比例 (0-1)，用于 kl 和 kl_incremental
            increment_ratio: 增量筛选的保留比例 (0-1)，用于 kl_incremental 和 incremental_only
        """
        self.entropy_strategy = strategy
        self.enable_entropy = (strategy not in ['origin'])
        self.kl_ratio = kl_ratio
        self.increment_ratio = increment_ratio if strategy in ['kl_incremental', 'incremental_only'] else None

    def forward(self, x, hw_shape, attn_entropy_cache=None):
        """前向传播

        Args:
            x: (B, L, C) 输入特征
            hw_shape: (H, W) 空间形状
            attn_entropy_cache: (B * N_win,) 来自前一个block的熵缓存，仅kl_incremental使用

        Returns:
            x: 输出特征
            attn_entropy_cache: 更新后的熵缓存，仅kl_incremental策略返回值有意义
        """
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C)

        # Padding
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._create_attn_mask(H_pad, W_pad)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = self.window_partition(shifted_x)
        total_windows = x_windows.shape[0]
        N_win = total_windows // B

        # 计算batch offsets用于索引
        batch_offsets = (torch.arange(B, device=x.device) * N_win).unsqueeze(1)

        # ============================================
        # 熵剪枝逻辑 - 根据策略分支
        # ============================================
        x_to_attn = x_windows
        mask_to_attn = attn_mask
        kl_keep_flat = None
        inc_topk_idx_flat = None

        if self.enable_entropy and self.shift_size == 0:
            # ---------- KL筛选阶段 ----------
            if self.kl_ratio is not None and self.kl_ratio > 0:
                window_scores = compute_window_relative_entropy(x_windows, B, self.window_size)
                window_scores = window_scores.view(B, -1)

                k = max(1, int(N_win * self.kl_ratio))
                _, keep_idx = torch.topk(window_scores, k=k, dim=1)
                keep_idx = keep_idx.sort(dim=1)[0]

                kl_keep_flat = (keep_idx + batch_offsets).view(-1)

                # Gather筛选后的窗口
                x_to_attn = x_windows[kl_keep_flat]
                mask_to_attn = attn_mask[kl_keep_flat] if attn_mask is not None else None

        x_to_attn = x_to_attn.view(-1, self.window_size * self.window_size, C)

        # ---------- Attention ----------
        identity_attn = x_to_attn
        x_after_attn = self.norm1(x_to_attn)
        x_after_attn = self.attn.w_msa(x_after_attn, mask=mask_to_attn)

        if self.drop_path is not None:
            x_after_attn = identity_attn + self.drop_path(x_after_attn)
        else:
            x_after_attn = identity_attn + x_after_attn

        # 计算当前熵值
        cur_entropy = compute_attention_entropy_from_features(x_after_attn)

        # 初始化熵缓存
        if attn_entropy_cache is None:
            attn_entropy_cache = torch.zeros(total_windows, device=x.device)

        # ============================================
        # 策略分支：决定如何处理Attention输出
        # ============================================
        if self.enable_entropy and self.shift_size == 0:
            print(f"[Forward] strategy={self.entropy_strategy}, kl_ratio={self.kl_ratio}, inc_ratio={self.increment_ratio}, kl_keep_flat={kl_keep_flat is not None}")
            if self.entropy_strategy == 'incremental_only' and self.increment_ratio is not None:
                # ---------- incremental_only: 无KL预筛选，直接增量筛选 ----------
                x_windows_processed = self._forward_incremental_only(
                    x_windows, x_after_attn, cur_entropy,
                    attn_entropy_cache, batch_offsets, B, C
                )
            elif self.entropy_strategy == 'kl_incremental' and self.increment_ratio is not None and kl_keep_flat is not None:
                # ---------- kl_incremental: KL预筛选 + 增量筛选 ----------
                x_windows_processed = self._forward_kl_incremental(
                    x_windows, x_after_attn, kl_keep_flat, cur_entropy,
                    attn_entropy_cache, batch_offsets, B, C
                )
                # 更新缓存
                if inc_topk_idx_flat is not None:
                    final_keep_idx = kl_keep_flat[inc_topk_idx_flat]
                    new_entropy = compute_attention_entropy_from_features(
                        x_after_attn[inc_topk_idx_flat]
                    )
                    attn_entropy_cache[final_keep_idx] = new_entropy
            elif kl_keep_flat is not None:
                # ---------- kl: 所有KL保留窗口都过FFN ----------
                x_windows_processed = self._forward_kl_only(
                    x_windows, x_after_attn, kl_keep_flat, B, C
                )
                # 更新缓存
                attn_entropy_cache[kl_keep_flat] = compute_attention_entropy_from_features(
                    x_after_attn
                )
            else:
                # ---------- 未匹配任何策略（不应该发生） ----------
                x_windows_processed = x_windows
        else:
            # ---------- origin: 不剪枝 ----------
            x_windows_processed = x_windows

        # Window reverse & unshift
        attn_windows = x_windows_processed.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Unpad
        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        return x, attn_entropy_cache

    def _forward_kl_only(self, x_windows, x_after_attn, kl_keep_flat, B, C):
        """KL策略：所有KL筛选保留的窗口都过FFN"""
        # FFN处理
        identity_ffn = x_after_attn
        x_after_ffn = self.norm2(x_after_attn)
        x_after_ffn = self.ffn(x_after_ffn)

        if self.drop_path is not None:
            x_after_ffn = identity_ffn + self.drop_path(x_after_ffn)
        else:
            x_after_ffn = identity_ffn + x_after_ffn

        # Scatter回填
        x_windows_new = x_windows.clone()
        x_high_reshaped = x_after_ffn.view(-1, self.window_size, self.window_size, C)
        x_windows_new[kl_keep_flat] = x_high_reshaped

        return x_windows_new

    def _forward_kl_incremental(self, x_windows, x_after_attn, kl_keep_flat,
                                 cur_entropy, attn_entropy_cache, batch_offsets, B, C):
        """KL_Incremental策略：增量筛选，只让熵变大的窗口过FFN"""
        # 使用缓存的熵值计算增量分数
        cached_entropy = attn_entropy_cache[kl_keep_flat]
        inc_scores = torch.abs(cur_entropy - cached_entropy)

        # 按Batch解耦计算Top-K
        num_kept = inc_scores.shape[0]
        k_per_batch = num_kept // B
        inc_scores_batch = inc_scores.view(B, k_per_batch)

        inc_k = max(1, int(k_per_batch * self.increment_ratio))
        _, inc_topk_idx_local = torch.topk(inc_scores_batch, k=inc_k, dim=1)
        inc_topk_idx_local = inc_topk_idx_local.sort(dim=1)[0]

        inc_batch_offsets = (torch.arange(B, device=x_windows.device) * k_per_batch).unsqueeze(1)
        inc_topk_idx_flat = (inc_topk_idx_local + inc_batch_offsets).view(-1)

        # 保存给外部使用的索引
        self._current_inc_topk_idx = inc_topk_idx_flat

        # FFN只处理增量筛选后的窗口
        x_ffn_input = x_after_attn[inc_topk_idx_flat]
        identity_ffn = x_ffn_input
        x_ffn_input = self.norm2(x_ffn_input)
        x_ffn_input = self.ffn(x_ffn_input)

        if self.drop_path is not None:
            x_ffn_input = identity_ffn + self.drop_path(x_ffn_input)
        else:
            x_ffn_input = identity_ffn + x_ffn_input

        # 双重Scatter回填
        x_windows_processed = x_windows.clone()

        # 第一步：FFN结果写回Attn输出对应位置
        attn_output_windows = x_after_attn.clone().view(-1, self.window_size, self.window_size, C)
        ffn_output_windows = x_ffn_input.view(-1, self.window_size, self.window_size, C)
        attn_output_windows[inc_topk_idx_flat] = ffn_output_windows

        # 第二步：Attn输出写回全图
        x_windows_processed[kl_keep_flat] = attn_output_windows

        return x_windows_processed

    def _forward_incremental_only(self, x_windows, x_after_attn, cur_entropy,
                                  attn_entropy_cache, batch_offsets, B, C):
        """Incremental Only策略：直接对完整特征做增量筛选，无KL预筛选

        逻辑：
        1. 所有窗口都过 Attention（不做 KL 预筛选）
        2. 计算当前熵与缓存熵的差值
        3. 按 increment_ratio 筛选熵变化最大的窗口过 FFN
        4. Scatter 回填
        """
        total_windows = x_windows.shape[0]

        # 计算增量分数：当前熵 vs 缓存熵
        inc_scores = torch.abs(cur_entropy - attn_entropy_cache)

        # 调试打印
        print(f"[incremental_only] total_windows={total_windows}, inc_ratio={self.increment_ratio}, selected={int(total_windows * self.increment_ratio)}")

        # 按Batch解耦计算Top-K
        k_per_batch = total_windows // B
        inc_scores_batch = inc_scores.view(B, k_per_batch)

        inc_k = max(1, int(k_per_batch * self.increment_ratio))
        _, inc_topk_idx_local = torch.topk(inc_scores_batch, k=inc_k, dim=1)
        inc_topk_idx_local = inc_topk_idx_local.sort(dim=1)[0]

        inc_batch_offsets = (torch.arange(B, device=x_windows.device) * k_per_batch).unsqueeze(1)
        inc_topk_idx_flat = (inc_topk_idx_local + inc_batch_offsets).view(-1)

        # 保存给外部使用的索引
        self._current_inc_topk_idx = inc_topk_idx_flat

        # FFN只处理增量筛选后的窗口
        x_ffn_input = x_after_attn[inc_topk_idx_flat]
        identity_ffn = x_ffn_input
        x_ffn_input = self.norm2(x_ffn_input)
        x_ffn_input = self.ffn(x_ffn_input)

        if self.drop_path is not None:
            x_ffn_input = identity_ffn + self.drop_path(x_ffn_input)
        else:
            x_ffn_input = identity_ffn + x_ffn_input

        # Scatter回填：FFN结果写回全图
        x_windows_processed = x_windows.clone()
        attn_output_windows = x_after_attn.clone().view(-1, self.window_size, self.window_size, C)
        ffn_output_windows = x_ffn_input.view(-1, self.window_size, self.window_size, C)
        attn_output_windows[inc_topk_idx_flat] = ffn_output_windows

        # Scatter回全图：只更新过了FFN的窗口
        x_windows_processed[inc_topk_idx_flat] = attn_output_windows[inc_topk_idx_flat]

        # 更新缓存：只更新过了FFN的窗口
        new_entropy = compute_attention_entropy_from_features(x_ffn_input)
        attn_entropy_cache[inc_topk_idx_flat] = new_entropy

        return x_windows_processed

    def _create_attn_mask(self, H_pad, W_pad):
        """Create attention mask for SW-MSA"""
        img_mask = torch.zeros((1, H_pad, W_pad, 1), device=torch.cuda.current_device())
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partition(img_mask)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def window_reverse(self, windows, H, W):
        """Reverse window partition"""
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, window_size, W // window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """Window partition"""
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


@MODELS.register_module()
class SwinTransformerEntropy(SwinTransformer):
    """Swin Transformer with Entropy-based Window Pruning

    在指定的stage和block中使用熵进行窗口级别的剪枝:

    策略说明:
    - 'origin': 不做任何剪枝，和原始Swin Transformer一样
    - 'kl': 基于KL散度的单次筛选，保留高分窗口过Attn+FFN
    - 'kl_incremental': 基于KL散度的增量筛选，KL预筛选 + 增量筛选决定谁过FFN
    - 'incremental_only': 直接对完整特征做增量筛选（无KL预筛选）

    配置示例:
        # kl_incremental 策略
        entropy_pruning=dict(
            enabled=True,
            entropy_strategy='kl_incremental',
            stages_to_prune=[2],
            block_indices=[0, 2, 4],
            kl_ratio={2: [0.8, 0.6, 0.4]},   # KL筛选保留比例
            increment_ratio={2: [0.8, 0.6]},   # 增量筛选保留比例
        )

        # incremental_only 策略（只需increment_ratio）
        entropy_pruning=dict(
            enabled=True,
            entropy_strategy='incremental_only',
            stages_to_prune=[2],
            block_indices=[0, 2, 4],
            increment_ratio={2: [0.5, 0.5, 0.5]},  # 增量筛选保留比例
        )

    注意:
        - 剪枝只在shift_size=0的block上生效（对应W-MSA，不是SW-MSA）
        - Swin的block交替使用W-MSA和SW-MSA，所以只有一半的block会被剪枝
    """

    # 支持的策略列表
    # - 'origin': 不做剪枝，和原始Swin一样
    # - 'kl': 基于KL散度筛选，高分窗口过Attn+FFN
    # - 'kl_incremental': KL预筛选 + 增量筛选决定谁过FFN
    # - 'incremental_only': 直接对完整特征做增量筛选（无KL预筛选）
    VALID_STRATEGIES = ['origin', 'kl', 'kl_incremental', 'incremental_only']

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None,
                 entropy_pruning: Optional[Dict] = None):

        # ========================================
        # 解析并验证熵剪枝配置
        # ========================================
        self.entropy_pruning_cfg = entropy_pruning if entropy_pruning else {}
        self.entropy_enabled = self.entropy_pruning_cfg.get('enabled', True)
        self.entropy_stages = self.entropy_pruning_cfg.get('stages_to_prune', [2])
        self.prune_block_indices = self.entropy_pruning_cfg.get('block_indices', [0, 2, 4])

        # 验证策略名称
        self.entropy_strategy = self.entropy_pruning_cfg.get('entropy_strategy', 'kl')
        if self.entropy_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid entropy_strategy '{self.entropy_strategy}'. "
                           f"Must be one of {self.VALID_STRATEGIES}")

        # 根据策略加载对应配置
        self._parse_strategy_config()

        # 调试打印
        print(f"[EntropyPruning] strategy={self.entropy_strategy}, enabled={self.entropy_enabled}")
        print(f"[EntropyPruning] stages={self.entropy_stages}, block_indices={self.prune_block_indices}")
        print(f"[EntropyPruning] kl_ratio={self.kl_ratio_cfg}, increment_ratio={self.increment_ratio_cfg}")

        self._depths = depths

        # 调用父类初始化
        super(SwinTransformerEntropy, self).__init__(
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
            init_cfg=init_cfg)

        if self.entropy_enabled:
            self._replace_blocks_for_entropy()

    def _parse_strategy_config(self):
        """统一解析策略相关配置"""
        if self.entropy_strategy == 'kl_incremental':
            # kl_incremental: 需要 kl_ratio 和 increment_ratio
            self.kl_ratio_cfg = self.entropy_pruning_cfg.get('kl_ratio', {
                2: [0.7, 0.5, 0.3]
            })
            self.increment_ratio_cfg = self.entropy_pruning_cfg.get('increment_ratio', {
                2: [0.8, 0.8]
            })
        elif self.entropy_strategy == 'kl':
            # kl: 只需要 kl_ratio
            self.kl_ratio_cfg = self.entropy_pruning_cfg.get('kl_ratio', {
                2: [0.7, 0.5, 0.3]
            })
            self.increment_ratio_cfg = None
        elif self.entropy_strategy == 'incremental_only':
            # incremental_only: 只需要 increment_ratio，不需要 kl_ratio
            # 注意：increment_ratio 长度应该等于 len(block_indices) - 1
            # 因为第一个 block（block 0）没有前一个缓存
            self.kl_ratio_cfg = None
            self.increment_ratio_cfg = self.entropy_pruning_cfg.get('increment_ratio', {
                2: [0.5, 0.5]  # 对应 block 2 和 block 4
            })
        else:
            # origin: 不需要额外配置
            self.kl_ratio_cfg = None
            self.increment_ratio_cfg = None

    def _replace_blocks_for_entropy(self):
        """替换需要剪枝的block为SwinBlockEntropy"""
        default_act_cfg = dict(type='GELU')
        default_norm_cfg = dict(type='LN')

        for stage_idx in self.entropy_stages:
            if stage_idx >= len(self.stages):
                continue

            stage = self.stages[stage_idx]

            # 获取当前stage的配置
            stage_kl_ratios = self.kl_ratio_cfg.get(stage_idx, []) if self.kl_ratio_cfg else []
            stage_inc_ratios = self.increment_ratio_cfg.get(stage_idx, []) if self.increment_ratio_cfg else []

            for block_idx, block in enumerate(stage.blocks):
                if block_idx not in self.prune_block_indices:
                    continue

                # 获取drop_path_prob
                drop_path_prob = 0.
                if hasattr(block.attn, 'dropout_layer') and block.attn.dropout_layer is not None:
                    if hasattr(block.attn.dropout_layer, 'drop_prob'):
                        drop_path_prob = block.attn.dropout_layer.drop_prob

                # 获取ffn的hidden_dim
                ffn_hidden_dim = block.ffn.layers[1].out_features
                embed_dims = block.attn.w_msa.embed_dims
                window_size = block.attn.window_size

                # 创建新的SwinBlockEntropy
                new_block = SwinBlockEntropy(
                    embed_dims=embed_dims,
                    num_heads=block.attn.w_msa.num_heads,
                    feedforward_channels=ffn_hidden_dim,
                    window_size=window_size,
                    shift=False if block_idx % 2 == 0 else True,
                    qkv_bias=True,
                    qk_scale=block.attn.w_msa.scale,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    drop_path_rate=drop_path_prob,
                    act_cfg=default_act_cfg,
                    norm_cfg=default_norm_cfg,
                    with_cp=False,
                    init_cfg=None
                )

                # 复制重要参数
                new_block.norm1 = block.norm1
                new_block.norm2 = block.norm2
                new_block.attn.w_msa = block.attn.w_msa
                new_block.ffn = block.ffn

                # 复制shift_size和window_size
                new_block.shift_size = block.attn.shift_size
                new_block.window_size = window_size

                # 设置熵剪枝配置
                # kl_ratio 的索引：基于 block 在 prune_block_indices 中的位置
                # prune_block_indices=[0, 2, 4] -> 位置 0, 1, 2
                # 所以 block_idx=0 -> 位置 0, block_idx=2 -> 位置 1, block_idx=4 -> 位置 2
                kl_ratio_idx_in_prune = self.prune_block_indices.index(block_idx) if block_idx in self.prune_block_indices else -1
                kl_ratio = stage_kl_ratios[kl_ratio_idx_in_prune] if (kl_ratio_idx_in_prune >= 0 and kl_ratio_idx_in_prune < len(stage_kl_ratios)) else None

                # increment_ratio 的索引：基于 W-MSA block 在 prune_block_indices 中的顺序（第1个W-MSA无缓存）
                # prune_block_indices=[0, 2, 4] 都是 W-MSA blocks (偶数 index)
                # 位置 0: block 0 -> 无前一个缓存，inc_ratio = None
                # 位置 1: block 2 -> inc_ratio[0]
                # 位置 2: block 4 -> inc_ratio[1]
                inc_ratio_idx = kl_ratio_idx_in_prune - 1 if kl_ratio_idx_in_prune > 0 else -1
                inc_ratio = stage_inc_ratios[inc_ratio_idx] if (inc_ratio_idx >= 0 and inc_ratio_idx < len(stage_inc_ratios)) else None

                new_block.set_entropy_config(
                    strategy=self.entropy_strategy,
                    kl_ratio=kl_ratio,
                    increment_ratio=inc_ratio
                )
                # 调试打印
                print(f"[EntropyPruning] stage={stage_idx}, block={block_idx}, strategy={self.entropy_strategy}, kl_ratio={kl_ratio}, inc_ratio={inc_ratio}")

                # 替换
                stage.blocks[block_idx] = new_block

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []

        for i, stage in enumerate(self.stages):
            if self.entropy_enabled and i in self.entropy_stages:
                x, hw_shape, out, out_hw_shape = self._forward_with_entropy(stage, x, hw_shape, i)
            else:
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)

                out_reshaped = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out_reshaped)

        return outs

    def _forward_with_entropy(self, stage, x, hw_shape, stage_idx):
        """Forward stage with entropy-based window pruning"""
        depth = self._depths[stage_idx]
        stage_blocks = stage.blocks

        # 获取当前stage的配置
        stage_kl_ratios = self.kl_ratio_cfg.get(stage_idx, []) if self.kl_ratio_cfg else []
        stage_inc_ratios = self.increment_ratio_cfg.get(stage_idx, []) if self.increment_ratio_cfg else []

        attn_entropy_cache = None

        for block_idx in range(depth):
            block = stage_blocks[block_idx]

            if block_idx in self.prune_block_indices:
                # 更新block的熵配置（确保配置一致）
                # kl_ratio 的索引：基于 block 在 prune_block_indices 中的位置
                kl_ratio_idx_in_prune = self.prune_block_indices.index(block_idx) if block_idx in self.prune_block_indices else -1
                kl_ratio = stage_kl_ratios[kl_ratio_idx_in_prune] if (kl_ratio_idx_in_prune >= 0 and kl_ratio_idx_in_prune < len(stage_kl_ratios)) else None
                # increment_ratio 索引：基于 W-MSA block 在 prune_block_indices 中的顺序
                inc_ratio_idx = kl_ratio_idx_in_prune - 1 if kl_ratio_idx_in_prune > 0 else -1
                inc_ratio = stage_inc_ratios[inc_ratio_idx] if (inc_ratio_idx >= 0 and inc_ratio_idx < len(stage_inc_ratios)) else None

                block.set_entropy_config(
                    strategy=self.entropy_strategy,
                    kl_ratio=kl_ratio,
                    increment_ratio=inc_ratio
                )

                x, attn_entropy_cache = block(x, hw_shape, attn_entropy_cache=attn_entropy_cache)
            else:
                x = block(x, hw_shape)

        if stage.downsample is not None:
            x_down, down_hw_shape = stage.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

    def get_entropy_config(self) -> Dict:
        """Get current entropy pruning configuration"""
        return {
            'enabled': self.entropy_enabled,
            'entropy_strategy': self.entropy_strategy,
            'stages': self.entropy_stages,
            'block_indices': self.prune_block_indices,
            'kl_ratio': self.kl_ratio_cfg,
            'increment_ratio': self.increment_ratio_cfg,
        }

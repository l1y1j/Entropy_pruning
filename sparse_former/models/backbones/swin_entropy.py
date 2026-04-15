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
    
    在Block级别做完整的Gather-Scatter:
    1. Window partition + padding + cyclic shift
    2. 计算熵并Top-K筛选
    3. Gather高分窗口
    4. Attn + FFN（在小尺寸上）
    5. Scatter拼回全图
    6. Window reverse + unshift + unpad
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_entropy = False
        self.block_keep_ratio = None
        self.kl_ratio = None
        self.increment_ratio = None
        self.entropy_strategy = 'kl'
        
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        if drop_path_rate > 0:
            from mmdet.models.utils import build_dropout
            self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        else:
            self.drop_path = None
    
    def forward(self, x, hw_shape, attn_entropy_cache=None):
        """Forward with KL incremental pruning
        
        Supports two strategies:
        - 'kl': Original KL-only pruning
        - 'kl_incremental': Double-filter pruning with attention entropy cache
        
        Args:
            x: (B, L, C) input features
            hw_shape: (H, W) spatial shape
            attn_entropy_cache: (B * N_win,) cache from previous block
        
        Returns:
            x: output features
            attn_entropy_cache: updated cache for next block
        """
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        
        self._B = B
        self._H = H
        self._W = W
        self._C = C
        
        x = x.view(B, H, W, C)
        
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]
        
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._create_attn_mask(H_pad, W_pad)
        else:
            shifted_x = x
            attn_mask = None
        
        x_windows = self.window_partition(shifted_x)
        
        total_windows = x_windows.shape[0]
        N_win = total_windows // B
        
        global_window_idx = torch.arange(total_windows, device=x.device)
        batch_offsets = (torch.arange(B, device=x.device) * N_win).unsqueeze(1)
        
        kl_keep_mask = None
        kl_keep_flat = None
        inc_ratio = None
        
        if self.enable_entropy and self.shift_size == 0:
            kl_ratio = self.kl_ratio if self.kl_ratio is not None else self.block_keep_ratio
            
            if kl_ratio is not None and kl_ratio > 0:
                window_scores = compute_window_relative_entropy(x_windows, B, self.window_size)
                window_scores = window_scores.view(B, -1)
                
                k = max(1, int(N_win * kl_ratio))
                _, keep_idx = torch.topk(window_scores, k=k, dim=1)
                keep_idx = keep_idx.sort(dim=1)[0]
                
                kl_keep_flat = (keep_idx + batch_offsets).view(-1)
                
                kl_keep_mask = torch.zeros(total_windows, dtype=torch.bool, device=x.device)
                kl_keep_mask[kl_keep_flat] = True
            
            if self.entropy_strategy == 'kl_incremental' and self.increment_ratio is not None:
                inc_ratio = self.increment_ratio
        
        x_to_attn = x_windows
        mask_to_attn = attn_mask
        
        if kl_keep_mask is not None:
            x_to_attn = x_windows[kl_keep_mask]
            mask_to_attn = attn_mask[kl_keep_mask] if attn_mask is not None else None
        
        x_to_attn = x_to_attn.view(-1, self.window_size * self.window_size, C)
        
        identity_attn = x_to_attn
        x_after_attn = self.norm1(x_to_attn)
        x_after_attn = self.attn.w_msa(x_after_attn, mask=mask_to_attn)
        
        if self.drop_path is not None:
            x_after_attn = identity_attn + self.drop_path(x_after_attn)
        else:
            x_after_attn = identity_attn + x_after_attn
        
        cur_entropy = compute_attention_entropy_from_features(x_after_attn)
        
        if attn_entropy_cache is None:
            attn_entropy_cache = torch.zeros(total_windows, device=x.device)
        
        if kl_keep_mask is not None and self.entropy_strategy == 'kl_incremental' and self.increment_ratio is not None:
            # 使用 kl_keep_flat 索引更直接
            cached_entropy = attn_entropy_cache[kl_keep_flat]
            inc_scores = torch.abs(cur_entropy - cached_entropy)
            
            # 按Batch解耦计算Top-K，防止串图
            num_kept_total = inc_scores.shape[0]
            k_kl = num_kept_total // B
            inc_scores_batch = inc_scores.view(B, k_kl)
            
            inc_k = max(1, int(k_kl * inc_ratio))
            
            _, inc_topk_idx_local = torch.topk(inc_scores_batch, k=inc_k, dim=1)
            inc_topk_idx_local = inc_topk_idx_local.sort(dim=1)[0]
            
            inc_batch_offsets = (torch.arange(B, device=x.device) * k_kl).unsqueeze(1)
            inc_topk_idx_flat = (inc_topk_idx_local + inc_batch_offsets).view(-1)
            
            # 从attention结果中提取精英进入FFN
            x_ffn_input = x_after_attn[inc_topk_idx_flat]
            identity_ffn = x_ffn_input
            x_ffn_input = self.norm2(x_ffn_input)
            x_ffn_input = self.ffn(x_ffn_input)
            
            if self.drop_path is not None:
                x_ffn_input = identity_ffn + self.drop_path(x_ffn_input)
            else:
                x_ffn_input = identity_ffn + x_ffn_input
            
            new_entropy = compute_attention_entropy_from_features(x_ffn_input)
            
            # 更新缓存：kl_keep_flat是所有过KL关的绝对索引，提取其中过FFN的那些
            final_keep_absolute_idx = kl_keep_flat[inc_topk_idx_flat]
            attn_entropy_cache[final_keep_absolute_idx] = new_entropy
            
            # 双重Scatter回填
            x_windows_processed = x_windows.clone()
            
            attn_output_windows = x_after_attn.clone().view(-1, self.window_size, self.window_size, C)
            ffn_output_windows = x_ffn_input.view(-1, self.window_size, self.window_size, C)
            
            # 退栈1: FFN结果 -> Attn输出
            attn_output_windows[inc_topk_idx_flat] = ffn_output_windows
            # 退栈2: Attn输出 -> 全图
            x_windows_processed[kl_keep_flat] = attn_output_windows
        else:
            identity_ffn = x_after_attn
            x_after_ffn = self.norm2(x_after_attn)
            x_after_ffn = self.ffn(x_after_ffn)
            
            if self.drop_path is not None:
                x_after_ffn = identity_ffn + self.drop_path(x_after_ffn)
            else:
                x_after_ffn = identity_ffn + x_after_ffn
            
            if kl_keep_mask is not None:
                attn_entropy_cache[kl_keep_mask] = compute_attention_entropy_from_features(x_after_ffn)
            
            x_windows_new = x_windows.clone()
            x_high_reshaped = x_after_ffn.view(-1, self.window_size, self.window_size, C)
            
            if kl_keep_flat is not None:
                x_windows_new[kl_keep_flat] = x_high_reshaped
            else:
                x_windows_new = x_windows_new
            
            x_windows_processed = x_windows_new
        
        attn_windows = x_windows_processed.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        
        return x, attn_entropy_cache
    
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
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
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
    
    在指定的stage和block中使用相对熵进行窗口级别的剪枝:
    1. 在Block级别做Gather-Scatter
    2. 只对高分窗口执行Attn和FFN
    3. 低分窗口保持原始identity
    
    Args:
        entropy_pruning (dict, optional): Configuration for entropy pruning
            - enabled (bool): Whether to enable pruning (default: True)
            - stages_to_prune (list): Stages to apply pruning (default: [2])
            - block_indices (list): Block indices to prune (default: [0, 2, 4])
            - block_keep_ratio (dict): Keep ratio for each stage and block
                e.g., {2: [0.7, 0.5, 0.3]}
    """

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
        
        self.entropy_pruning_cfg = entropy_pruning if entropy_pruning else {}
        self.entropy_enabled = self.entropy_pruning_cfg.get('enabled', True)
        self.entropy_stages = self.entropy_pruning_cfg.get('stages_to_prune', [2])
        self.prune_block_indices = self.entropy_pruning_cfg.get('block_indices', [0, 2, 4])
        self.entropy_strategy = self.entropy_pruning_cfg.get('entropy_strategy', 'kl')
        
        if self.entropy_strategy == 'kl_incremental':
            self.kl_ratio_cfg = self.entropy_pruning_cfg.get('kl_ratio', {
                2: [0.7, 0.5, 0.3]
            })
            self.increment_ratio_cfg = self.entropy_pruning_cfg.get('increment_ratio', {
                2: [0.8, 0.8]
            })
        else:
            self.block_keep_ratio = self.entropy_pruning_cfg.get('block_keep_ratio', {
                2: [0.7, 0.5, 0.3]
            })
        
        self._depths = depths
        
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

    def _replace_blocks_for_entropy(self):
        """替换需要剪枝的block为SwinBlockEntropy"""
        default_act_cfg = dict(type='GELU')
        default_norm_cfg = dict(type='LN')
        
        for stage_idx in self.entropy_stages:
            if stage_idx >= len(self.stages):
                continue
            
            stage = self.stages[stage_idx]
            
            if self.entropy_strategy == 'kl_incremental':
                kl_ratios = self.kl_ratio_cfg.get(stage_idx, [0.7, 0.5, 0.3])
                inc_ratios = self.increment_ratio_cfg.get(stage_idx, [0.8, 0.8])
                ratios = kl_ratios
            else:
                ratios = self.block_keep_ratio.get(stage_idx, [0.7])
            
            for block_idx, block in enumerate(stage.blocks):
                if block_idx in self.prune_block_indices:
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
                    
                    # 设置entropy参数
                    new_block.enable_entropy = True
                    new_block.entropy_strategy = self.entropy_strategy
                    
                    if self.entropy_strategy == 'kl_incremental':
                        new_block.kl_ratio = ratios[block_idx] if block_idx < len(ratios) else ratios[-1]
                        if block_idx in [2, 4]:
                            new_block.increment_ratio = inc_ratios[block_idx - 2] if (block_idx - 2) < len(inc_ratios) else inc_ratios[-1]
                        else:
                            new_block.increment_ratio = None
                    else:
                        new_block.block_keep_ratio = ratios[block_idx] if block_idx < len(ratios) else ratios[-1]
                        new_block.kl_ratio = None
                        new_block.increment_ratio = None
                    
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
        
        if self.entropy_strategy == 'kl_incremental':
            kl_ratios = self.kl_ratio_cfg.get(stage_idx, [0.7, 0.5, 0.3])
            inc_ratios = self.increment_ratio_cfg.get(stage_idx, [0.8, 0.8])
        else:
            ratios = self.block_keep_ratio.get(stage_idx, [0.7])
        
        attn_entropy_cache = None
        
        for block_idx in range(depth):
            block = stage_blocks[block_idx]
            
            if block_idx in self.prune_block_indices:
                if self.entropy_strategy == 'kl_incremental':
                    block.kl_ratio = kl_ratios[block_idx] if block_idx < len(kl_ratios) else kl_ratios[-1]
                    if block_idx in [2, 4]:
                        block.increment_ratio = inc_ratios[block_idx - 2] if (block_idx - 2) < len(inc_ratios) else inc_ratios[-1]
                    else:
                        block.increment_ratio = None
                    x, attn_entropy_cache = block(x, hw_shape, attn_entropy_cache=attn_entropy_cache)
                else:
                    block.block_keep_ratio = ratios[block_idx] if block_idx < len(ratios) else ratios[-1]
                    x, _ = block(x, hw_shape)
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
            'stages': self.entropy_stages,
            'block_indices': self.prune_block_indices,
            'block_keep_ratio': self.block_keep_ratio
        }
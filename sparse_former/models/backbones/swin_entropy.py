import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.models.backbones.swin import SwinTransformer
import numpy as np
from PIL import Image

from sparse_former.utils.block_entropy import (
    compute_block_importance_score,
    compute_softmax_topk_score,
    compute_local_relative_entropy,
    normalize_heatmap,
    resize_heatmap,
    visualize_scores_3panel
)


@MODELS.register_module()
class SwinTransformerEntropy(SwinTransformer):
    """Swin Transformer with Entropy-based Pruning Support
    
    This is an extension of SwinTransformer that adds:
    1. Entropy computation for each stage
    2. Configurable pruning strategies (kl, variance, softmax_topk)
    3. Visualization of entropy/importance maps
    4. Optional actual pruning with soft masks
    
    Args:
        entropy_pruning (dict, optional): Configuration for entropy pruning
            - enabled (bool): Whether to enable actual pruning (default: False)
            - strategy (str): 'kl', 'variance', or 'softmax_topk' (default: 'kl')
            - block_size (int): Block size for KL/variance (default: 2)
            - keep_ratio (float): Ratio of tokens to keep (default: 0.7)
            - stages_to_prune (list): Which stages to compute entropy for (default: [2, 3])
            - vis_interval (int): Visualization interval in epochs (default: 5)
            - vis_enabled (bool): Enable visualization (default: True)
        (Other args same as SwinTransformer)
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
        self.entropy_enabled = self.entropy_pruning_cfg.get('enabled', False)
        self.entropy_strategy = self.entropy_pruning_cfg.get('strategy', 'kl')
        self.entropy_block_size = self.entropy_pruning_cfg.get('block_size', 2)
        self.entropy_keep_ratio = self.entropy_pruning_cfg.get('keep_ratio', 0.7)
        self.entropy_stages = self.entropy_pruning_cfg.get('stages_to_prune', [2, 3])
        self.entropy_vis_interval = self.entropy_pruning_cfg.get('vis_interval', 5)
        self.entropy_vis_enabled = self.entropy_pruning_cfg.get('vis_enabled', True)
        
        self._entropy_scores: Dict[int, torch.Tensor] = {}
        self._current_epoch = 0
        self._vis_count = 0
        
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

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                
                out_reshaped = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out_reshaped)
                
                if i in self.entropy_stages:
                    self._compute_entropy_for_stage(out_reshaped, i)
        
        return outs
    
    def _compute_entropy_for_stage(self, features: torch.Tensor, stage_idx: int):
        """Compute entropy scores for a specific stage"""
        if features.dim() == 4:
            B, C, H, W = features.shape
            features_nhwc = features.permute(0, 2, 3, 1)
        else:
            return
        
        if self.entropy_strategy == 'kl':
            scores = compute_block_importance_score(
                features_nhwc, 
                block_size=self.entropy_block_size, 
                mode='kl'
            )
        elif self.entropy_strategy == 'variance':
            scores = compute_block_importance_score(
                features_nhwc,
                block_size=self.entropy_block_size,
                mode='variance'
            )
        elif self.entropy_strategy == 'softmax_topk':
            scores = compute_softmax_topk_score(
                features_nhwc,
                top_k_ratio=self.entropy_keep_ratio
            )
        else:
            scores = None
            
        if scores is not None:
            self._entropy_scores[stage_idx] = scores
    
    def compute_prune_mask(self, features: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """Compute pruning mask based on entropy scores
        
        Args:
            features: (B, C, H, W) - feature tensor
            stage_idx: stage index
            
        Returns:
            mask: (B, 1, H, W) - soft mask tensor
        """
        if features.dim() == 4:
            B, C, H, W = features.shape
            features_nhwc = features.permute(0, 2, 3, 1)
        else:
            return torch.ones(1, 1, 1, 1, device=features.device)
        
        if self.entropy_strategy == 'kl':
            scores = compute_block_importance_score(
                features_nhwc,
                block_size=self.entropy_block_size,
                mode='kl'
            )
        elif self.entropy_strategy == 'variance':
            scores = compute_block_importance_score(
                features_nhwc,
                block_size=self.entropy_block_size,
                mode='variance'
            )
        elif self.entropy_strategy == 'softmax_topk':
            scores = compute_softmax_topk_score(
                features_nhwc,
                top_k_ratio=self.entropy_keep_ratio
            )
        else:
            return torch.ones_like(features[:, :1, :, :])
        
        if scores is None:
            return torch.ones_like(features[:, :1, :, :])
        
        H_score, W_score = scores.shape
        H_feat, W_feat = H, W
        
        scores_resized = F.interpolate(
            scores.unsqueeze(0).unsqueeze(0).float(),
            size=(H_feat, W_feat),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        k = int(self.entropy_keep_ratio * scores_resized.numel())
        k = max(1, k)
        
        threshold, _ = torch.kthvalue(scores_resized.flatten(), scores_resized.numel() - k)
        
        mask = (scores_resized >= threshold).float()
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, H_feat, W_feat)
        
        return mask
    
    def get_entropy_scores(self) -> Dict[int, torch.Tensor]:
        """Get the computed entropy scores for all stages
        
        Returns:
            dict: stage_idx -> score_map (H, W)
        """
        return self._entropy_scores.copy()
    
    def clear_entropy_scores(self):
        """Clear stored entropy scores"""
        self._entropy_scores.clear()
    
    def set_epoch(self, epoch: int):
        """Set current training epoch for visualization control"""
        self._current_epoch = epoch
    
    def should_visualize(self) -> bool:
        """Check if visualization should be performed this epoch"""
        if not self.entropy_vis_enabled:
            return False
        return (self._current_epoch + 1) % self.entropy_vis_interval == 0
    
    def visualize_entropy(
        self,
        image: Optional[torch.Tensor] = None,
        save_dir: str = './entropy_vis',
        stage_indices: Optional[List[int]] = None
    ):
        """Visualize entropy scores
        
        Args:
            image: (C, H, W) or (H, W, C) input image for overlay
            save_dir: directory to save visualizations
            stage_indices: which stages to visualize (default: self.entropy_stages)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if stage_indices is None:
            stage_indices = self.entropy_stages
        
        for stage_idx in stage_indices:
            if stage_idx in self._entropy_scores:
                score_map = self._entropy_scores[stage_idx]
                save_path = os.path.join(
                    save_dir, 
                    f'epoch_{self._current_epoch}_stage_{stage_idx}_entropy_{self.entropy_strategy}.png'
                )
                
                img_pil = None
                if image is not None:
                    if image.dim() == 3:
                        if image.shape[0] == 3:
                            img_pil = Image.fromarray(
                                image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                            )
                        else:
                            img_pil = Image.fromarray(
                                image.cpu().numpy().astype(np.uint8)
                            )
                
                visualize_scores_3panel(
                    score_map=score_map,
                    image=img_pil,
                    stage_idx=stage_idx,
                    save_path=save_path,
                    original_size=tuple(score_map.shape[::-1])
                )
                print(f"Saved entropy visualization to {save_path}")
    
    def get_entropy_config(self) -> Dict:
        """Get current entropy pruning configuration"""
        return {
            'enabled': self.entropy_enabled,
            'strategy': self.entropy_strategy,
            'block_size': self.entropy_block_size,
            'keep_ratio': self.entropy_keep_ratio,
            'stages': self.entropy_stages,
            'vis_interval': self.entropy_vis_interval,
            'vis_enabled': self.entropy_vis_enabled
        }
    
    def update_entropy_config(self, **kwargs):
        """Update entropy pruning configuration"""
        for key, value in kwargs.items():
            attr_name = f'entropy_{key}'
            if hasattr(self, attr_name):
                setattr(self, attr_name, value)
        
        if 'strategy' in kwargs:
            self.entropy_strategy = kwargs['strategy']
        if 'block_size' in kwargs:
            self.entropy_block_size = kwargs['block_size']
        if 'keep_ratio' in kwargs:
            self.entropy_keep_ratio = kwargs['keep_ratio']
        if 'stages_to_prune' in kwargs:
            self.entropy_stages = kwargs['stages_to_prune']
        if 'vis_interval' in kwargs:
            self.entropy_vis_interval = kwargs['vis_interval']
        if 'vis_enabled' in kwargs:
            self.entropy_vis_enabled = kwargs['vis_enabled']
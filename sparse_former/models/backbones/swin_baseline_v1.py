import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmdet.registry import MODELS
from mmdet.models.backbones.swin import (
    SwinTransformer, SwinBlock, SwinBlockSequence, 
    ShiftWindowMSA, WindowMSA
)


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
    kl = (local_dist * torch.log(local_dist / (global_dist + 1e-8))).sum(dim=-1)
    
    return kl.view(-1)


class SwinBlockV1(nn.Module):
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
    
    def forward(self, x: torch.Tensor, hw_shape: tuple, entropy_cache: torch.Tensor = None) -> tuple:
        """Forward function"""
        can_prune_kl = self.kl_ratio is not None and self.kl_ratio < 1.0
        can_prune_inc = self.inc_ratio is not None and self.inc_ratio < 1.0
        
        if self.shift_size > 0:
            return self._forward_base(x, hw_shape), None
        
        if not can_prune_kl and not can_prune_inc:
            return self._forward_base(x, hw_shape), None
        
        if self.strategy is not None:
            if self.strategy == 'kl_inc' and can_prune_kl and can_prune_inc:
                return self._forward_kl_inc(x, hw_shape, entropy_cache)
            elif self.strategy in ['kl', 'kl_inc'] and can_prune_kl:
                return self._forward_kl(x, hw_shape)
            elif self.strategy in ['inc', 'kl_inc'] and can_prune_inc:
                return self._forward_inc(x, hw_shape, entropy_cache)
        
        return self._forward_base(x, hw_shape), None
    
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
            return self._forward_base_with_pad(x, hw_shape, pad_r, pad_b), None
        
        window_scores = compute_window_relative_entropy(x_windows, B, self.window_size)
        window_scores = window_scores.view(B, -1)
        
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
        
        return x, full_entropy

    def _forward_inc(self, x: torch.Tensor, hw_shape: tuple, entropy_cache: torch.Tensor = None) -> tuple:
        """Incremental pruning forward with cross-layer comparison"""
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
            return self._forward_base_with_pad(x, hw_shape, pad_r, pad_b), None
        
        x_windows_flat = x_windows.view(-1, self.window_size * self.window_size, C)
        identity_attn = x_windows_flat
        x_after_attn = self.norm1(x_windows_flat)
        x_after_attn = self.attn.w_msa(x_after_attn)
        x_after_attn = identity_attn + x_after_attn
        
        cur_entropy = self._compute_entropy(x_after_attn)
        
        if entropy_cache is None or entropy_cache.shape[0] != total_windows:
            inc_scores = torch.zeros_like(cur_entropy)
        else:
            inc_scores = torch.abs(cur_entropy - entropy_cache.detach())
        inc_scores = inc_scores.view(B, -1)
        
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
        
        return x, cur_entropy.detach()

    def _forward_kl_inc(self, x: torch.Tensor, hw_shape: tuple, entropy_cache: torch.Tensor = None) -> tuple:
        """KL + INC 串联筛选 with cross-layer comparison"""
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
            return self._forward_base_with_pad(x, hw_shape, pad_r, pad_b), None
        
        window_scores = compute_window_relative_entropy(x_windows, B, self.window_size)
        window_scores = window_scores.view(B, -1)
        
        k_kl = max(1, int(N_win * self.kl_ratio))
        _, kl_keep_idx = torch.topk(window_scores, k=k_kl, dim=1)
        kl_keep_idx = kl_keep_idx.sort(dim=1)[0]
        
        kl_keep_idx_flat = (kl_keep_idx + batch_offsets).view(-1)
        
        x_kl = x_windows[kl_keep_idx_flat]
        x_kl = x_kl.view(-1, self.window_size * self.window_size, C)
        identity_attn = x_kl
        x_kl = self.norm1(x_kl)
        x_kl = self.attn.w_msa(x_kl)
        x_kl = identity_attn + x_kl
        
        cur_entropy = self._compute_entropy(x_kl)
        
        if entropy_cache is None or entropy_cache.shape[0] != total_windows:
            inc_scores = torch.zeros_like(cur_entropy)
        else:
            inc_scores = torch.abs(cur_entropy - entropy_cache[kl_keep_idx_flat].detach())
        inc_scores = inc_scores.view(B, -1)
        
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
        
        full_entropy = torch.zeros(total_windows, device=x.device)
        full_entropy[kl_keep_idx_flat] = cur_entropy.detach()
        
        return x, full_entropy

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


class SwinBlockSequenceV1(nn.Module):
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
            
            block = SwinBlockV1(
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
            )
            self.blocks.append(block)
        
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor, hw_shape: tuple):
        """Forward function
        
        Args:
            x: (B, L, C) input features
            hw_shape: (H, W) spatial shape
        
        Returns:
            x_down: output features after downsample
            down_hw_shape: new spatial shape
            x: output features before downsample
            hw_shape: original spatial shape
        """
        entropy_cache = None
        prev_entropy = None
        
        for i, block in enumerate(self.blocks):
            result = block(x, hw_shape, entropy_cache)
            if isinstance(result, tuple):
                x, new_entropy = result
                if block.shift_size == 0:
                    prev_entropy = new_entropy
                entropy_cache = prev_entropy
        
        if self.downsample is not None:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@MODELS.register_module()
class SwinTransformerV1(SwinTransformer):
    """Swin Transformer V1 with optional KL pruning
    
    Simplified implementation with two strategies:
    - 'base': No pruning, same as original Swin
    - 'kl': KL-based window pruning
    
    Config:
        backbone=dict(
            type='SwinTransformerV1',
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
    ):
        self.strategy = strategy
        self.stage_config = stage_config or {}
        self.inc_stage_config = inc_stage_config or {}
        
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
            new_stage = SwinBlockSequenceV1(
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
            
            new_stage = SwinBlockSequenceV1(
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
            
            new_stage = SwinBlockSequenceV1(
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
"""SparseNetEntropy — SparseNet with Entropy-driven Information Scoring.

Extends SparseNet.  Replaces BlockSequence → EntropyBlockSequence in all stages.
The Gather → BasicBlock Conv → Scatter execution framework is unchanged.
"""

import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.logging import MMLogger
from mmengine.utils import to_2tuple
from mmdet.models import PatchEmbed, PatchMerging

from sparse_former.registry import MODELS
from .threshold_predictor import InformationThresholdPredictor
from .entropy_block import EntropyBlockSequence
from .entropy_utils import compute_erb_from_conv


@MODELS.register_module()
class SparseNetEntropy(BaseModule):
    """SparseNet with Information-driven Adaptive Window Scoring.

    Compared to SparseNet:
        score_net (Linear)    →  InformationScoreNet (KL + ThresholdPredictor)
        Top-K (fixed ratio)   →  soft_mask-based selection (per-image adaptive)
        No gate loss          →  Gate loss for end-to-end learnable pruning rate

    Unchanged from SparseNet:
        - Residual feature extraction
        - GlobalBlock (global context)
        - LocalBlock (Gather → BasicBlock Conv → Scatter)
        - PatchMerging downsampling
        - PatchEmbed stem
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 layers=(3, 4, 6, 3),
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 top_k=(0.7, 0.6, 0.5, 0.5),
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
                 # ===== Entropy Pruning parameters =====
                 use_learnable_gate=False,
                 temperature=1.0,
                 lambda_kl=0.1,
                 lambda_inc=0.1,
                 strategy='kl_inc',
                 **kwargs):
        # ---- Boilerplate: pretrained / init_cfg handling (from SparseNet) ----
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super().__init__(init_cfg=self.init_cfg)

        # ---- Entropy parameters ----
        self.strategy = strategy
        self.use_learnable_gate = use_learnable_gate
        self.temperature = temperature
        self.lambda_kl = lambda_kl
        self.lambda_inc = lambda_inc

        # Shared ThresholdPredictors across all stages
        if use_learnable_gate:
            self.kl_predictor = InformationThresholdPredictor()
            self.inc_predictor = InformationThresholdPredictor()
        else:
            self.kl_predictor = None
            self.inc_predictor = None

        # ---- Stem (from SparseNet) ----
        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # Stochastic depth decay
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        # ---- Build stages (EntropyBlockSequence instead of BlockSequence) ----
        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = EntropyBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                layers=layers[i],
                stage_id=i,
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                top_k=top_k[i],
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
                # Entropy params forwarded to EntropyBlockSequence
                temperature=temperature,
                lambda_kl=lambda_kl,
                lambda_inc=lambda_inc,
                kl_predictor=self.kl_predictor,
                inc_predictor=self.inc_predictor,
                use_learnable_gate=use_learnable_gate,
            )
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Output norm layers
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        # Gate loss dict (read by DINOWithGateLoss)
        self._gate_losses = {}

    # ------------------------------------------------------------------
    #  train / freeze / init_weights  (from SparseNet, adapted)
    # ------------------------------------------------------------------
    def train(self, mode=True):
        """Convert the model into training mode while keeping layers frozen."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                from mmengine.model.weight_init import trunc_normal_
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    from mmengine.model.weight_init import trunc_normal_init
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    from mmengine.model.weight_init import constant_init
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, \
                f'Only support `Pretrained` in `init_cfg`'
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            # Strip prefix
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # Handle absolute_pos_embed reshape
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L1, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L1 != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # Interpolate relative_position_bias_table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2), mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # Load with strict=False — new entropy params have no pretrained weights
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False
            )
            if missing_keys:
                logger.info(f'SparseNetEntropy: missing keys (expected for '
                            f'entropy modules): {len(missing_keys)} keys')
            if unexpected_keys:
                logger.warning(f'SparseNetEntropy: unexpected keys: '
                               f'{unexpected_keys}')

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """Forward pass with entropy scoring and gate loss collection.

        Returns:
            outs: list of feature maps [(B, C_i, H_i, W_i), ...]
        """
        # ---- C_b: image complexity from raw Conv output (pre-LayerNorm) ----
        x_conv = self.patch_embed.projection(x)
        _, _, H_patches, W_patches = x_conv.shape
        C_b = compute_erb_from_conv(x_conv, window_size=7)

        # Continue patch_embed: flatten + LayerNorm
        x = x_conv.flatten(2).transpose(1, 2)
        if self.patch_embed.norm is not None:
            x = self.patch_embed.norm(x)
        hw_shape = (H_patches, W_patches)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        # ---- Gate loss + keep ratio accumulation ----
        total_kl_gate_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        total_inc_gate_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        total_kl_keep_ratio = 0.0
        total_inc_keep_ratio = 0.0
        kl_count = 0
        inc_count = 0

        outs = []
        for i, stage in enumerate(self.stages):
            # Clear previous gate losses
            stage.gate_losses = []

            x, hw_shape, out, out_hw_shape = stage(x, hw_shape, C_b=C_b)

            # Collect: gate_losses = [(kl_loss, inc_loss, kl_keep, inc_keep), ...]
            if self.use_learnable_gate:
                for kl_g, inc_g, kl_r, inc_r in stage.gate_losses:
                    if kl_g is not None:
                        total_kl_gate_loss = total_kl_gate_loss + kl_g
                        total_kl_keep_ratio += kl_r if kl_r is not None else 0
                        kl_count += 1
                    if inc_g is not None:
                        total_inc_gate_loss = total_inc_gate_loss + inc_g
                        total_inc_keep_ratio += inc_r if inc_r is not None else 0
                        inc_count += 1

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(
                    -1, *out_hw_shape, self.num_features[i]
                ).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        # ---- Store gate losses + keep ratios for DINOWithGateLoss ----
        self._gate_losses = {}
        if self.use_learnable_gate:
            if kl_count > 0:
                self._gate_losses['loss_kl_gate'] = total_kl_gate_loss / kl_count
                self._gate_losses['kl_keep_ratio'] = torch.tensor(
                    total_kl_keep_ratio / kl_count, device=x.device
                )
            if inc_count > 0:
                self._gate_losses['loss_inc_gate'] = total_inc_gate_loss / inc_count
                self._gate_losses['inc_keep_ratio'] = torch.tensor(
                    total_inc_keep_ratio / inc_count, device=x.device
                )

        return outs

    def get_gate_losses(self):
        """Return accumulated gate losses for the current forward pass.

        Called by DINOWithGateLoss.loss() to add gate loss to the total loss.
        """
        return self._gate_losses

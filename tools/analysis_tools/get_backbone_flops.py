# Copyright (c) OpenMMLab. All rights reserved.
"""Backbone FLOPs calculation tool for Swin Transformer V1.

This tool calculates FLOPs for the backbone only (not the full detector),
supporting both origin mode (no pruning) and kl_inc mode (with entropy pruning).

Usage:
    # Origin mode FLOPs:
    python tools/analysis_tools/get_backbone_flops.py \
        configs/dino/dino-4scale_swin_v1_kl_8xb2-36e_panda.py

    # KL_Inc mode FLOPs (use --cfg-options to switch strategy):
    python tools/analysis_tools/get_backbone_flops.py \
        configs/dino/dino-4scale_swin_v1_kl_8xb2-36e_panda.py \
        --cfg-options backbone.strategy='kl_inc'
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
FILE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FILE_ROOT))

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmdet.registry import MODELS

# Import sparse_former modules to register models
from sparse_former.datasets import *  # noqa: F401, F403
from sparse_former.models import *  # noqa: F401, F403

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get backbone FLOPs (origin vs kl_inc)')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args


def calculate_backbone_pruned_flops(backbone, input_shape):
    """Calculate expected FLOPs reduction for kl_inc mode.

    This is an estimation based on the pruning ratios configured.
    For origin/base mode, returns None (use actual measurement).

    The estimation assumes:
    - Stage 2 contains 6 blocks and accounts for ~50% of backbone FLOPs
    - Other stages (0, 1, 3) have 2 blocks each, accounting for ~50% combined
    - Attention FLOPs ~40%, FFN FLOPs ~60%

    Args:
        backbone: Built backbone model with strategy, stage_config, inc_stage_config
        input_shape: (H, W) input shape (unused, for API compatibility)

    Returns:
        estimated_flops: Estimated FLOPs after pruning, or None for origin mode
    """
    strategy = backbone.strategy

    if strategy == 'base':
        return None  # Use actual measurement

    stage_config = backbone.stage_config
    inc_stage_config = backbone.inc_stage_config

    # Stage FLOPs weights (empirical)
    STAGE_WEIGHTS = {0: 0.1, 1: 0.2, 2: 0.5, 3: 0.2}
    ATTN_RATIO = 0.4
    FFN_RATIO = 0.6

    num_stages = len(backbone.stages)

    total_reduction = 0.0

    for stage_idx in range(num_stages):
        stage_weight = STAGE_WEIGHTS.get(stage_idx, 0.25)
        stage_depth = len(backbone.stages[stage_idx].blocks)

        # Get pruning config for this stage
        kl_cfg = stage_config.get(stage_idx, {})
        kl_blocks = kl_cfg.get('blocks', [])
        kl_ratios = kl_cfg.get('ratio', [])

        inc_cfg = inc_stage_config.get(stage_idx, {})
        inc_blocks = inc_cfg.get('blocks', [])
        inc_ratios = inc_cfg.get('inc_ratio', [])

        # Convert ratios to list if needed
        if isinstance(kl_ratios, (int, float)):
            kl_ratios = [kl_ratios]
        if isinstance(inc_ratios, (int, float)):
            inc_ratios = [inc_ratios]

        stage_reduction = 0.0

        for block_idx in range(stage_depth):
            if block_idx in kl_blocks:
                # Find the ratio index for this block
                block_list_idx = kl_blocks.index(block_idx)
                kl_ratio = kl_ratios[block_list_idx] if block_list_idx < len(kl_ratios) else 1.0

                if block_idx in inc_blocks:
                    inc_idx = inc_blocks.index(block_idx)
                    inc_ratio = inc_ratios[inc_idx] if inc_idx < len(inc_ratios) else 1.0
                    # KL+INC: attn pruned by kl_ratio, ffn pruned by kl_ratio * inc_ratio
                    block_factor = ATTN_RATIO * kl_ratio + FFN_RATIO * kl_ratio * inc_ratio
                else:
                    # KL only: both attn and ffn pruned by kl_ratio
                    block_factor = ATTN_RATIO * kl_ratio + FFN_RATIO * kl_ratio
            else:
                # No pruning - full FLOPs
                block_factor = 1.0

            stage_reduction += block_factor

        # Average over blocks in stage
        avg_stage_factor = stage_reduction / stage_depth
        total_reduction += stage_weight * avg_stage_factor

    return total_reduction


def inference(args, logger):
    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape[0], args.shape[1]
    else:
        raise ValueError('invalid input shape')

    # Build backbone only
    backbone_cfg = cfg.model.backbone
    backbone = MODELS.build(backbone_cfg)

    if torch.cuda.is_available():
        backbone = backbone.cuda()
    backbone = revert_sync_batchnorm(backbone)
    backbone.eval()

    # Get strategy info from the built backbone (after cfg override)
    strategy = backbone.strategy
    stage_config = backbone.stage_config
    inc_stage_config = backbone.inc_stage_config

    # Calculate actual backbone FLOPs
    outputs = get_model_complexity_info(
        backbone,
        input_shape=(3, h, w),
        show_table=False,
        show_arch=False
    )

    actual_flops = outputs['flops']
    params = outputs['params']

    # Estimate pruned FLOPs if using kl_inc
    estimated_reduction = calculate_backbone_pruned_flops(backbone, (h, w))

    result = {
        'strategy': strategy,
        'actual_flops': actual_flops,
        'params': params,
        'estimated_reduction': estimated_reduction,
        'input_shape': (h, w),
        'stage_config': stage_config,
        'inc_stage_config': inc_stage_config,
    }

    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)

    split_line = '=' * 50

    actual_flops_str = _format_size(result['actual_flops'])
    params_str = _format_size(result['params'])

    print(f'{split_line}')
    print(f'Backbone FLOPs Analysis')
    print(f'{split_line}')
    print(f'Strategy: {result["strategy"]}')
    print(f'Input shape: {result["input_shape"]}')
    print(f'{split_line}')
    print(f'Actual FLOPs: {actual_flops_str}')
    print(f'Params: {params_str}')

    if result['estimated_reduction'] is not None:
        estimated_flops = result['actual_flops'] * result['estimated_reduction']
        estimated_flops_str = _format_size(estimated_flops)
        reduction_pct = (1 - result['estimated_reduction']) * 100
        print(f'Estimated reduction factor: {result["estimated_reduction"]:.3f}')
        print(f'Estimated FLOPs (kl_inc): {estimated_flops_str} ({reduction_pct:.1f}% reduction)')

    print(f'{split_line}')
    print('\nStage config (KL):')
    for stage, cfg in result['stage_config'].items():
        print(f'  Stage {stage}: blocks={cfg.get("blocks")}, ratio={cfg.get("ratio")}')
    print('\nStage config (Inc):')
    for stage, cfg in result['inc_stage_config'].items():
        print(f'  Stage {stage}: blocks={cfg.get("blocks")}, inc_ratio={cfg.get("inc_ratio")}')
    print(f'{split_line}')


if __name__ == '__main__':
    main()

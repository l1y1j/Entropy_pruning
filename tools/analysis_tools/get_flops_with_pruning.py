# Copyright (c) OpenMMLab. All rights reserved.
"""FLOPs calculation tool with entropy pruning estimation.

This tool extends the original get_flops.py to estimate FLOPs reduction
under different pruning configurations (KL only vs KL + incremental).

The estimation is based on:
- F_total = F_attn + F_ffn
- KL pruning reduces both Attn and FFN by r_kl
- Incremental pruning only reduces FFN by r_inc (after KL filtering)

Formula:
- KL only: F_pruned = r_kl * F_attn + r_kl * F_ffn = r_kl * F_total
- KL + incremental: F_pruned = r_kl * F_attn + r_kl * r_inc * F_ffn

Note: FLOPs are summed across blocks/stages, not multiplied.
"""

import argparse
import tempfile
from functools import partial
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')

# Attn/FFN ratio for Swin Transformer
# Based on standard Swin paper and common implementations
# NOTE: This is an empirical assumption (0.4/0.6), not a measured value.
# In practice, the ratio varies with resolution and model depth.
# For high-resolution stages, attention dominates; for low-resolution, FFN is heavier.
# Reviewers may note this as a heuristic approximation.
ATTN_RATIO = 0.4
FFN_RATIO = 0.6

# Stage FLOPs proportions based on Swin architecture analysis
# Stage 0: patch embed + few blocks at highest res (~10%)
# Stage 1: 2 blocks at high res (~20%)
# Stage 2: 6 blocks at mid res, where most computation happens (~50%) <- pruning happens here
# Stage 3: 2 blocks at low res (~20%)
# NOTE: These are empirical weights, not measured from real FLOPs profiling.
STAGE_WEIGHTS = [0.1, 0.2, 0.5, 0.2]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get a detector flops with pruning estimation')
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
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def calculate_pruned_flops(base_flops, backbone_cfg, entropy_cfg):
    """Calculate pruned FLOPs using weighted sum, not multiplication.

    We decompose FLOPs by stage and apply pruning factors accordingly:
    - For each pruned block: F_block_pruned = r_kl * F_attn + r_kl * r_inc * F_ffn
    - Total FLOPs = sum of all (original or pruned) block FLOPs

    Args:
        base_flops: Total base FLOPs from get_model_complexity_info
        backbone_cfg: Backbone config dict
        entropy_cfg: Entropy pruning config dict

    Returns:
        estimated_flops: Estimated FLOPs after pruning
        prune_ratio: Overall pruning ratio (0-1, where 1 means no reduction)
        pruning_details: List of details per pruned block
    """
    entropy_enabled = entropy_cfg.get('enabled', False)
    entropy_strategy = entropy_cfg.get('entropy_strategy', 'kl')

    if not entropy_enabled or not entropy_cfg.get('stages_to_prune'):
        return base_flops, 1.0, ['No pruning enabled']

    depths = backbone_cfg.get('depths', [2, 2, 6, 2])
    stages_to_prune = entropy_cfg.get('stages_to_prune', [])
    block_indices = entropy_cfg.get('block_indices', [])
    kl_ratio_cfg = entropy_cfg.get('kl_ratio', {})
    inc_ratio_cfg = entropy_cfg.get('increment_ratio', {})

    # Calculate total blocks and blocks per stage
    num_stages = len(depths)

    # Calculate backbone FLOPs (assume backbone is ~60% of total FLOPs for detector)
    backbone_flops = base_flops * 0.6

    # Calculate pruned FLOPs using direct SUM, not delta accumulation
    # FLOPs_total = sum of all stage FLOPs
    pruned_flops = 0.0
    pruning_details = []

    # Non-backbone FLOPs (neck, head, etc.) remain unchanged
    non_backbone_flops = base_flops * 0.4

    for stage_idx in range(num_stages):
        stage_depth = depths[stage_idx]
        stage_proportion = STAGE_WEIGHTS[stage_idx]
        stage_flops = backbone_flops * stage_proportion

        # FLOPs per block in this stage (evenly divided)
        flops_per_block = stage_flops / stage_depth

        stage_pruned_flops = 0.0

        # Check if this stage has pruning blocks
        if stage_idx in stages_to_prune:
            kl_ratios = kl_ratio_cfg.get(stage_idx, [])
            inc_ratios = inc_ratio_cfg.get(stage_idx, [])

            # Filter valid blocks for this stage (block_indices is shared across stages)
            valid_blocks = [b for b in block_indices if b < stage_depth]

            for i, block_idx in enumerate(valid_blocks):
                kl_ratio = kl_ratios[i] if i < len(kl_ratios) else 1.0

                if entropy_strategy == 'kl_incremental':
                    # Incremental only affects FFN (after KL filtering)
                    # block_idx=0: no incremental, inc_ratio=1.0
                    # block_idx=2: inc_ratios[0] (first in list)
                    # block_idx=4: inc_ratios[1] (second in list)
                    if block_idx == 0:
                        inc_ratio = 1.0
                    else:
                        inc_idx = i - 1  # block_idx=2 -> inc_idx=0, block_idx=4 -> inc_idx=1
                        inc_ratio = inc_ratios[inc_idx] if inc_idx < len(inc_ratios) else 1.0
                    # KL affects both Attn and FFN, increment only affects FFN
                    block_factor = ATTN_RATIO * kl_ratio + FFN_RATIO * kl_ratio * inc_ratio
                    detail = (f"Stage{stage_idx}/Block{block_idx}: "
                              f"KL={kl_ratio}, Inc={inc_ratio}, "
                              f"factor={block_factor:.3f}")
                else:
                    # KL affects both Attn and FFN equally
                    block_factor = kl_ratio
                    detail = f"Stage{stage_idx}/Block{block_idx}: KL={kl_ratio}, factor={block_factor:.3f}"

                block_pruned_flops = flops_per_block * block_factor
                stage_pruned_flops += block_pruned_flops
                pruning_details.append(detail)
        else:
            # No pruning in this stage - keep original FLOPs
            stage_pruned_flops = stage_flops

        pruned_flops += stage_pruned_flops

    estimated_flops = pruned_flops + non_backbone_flops
    prune_ratio = estimated_flops / base_flops

    return estimated_flops, prune_ratio, pruning_details


def inference(args, logger):
    if str(torch.__version__) < '1.12':
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')
    result = {}

    result['ori_shape'] = None
    result['pad_shape'] = None

    logger.warning(
        'Failed to directly get FLOPs, try to get flops with real data')
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    data_batch = next(iter(data_loader))

    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    _forward = model.forward
    data = model.data_preprocessor(data_batch)
    result['ori_shape'] = data['data_samples'][0].ori_shape
    result['pad_shape'] = data['data_samples'][0].pad_shape

    del data_loader
    model.forward = partial(_forward, data_samples=data['data_samples'])
    outputs = get_model_complexity_info(
        model,
        None,
        inputs=data['inputs'],
        show_table=False,
        show_arch=False)
    flops = outputs['flops']
    params = outputs['params']
    result['compute_type'] = 'dataloader: load a picture from the dataset'

    # Extract entropy pruning config
    entropy_cfg = cfg.model.backbone.get('entropy_pruning', {})
    entropy_strategy = entropy_cfg.get('entropy_strategy', 'kl')

    # Calculate pruned FLOPs using weighted sum
    estimated_flops, prune_ratio, pruning_details = calculate_pruned_flops(
        flops, cfg.model.backbone, entropy_cfg)

    result['pruning_strategy'] = entropy_strategy
    result['prune_ratio'] = prune_ratio
    result['pruning_details'] = pruning_details
    result['base_flops'] = flops
    result['flops'] = estimated_flops
    result['params'] = params

    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)

    split_line = '=' * 50
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    base_flops = result['base_flops']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']
    prune_ratio = result['prune_ratio']
    pruning_strategy = result['pruning_strategy']
    pruning_details = result['pruning_details']

    base_flops_str = _format_size(base_flops)
    flops_str = _format_size(flops)
    params_str = _format_size(params)
    reduction = (1 - prune_ratio) * 100

    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')

    print(f'{split_line}')
    print(f'Compute type: {compute_type}')
    print(f'Input shape: {pad_shape}')
    print(f'{split_line}')
    print(f'Base FLOPs (without pruning): {base_flops_str}')
    print(f'Pruning strategy: {pruning_strategy}')
    print(f'Attn/FFN ratio assumption: {ATTN_RATIO}/{FFN_RATIO}')
    print(f'Estimated FLOPs reduction: {reduction:.1f}%')
    print(f'Estimated FLOPs (with pruning): {flops_str}')
    print(f'Params: {params_str}')
    print(f'{split_line}')

    print('\nPruning details:')
    for detail in pruning_details:
        print(f'  - {detail}')

    print('\n!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')


if __name__ == '__main__':
    main()

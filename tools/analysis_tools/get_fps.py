# Copyright (c) OpenMMLab. All rights reserved.
"""FPS measurement tool for Swin Transformer V1 models.

This tool measures the actual inference speed (FPS) of the backbone in both
origin mode (no pruning) and kl_inc mode (with entropy pruning).

Usage:
    # Origin mode FPS (backbone only):
    python tools/analysis_tools/get_fps.py \
        configs/dino/dino-4scale_swin_v1_kl_8xb2-36e_panda.py \
        --backbone-only

    # KL_Inc mode FPS (backbone only):
    python tools/analysis_tools/get_fps.py \
        configs/dino/dino-4scale_swin_v1_kl_8xb2-36e_panda.py \
        --backbone-only \
        --cfg-options backbone.strategy='kl_inc'
"""

import argparse
import os
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

# Add project root to path
FILE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FILE_ROOT))

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet.registry import MODELS

# Import sparse_former modules to register models
from sparse_former.datasets import *  # noqa: F401, F403
from sparse_former.models import *  # noqa: F401, F403


def parse_args():
    parser = argparse.ArgumentParser(description='Get backbone FPS')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--backbone-only',
        action='store_true',
        help='measure backbone only, not full detector')
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='number of warmup iterations')
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='number of iterations to measure')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    args = parser.parse_args()
    return args


def measure_fps(model, inputs, num_warmup=10, num_iterations=100):
    """Measure FPS by running inference iterations.

    Args:
        model: The model to measure
        inputs: Input tensor or dict with 'inputs' and 'data_samples'
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations to measure

    Returns:
        fps: Frames per second
        avg_time: Average time per frame in seconds
        std_time: Standard deviation of times
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if isinstance(inputs, dict):
                _ = model.forward(inputs['inputs'], data_samples=inputs.get('data_samples'))
            else:
                _ = model.forward(inputs)

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()

            if isinstance(inputs, dict):
                _ = model.forward(inputs['inputs'], data_samples=inputs.get('data_samples'))
            else:
                _ = model.forward(inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)

    times = times[:num_iterations]
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    fps = 1.0 / avg_time

    return fps, avg_time, std_time


def inference(args, logger):
    if not torch.cuda.is_available():
        logger.warning('CUDA is not available, FPS measurement may be inaccurate')

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

    if args.backbone_only:
        # Build backbone only
        backbone_cfg = cfg.model.backbone
        backbone = MODELS.build(backbone_cfg)

        if torch.cuda.is_available():
            backbone = backbone.cuda()
        backbone = revert_sync_batchnorm(backbone)
        backbone.eval()

        strategy = backbone_cfg.get('strategy', 'base')

        # Create dummy input
        dummy_input = torch.randn(1, 3, h, w)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()

        # Measure FPS
        fps, avg_time, std_time = measure_fps(
            backbone,
            dummy_input,
            num_warmup=args.warmup,
            num_iterations=args.iterations
        )

        return {
            'fps': fps,
            'avg_time': avg_time,
            'std_time': std_time,
            'input_shape': (h, w),
            'strategy': strategy,
            'backbone_only': True,
            'warmup': args.warmup,
            'iterations': args.iterations,
        }
    else:
        # Build full detector
        data_loader = Runner.build_dataloader(cfg.val_dataloader)
        data_batch = next(iter(data_loader))

        model = MODELS.build(cfg.model)
        if torch.cuda.is_available():
            model = model.cuda()
        model = revert_sync_batchnorm(model)
        model.eval()

        strategy = cfg.model.backbone.get('strategy', 'base')

        data = model.data_preprocessor(data_batch)
        ori_shape = data['data_samples'][0].ori_shape
        pad_shape = data['data_samples'][0].pad_shape

        fps, avg_time, std_time = measure_fps(
            model,
            data,
            num_warmup=args.warmup,
            num_iterations=args.iterations
        )

        return {
            'fps': fps,
            'avg_time': avg_time,
            'std_time': std_time,
            'ori_shape': ori_shape,
            'pad_shape': pad_shape,
            'strategy': strategy,
            'backbone_only': False,
            'warmup': args.warmup,
            'iterations': args.iterations,
        }


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)

    split_line = '=' * 50

    print(f'{split_line}')
    print(f'FPS Measurement Results')
    print(f'{split_line}')
    print(f'Mode: Backbone only' if result['backbone_only'] else 'Mode: Full detector')
    print(f'Strategy: {result["strategy"]}')

    if result['backbone_only']:
        print(f'Input shape: {result["input_shape"]}')
    else:
        print(f'Input shape: {result["pad_shape"]} (original: {result["ori_shape"]})')

    print(f'Warmup iterations: {result["warmup"]}')
    print(f'Measured iterations: {result["iterations"]}')
    print(f'{split_line}')
    print(f'FPS: {result["fps"]:.2f}')
    print(f'Average time per frame: {result["avg_time"]*1000:.2f} ms')
    print(f'Std deviation: {result["std_time"]*1000:.2f} ms')
    print(f'{split_line}')

    if result["std_time"] / result["avg_time"] > 0.05:
        print('Warning: High variance detected (>5%). Consider increasing iterations.')


if __name__ == '__main__':
    main()

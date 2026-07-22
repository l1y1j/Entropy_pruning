# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.append('.')

import argparse
import tempfile
from functools import partial
import os
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version

from sparse_former.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='checkpoint file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='num images of calculate model flops')
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
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to save the log file. If not specified, logs will only be printed to stdout.')
    args = parser.parse_args()
    return args


def calculate_component_flops(model, component_name, data, logger):
    """Calculate FLOPs for a specific model component."""
    component = getattr(model, component_name, None)
    if component is None:
        logger.warning(f'{component_name} not found in the model.')
        return None

    _forward = component.forward
    try:
        # Pass only the inputs to the forward method
        outputs = get_model_complexity_info(
            component,
            None,
            inputs=data['inputs'],
            show_table=True,
            show_arch=True
        )
        return outputs['flops']
    except Exception as e:
        logger.warning(f'Failed to calculate FLOPs for component {component_name}: {e}')
        return None
    finally:
        component.forward = _forward  # Restore original forward method


def calculate_module_flops_and_params(model, data, logger):
    """Calculate FLOPs and params for each module in the model."""
    module_stats = {}

    for name, module in model.named_children():
        _forward = module.forward
        try:
            # Pass only the inputs to the forward method
            outputs = get_model_complexity_info(
                module,
                None,
                inputs=data['inputs'],
                show_table=True,
                show_arch=True
            )
            module_stats[name] = {
                'flops': _format_size(outputs['flops']),
                'params': _format_size(outputs['params'])
            }
        except Exception as e:
            logger.warning(f'Failed to calculate FLOPs for module {name}: {e}')
        finally:
            module.forward = _forward  # Restore original forward method

    return module_stats


def inference(args, logger):
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment，
    # this question involves models like configs/strong_baselines
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)

    result = {}
    avg_flops = []
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model)
    if args.checkpoint:
        from mmengine.runner import load_checkpoint
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    _forward = model.forward

    for idx, data_batch in enumerate(data_loader):
        if idx == args.num_images:
            break
        data = model.data_preprocessor(data_batch)
        result['ori_shape'] = data['data_samples'][0].ori_shape
        result['pad_shape'] = data['data_samples'][0].pad_shape
        if hasattr(data['data_samples'][0], 'batch_input_shape'):
            result['pad_shape'] = data['data_samples'][0].batch_input_shape
        model.forward = partial(_forward, data_samples=data['data_samples'])
        outputs = get_model_complexity_info(
            model,
            None,
            inputs=data['inputs'],
            show_table=True,
            show_arch=True)
        avg_flops.append(outputs['flops'])
        params = outputs['params']
        result['compute_type'] = 'dataloader: load a picture from the dataset'
    del data_loader

    mean_flops = _format_size(int(np.average(avg_flops)))
    params = _format_size(params)
    result['flops'] = mean_flops
    result['params'] = params
    result['module_stats'] = calculate_module_flops_and_params(model, data, logger)

    return result


def main():
    args = parse_args()

    # Set up logging to file if log-file is specified
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = os.path.basename(args.config).replace('.py', '.log')
        log_file = f'local-data/local-log/analysis/{log_file}'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Ensure logger writes to both stdout and file
    logger = MMLogger.get_instance(name='MMLogger', log_file=log_file, log_level='INFO')

    result = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']
    module_stats = result.get('module_stats', {})

    log_output = []
    if pad_shape != ori_shape:
        log_output.append(f'{split_line}\nUse size divisor set input shape '
                          f'from {ori_shape} to {pad_shape}')
    log_output.append(f'{split_line}\nCompute type: {compute_type}\n'
                      f'Input shape: {pad_shape}\nFlops: {flops}\n'
                      f'Params: {params}\n{split_line}')

    for module_name, stats in module_stats.items():
        log_output.append(f'Module: {module_name}\n  FLOPs: {stats["flops"]}\n  Params: {stats["params"]}')

    log_output.append('!!!Please be cautious if you use the results in papers. '
                      'You may need to check if all ops are supported and verify '
                      'that the flops computation is correct.')

    # Print to stdout and write to log file
    for line in log_output:
        print(line)
        logger.info(line)

if __name__ == '__main__':
    main()

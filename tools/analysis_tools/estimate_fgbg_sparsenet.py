#!/usr/bin/env python
"""Rough FG/BG FLOPs estimate for SparseNetEntropy (no per-window masks).

Since SparseNetEntropy doesn't expose per-window execution masks (no selection_vis),
we estimate: (1) total backbone FLOPs from trace, (2) keep ratios from gate_losses,
(3) split by per-stage FG token ratios assuming random pruning within each stage.

Usage:
  python tools/analysis_tools/estimate_fgbg_sparsenet.py \
      configs/dino/dino-4scale_sparsenet_entropy_8xb2-36e_panda.py \
      outputs/sparsenet_entropy/run3/epoch_36.pth \
      --num-images 100
"""

import argparse
import os
import sys

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet.registry import MODELS, DATASETS as MMDET_DATASETS
from mmdet.utils import register_all_modules
from sparse_former.registry import DATASETS as SPARSE_DATASETS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--total-backbone-gflops", type=float, default=None,
                   help="Total backbone GFLOPs from trace tool; if not set, "
                        "computes analytic FLOPs")
    return p.parse_args()


def _metainfo(ds):
    mi = ds.metainfo if hasattr(ds, "metainfo") else {}
    return {
        "img_shape": mi.get("img_shape", None),
        "ori_shape": mi.get("ori_shape", None),
        "scale_factor": mi.get("scale_factor", None),
    }


def build_stage_fg_map(fg_mask, target_h, target_w):
    return F.interpolate(fg_mask[None, None], size=(target_h, target_w), mode='area')


def main():
    args = parse_args()
    register_all_modules()

    import sparse_former.models.backbones.sparsenet_entropy     # noqa
    from sparse_former.models.backbones.sparsenet_entropy import SparseNetEntropy
    import sparse_former.models.detectors.dino_with_gate_loss   # noqa
    import sparse_former.datasets.panda                         # noqa

    # SparseNetEntropy is registered under sparse_former.registry.MODELS (child),
    # but mmdet's build uses mmdet.registry.MODELS (parent) which can't find
    # child registrations. Register it directly into mmdet's MODELS.
    MODELS.register_module(module=SparseNetEntropy, force=True)

    MMDET_DATASETS._module_dict.update(SPARSE_DATASETS._module_dict)

    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    model.to(args.device)

    dataset = MMDET_DATASETS.build(cfg.test_dataloader.dataset)
    num_images = min(args.num_images, len(dataset))

    # Accumulators
    total_kl_keep = 0.0
    total_inc_keep = 0.0
    total_blocks_kl = 0
    total_blocks_inc = 0
    total_fg_area_ratio = 0.0
    stage_hw = {}

    print(f"Processing {num_images} images...")

    for idx in range(num_images):
        data_ = dataset[idx]
        ds = data_["data_samples"]
        meta = _metainfo(ds)

        gt_bboxes = ds.gt_instances.bboxes
        if hasattr(gt_bboxes, 'tensor'):
            gt_bboxes = gt_bboxes.tensor
        scale_factor = meta.get("scale_factor", None)

        raw_inputs = data_["inputs"]
        if not isinstance(raw_inputs, list):
            raw_inputs = [raw_inputs]

        batch = model.data_preprocessor(
            {"inputs": raw_inputs, "data_samples": [ds]}, training=False,
        )
        inputs_proc = batch["inputs"]
        if isinstance(inputs_proc, list):
            inputs_proc = torch.stack(inputs_proc)
        if inputs_proc.dim() == 3:
            inputs_proc = inputs_proc.unsqueeze(0)

        _, _, proc_H, proc_W = inputs_proc.shape

        # GT boxes to preprocessed coords
        if scale_factor is not None:
            sf = scale_factor.tolist() if hasattr(scale_factor, 'tolist') else list(scale_factor)
            w_scale, h_scale = float(sf[0]), float(sf[1]) if len(sf) > 1 else float(sf[0])
        else:
            w_scale, h_scale = 1.0, 1.0

        boxes_resized = torch.zeros((0, 4))
        if gt_bboxes.shape[0] > 0:
            boxes_resized = gt_bboxes.clone().float()
            boxes_resized[:, 0] *= w_scale
            boxes_resized[:, 1] *= h_scale
            boxes_resized[:, 2] *= w_scale
            boxes_resized[:, 3] *= h_scale

        # Binary FG mask at input resolution
        fg_mask = torch.zeros((proc_H, proc_W), dtype=torch.float32)
        for box in boxes_resized:
            x1, y1, x2, y2 = box.round().long()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(proc_W, x2), min(proc_H, y2)
            if x2 > x1 and y2 > y1:
                fg_mask[y1:y2, x1:x2] = 1.0

        fg_area_ratio = fg_mask.sum().item() / (proc_H * proc_W)
        total_fg_area_ratio += fg_area_ratio

        # Run backbone
        with torch.no_grad():
            model.extract_feat(inputs_proc)

        # Collect keep ratios from gate_losses
        gate_losses = model.backbone._gate_losses
        kl_keep = gate_losses.get('kl_keep_ratio', None)
        inc_keep = gate_losses.get('inc_keep_ratio', None)

        if kl_keep is not None:
            total_kl_keep += kl_keep.item()
            total_blocks_kl += 1
        if inc_keep is not None:
            total_inc_keep += inc_keep.item()
            total_blocks_inc += 1

        # Get feature map sizes from patch_embed output shape
        with torch.no_grad():
            x_conv = model.backbone.patch_embed.projection(inputs_proc)
            _, _, H0, W0 = x_conv.shape
        H, W = H0, W0
        for s in range(4):
            if s not in stage_hw:
                stage_hw[s] = []
            stage_hw[s].append((H, W))
            H, W = (H + 1) // 2, (W + 1) // 2

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx + 1}/{num_images}] "
                  f"KL={kl_keep.item():.3f}" if kl_keep is not None else f"  [{idx + 1}/{num_images}]",
                  f" INC={inc_keep.item():.3f}" if inc_keep is not None else "",
                  f" FG area={fg_area_ratio*100:.1f}%")

    N = num_images
    avg_kl_keep = total_kl_keep / max(total_blocks_kl, 1) if total_blocks_kl > 0 else 1.0
    avg_inc_keep = total_inc_keep / max(total_blocks_inc, 1) if total_blocks_inc > 0 else 1.0

    # Average FG area ratio
    r_bar_F = total_fg_area_ratio / N

    # ---- Estimate FLOPs split ----
    # We don't have per-window masks, so we estimate using keep ratios × per-stage FG ratios.
    # This assumes random (non-FG-biased) pruning within each stage.
    # The dynamic portion is estimated as keep_ratio * (total_attn_ffn_flops).

    # Get analytic total backbone FLOPs if not provided
    if args.total_backbone_gflops:
        G_total = args.total_backbone_gflops
    else:
        # Use the 6.181 from swin as a placeholder — actual SparseNet total
        # would need to be measured with get_flops_2.py
        print("\nWARNING: No --total-backbone-gflops provided, using 6.181 as placeholder.")
        print("Run get_flops_2.py to measure the actual SparseNet backbone FLOPs.")
        G_total = 6.181

    # Rough estimate: dynamic = keep_ratio * fraction of total that is attn+ffn
    # For Swin-T: attn+ffn ≈ 91.87 G (full), total ≈ 92.75 G (full, from log)
    # So attn+ffn ≈ 99% of full total.
    # But with pruning, the dynamic fraction scales with keep_ratio.
    # Simplified: G_dyn ≈ avg_keep * G_total (since most FLOPs are in attn+ffn)
    avg_keep = (avg_kl_keep + avg_inc_keep) / 2.0
    G_dyn = avg_keep * G_total
    G_fixed = G_total - G_dyn

    # Split dynamic by per-stage FG ratios (random pruning assumption)
    # Use input-level FG ratio for simplicity
    G_dyn_fg = r_bar_F * G_dyn
    G_dyn_bg = (1 - r_bar_F) * G_dyn

    # Split fixed by per-stage FG ratios
    G_fixed_fg = r_bar_F * G_fixed
    G_fixed_bg = (1 - r_bar_F) * G_fixed

    G_F = G_dyn_fg + G_fixed_fg
    G_B = G_dyn_bg + G_fixed_bg

    print()
    print("=" * 60)
    print(f"SparseNetEntropy FG/BG FLOPs Estimate (ROUGH)")
    print("=" * 60)
    print(f"WARNING: No per-window masks — assumes RANDOM pruning within stages.")
    print(f"         This estimate does NOT capture FG-biased pruning behavior.")
    print()
    print(f"Images: {N}")
    print(f"Mean FG area ratio: {r_bar_F * 100:.2f}%")
    print(f"Avg KL keep ratio:  {avg_kl_keep * 100:.2f}%")
    print(f"Avg INC keep ratio: {avg_inc_keep * 100:.2f}%")
    print()
    print(f"Dynamic GFLOPs (est): {G_dyn:.4f}  ({G_dyn/G_total*100:.1f}% of total)")
    print(f"Fixed GFLOPs (est):   {G_fixed:.4f}")
    print()
    print(f"Final GFLOPs-F: {G_F:.4f}")
    print(f"Final GFLOPs-B: {G_B:.4f}")
    print(f"Final GFLOPs-O: {G_total}")
    print(f"F-ratio: {G_F / G_total * 100:.2f}%")


if __name__ == "__main__":
    main()

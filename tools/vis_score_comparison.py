#!/usr/bin/env python
"""Variance vs SRE comparison figure.

For each W-MSA block, renders a 1×3 figure:
  Original | Variance (VAR) heatmap | SRE (KL) heatmap

Both heatmaps use the same viridis colormap and unified normalisation
(p5–p95 pooled across both scores) so the comparison is fair.

Usage::

    python tools/vis_score_comparison.py \
        configs/dino/dino-4scale_swin_v3_kl_8xb2-36e_panda.py \
        outputs/v3_kl_inc/13/epoch_24.pth \
        --image-indices 79 --output-dir outputs/exec_states_2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image

import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet.registry import MODELS, DATASETS as MMDET_DATASETS
from mmdet.utils import register_all_modules
from sparse_former.registry import DATASETS as SPARSE_DATASETS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Variance vs SRE comparison")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--image-indices", type=str, default=None,
                   help="comma-separated image indices, e.g. '23'")
    p.add_argument("--include-sw", action="store_true",
                   help="include SW-MSA blocks (default: W-MSA only)")
    p.add_argument("--output-dir", default="outputs/exec_states_2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cmap", default="viridis",
                   help="matplotlib colormap (default: viridis)")
    return p.parse_args()


def _metainfo(ds):
    mi = ds.metainfo if hasattr(ds, "metainfo") else {}
    return {
        "img_shape": mi.get("img_shape", None),
        "ori_shape": mi.get("ori_shape", None),
        "img_path": mi.get("img_path", None),
    }


def _spatial_map(score_grid, ws, feat_H, feat_W, proc_H, proc_W, meta):
    """Window-grid scores → original-image pixel grid."""
    fm = np.repeat(np.repeat(score_grid, ws, axis=0), ws, axis=1)
    fm = fm[:feat_H, :feat_W]
    fm_input = cv2.resize(fm, (proc_W, proc_H), interpolation=cv2.INTER_NEAREST)
    img_h, img_w = meta["img_shape"] if meta["img_shape"] else (proc_H, proc_W)
    fm_input = fm_input[:img_h, :img_w]
    ori_h, ori_w = meta["ori_shape"] if meta["ori_shape"] else (img_h, img_w)
    if (img_h, img_w) != (ori_h, ori_w):
        fm_ori = cv2.resize(fm_input, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
    else:
        fm_ori = fm_input
    return fm_ori


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    register_all_modules()

    import sparse_former.models.backbones.swin_baseline_v3   # noqa
    import sparse_former.models.detectors.dino_with_gate_loss # noqa
    import sparse_former.datasets.panda                       # noqa

    MMDET_DATASETS._module_dict.update(SPARSE_DATASETS._module_dict)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load model ----------------------------------------------------------
    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    model.to(args.device)
    model.backbone.set_enable_selection_vis(True)

    # ---- dataset -------------------------------------------------------------
    dataset = MMDET_DATASETS.build(cfg.test_dataloader.dataset)
    if args.image_indices is not None:
        img_indices = [int(x.strip()) for x in args.image_indices.split(",")]
    else:
        img_indices = list(range(min(args.num_images, len(dataset))))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for idx in img_indices:
        data_ = dataset[idx]
        ds = data_["data_samples"]
        meta = _metainfo(ds)

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

        orig_img = None
        if meta["img_path"] and os.path.exists(meta["img_path"]):
            orig_img = np.array(Image.open(meta["img_path"]))

        with torch.no_grad():
            model.extract_feat(inputs_proc)

        # ---- iterate all W-MSA blocks ----------------------------------------
        for si, stage in enumerate(model.backbone.stages):
            for bi, block in enumerate(stage.blocks):
                vis = block.selection_vis
                if vis is None:
                    continue
                if vis['shift_size'] != 0 and not args.include_sw:
                    continue

                # require both kl_norm and var_norm
                if 'kl_norm' not in vis or 'var_norm' not in vis:
                    continue

                H_w = vis['H_pad'] // vis['window_size']
                W_w = vis['W_pad'] // vis['window_size']

                kl_scores = vis['kl_norm'].numpy()
                var_scores = vis['var_norm'].numpy()

                if len(kl_scores) != H_w * W_w or len(var_scores) != H_w * W_w:
                    continue

                # ---- spatial mapping for both scores -------------------------
                grid_kl = kl_scores.reshape(H_w, W_w)
                grid_var = var_scores.reshape(H_w, W_w)

                fm_kl = _spatial_map(
                    grid_kl, vis['window_size'],
                    vis['H'], vis['W'], proc_H, proc_W, meta,
                )
                fm_var = _spatial_map(
                    grid_var, vis['window_size'],
                    vis['H'], vis['W'], proc_H, proc_W, meta,
                )

                # ---- separate normalisation (p5–p95 per score) --------------
                p5_var, p95_var = np.percentile(fm_var, 5), np.percentile(fm_var, 95)
                p5_kl, p95_kl = np.percentile(fm_kl, 5), np.percentile(fm_kl, 95)

                fm_var_norm = np.clip((fm_var - p5_var) / max(p95_var - p5_var, 1e-8), 0.0, 1.0)
                fm_kl_norm = np.clip((fm_kl - p5_kl) / max(p95_kl - p5_kl, 1e-8), 0.0, 1.0)

                med_var = float(np.median(var_scores))
                med_kl = float(np.median(kl_scores))

                # ---- render 1×3 comparison figure ---------------------------
                fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
                fig.subplots_adjust(right=0.90)

                # Original
                if orig_img is not None:
                    axes[0].imshow(orig_img)
                axes[0].set_title("Original", fontsize=11)
                axes[0].axis("off")

                # Variance
                if orig_img is not None:
                    axes[1].imshow(orig_img)
                im1 = axes[1].imshow(fm_var_norm, alpha=0.55, cmap=args.cmap,
                                     vmin=0.0, vmax=1.0)
                axes[1].set_title(
                    f"S{si} B{bi:02d}  Variance (VAR)\nmedian={med_var:.4f}",
                    fontsize=10,
                )
                axes[1].axis("off")
                cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04,
                                     label="VAR (per-block norm.)")

                # SRE
                if orig_img is not None:
                    axes[2].imshow(orig_img)
                im2 = axes[2].imshow(fm_kl_norm, alpha=0.55, cmap=args.cmap,
                                     vmin=0.0, vmax=1.0)
                axes[2].set_title(
                    f"S{si} B{bi:02d}  SRE (KL Divergence)\nmedian={med_kl:.4f}",
                    fontsize=10,
                )
                axes[2].axis("off")
                cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04,
                                     label="SRE (per-block norm.)")

                fname = f"img{idx:02d}_S{si}_B{bi:02d}_compare_{args.cmap}.png"
                fig.savefig(os.path.join(args.output_dir, fname),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)

        print(f"[img {idx:2d}] done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""KL / DEV score heatmaps overlaid on the original image.

For every W-MSA block, the per-window normalised score is broadcast to the
feature map, spatially mapped to the original image, and rendered as a
viridis heatmap with the original image visible underneath.

Usage::

    python tools/vis_score_heatmaps.py \
        configs/dino/dino-4scale_swin_v3_kl_8xb2-36e_panda.py \
        outputs/v3_kl_inc/13/epoch_24.pth \
        --num-images 100 --output-dir outputs/exec_states_2
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
    p = argparse.ArgumentParser(description="KL / DEV score heatmap visualisation")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--score-type", type=str, default="both",
                   choices=["kl", "inc", "var", "both"],
                   help="which score to draw: kl/SRE, inc/DEV, var, or both (default: both)")
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--image-indices", type=str, default=None,
                   help="comma-separated image indices, e.g. '23'")
    p.add_argument("--include-sw", action="store_true",
                   help="include SW-MSA blocks (default: W-MSA only)")
    p.add_argument("--output-dir", default="outputs/exec_states_2")
    p.add_argument("--device", default="cuda:0")
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
    # broadcast window → feature map
    fm = np.repeat(np.repeat(score_grid, ws, axis=0), ws, axis=1)
    fm = fm[:feat_H, :feat_W]
    # resize to input
    fm_input = cv2.resize(fm, (proc_W, proc_H), interpolation=cv2.INTER_NEAREST)
    # remove data_preprocessor padding
    img_h, img_w = meta["img_shape"] if meta["img_shape"] else (proc_H, proc_W)
    fm_input = fm_input[:img_h, :img_w]
    # map to original
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

                H_w = vis['H_pad'] // vis['window_size']
                W_w = vis['W_pad'] // vis['window_size']

                score_names = ["kl", "inc"] if args.score_type == "both" else [args.score_type]
                for score_name in score_names:
                    key = f"{score_name}_norm"
                    if key not in vis:
                        continue
                    scores = vis[key].numpy()  # (N_win,)

                    if len(scores) != H_w * W_w:
                        continue

                    grid = scores.reshape(H_w, W_w)
                    fm_ori = _spatial_map(
                        grid, vis['window_size'],
                        vis['H'], vis['W'], proc_H, proc_W, meta,
                    )

                    # per-block robust normalisation (p5–p95)
                    p5 = float(np.percentile(fm_ori, 5))
                    p95 = float(np.percentile(fm_ori, 95))
                    denom = max(p95 - p5, 1e-8)
                    fm_norm = np.clip((fm_ori - p5) / denom, 0.0, 1.0)

                    median = float(np.median(scores))

                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    if orig_img is not None:
                        ax.imshow(orig_img)
                    im = ax.imshow(fm_norm, alpha=0.55, cmap="viridis",
                                   vmin=0.0, vmax=1.0)
                    score_label = {"kl": "SRE", "inc": "DEV", "var": "VAR"}[score_name]
                    ax.set_title(
                        f"S{si} B{bi:02d}  {score_label}  median={median:.4f}",
                        fontsize=10,
                    )
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                 label=f"{score_label} score (per-block norm.)")

                    fname = f"img{idx:02d}_S{si}_B{bi:02d}_{score_label}.png"
                    fig.savefig(os.path.join(args.output_dir, fname),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)

        print(f"[img {idx:2d}] done")


if __name__ == "__main__":
    main()

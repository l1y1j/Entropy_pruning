#!/usr/bin/env python
"""Fixed-budget attention selection (2nd row of the 3×3 figure).

Uses the same KL scores as ECHO, but applies a shared Top-K budget across
three blocks.  Budget stays constant per block; selected window positions
may vary because each block re-ranks windows with its own KL scores.

Usage::

    python tools/vis_fixed_budget.py \
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
    p = argparse.ArgumentParser(description="Fixed-budget attention visualisation")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--target-blocks", type=str, default="S0B0,S1B0,S2B0",
                   help="comma-separated 'S{stage}B{block}' pairs")
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--image-indices", type=str, default=None,
                   help="comma-separated image indices, e.g. '0,5,10'")
    p.add_argument("--fixed-ratio", type=str, default=None,
                   help="override ratio(s): single value = shared, or comma-list "
                        "matching --target-blocks order, e.g. '0.138,0.085,0.036'")
    p.add_argument("--output-dir", default="outputs/exec_states_2")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def _parse_block_spec(spec):
    """Parse 'S0B0' → (stage_idx, block_idx)."""
    s = spec.strip()
    assert s.startswith("S"), f"bad spec: {s}"
    parts = s[1:].split("B")
    return int(parts[0]), int(parts[1])


def _metainfo(ds):
    mi = ds.metainfo if hasattr(ds, "metainfo") else {}
    return {
        "img_shape": mi.get("img_shape", None),
        "ori_shape": mi.get("ori_shape", None),
        "img_path": mi.get("img_path", None),
    }


def _state_to_overlay(state_map, orig_img):
    """2-state overlay matching ECHO row colours.

    state=0 (not selected): dark blue 50%   #253A70
    state=1 (selected):     teal 45%         #35B7A5
    """
    H, W = state_map.shape
    if orig_img.shape[:2] != (H, W):
        orig_img = cv2.resize(orig_img, (W, H), interpolation=cv2.INTER_LINEAR)

    COLOURS = {
        0: np.array([0x25, 0x3A, 0x70], dtype=np.float32),  # dark blue
        1: np.array([0xF3, 0x9A, 0x38], dtype=np.float32),  # orange
    }
    ALPHAS = {0: 0.50, 1: 0.45}

    overlay = orig_img.astype(np.float32)
    for s, alpha in ALPHAS.items():
        mask = state_map == s
        if not mask.any():
            continue
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * COLOURS[s]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _build_state_map(mask_1d, H_w, W_w, ws, feat_H, feat_W):
    """Binary mask → feature-map state grid."""
    state = mask_1d.astype(np.uint8)          # 0 or 1
    state = state.reshape(H_w, W_w)
    state_fm = np.repeat(np.repeat(state, ws, axis=0), ws, axis=1)
    state_fm = state_fm[:feat_H, :feat_W]
    return state_fm


def _spatial_map_to_ori(state_fm, proc_H, proc_W, meta):
    state_input = cv2.resize(
        state_fm, (proc_W, proc_H), interpolation=cv2.INTER_NEAREST,
    )
    img_h, img_w = meta["img_shape"] if meta["img_shape"] else (proc_H, proc_W)
    state_input = state_input[:img_h, :img_w]
    ori_h, ori_w = meta["ori_shape"] if meta["ori_shape"] else (img_h, img_w)
    if (img_h, img_w) != (ori_h, ori_w):
        state_ori = cv2.resize(
            state_input, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST,
        )
    else:
        state_ori = state_input
    return state_ori


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

    block_specs = [s.strip() for s in args.target_blocks.split(",")]
    targets = [_parse_block_spec(b) for b in block_specs]
    print(f"Target blocks: {targets}")

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

        # ---- collect three blocks --------------------------------------------
        vis_list = []
        for si, bi in targets:
            block = model.backbone.stages[si].blocks[bi]
            vis = block.selection_vis
            if vis is None:
                print(f"  [SKIP] S{si}B{bi}: no selection_vis")
                continue
            vis_list.append((si, bi, vis))

        if len(vis_list) < 2:
            print(f"[img {idx}] not enough valid blocks, skip")
            continue

        # ---- per-block ratios -------------------------------------------------
        if args.fixed_ratio is not None:
            parts = [float(x.strip()) for x in args.fixed_ratio.split(",")]
            if len(parts) == 1:
                r_list = [parts[0]] * len(vis_list)
            else:
                r_list = parts
            print(f"  User-specified: {[f'{r:.3f}' for r in r_list]}")
        else:
            r_list = [int(v['kl_mask'].sum()) / (v['H_pad']//v['window_size'] * v['W_pad']//v['window_size'])
                      for _, _, v in vis_list]
            print(f"  ECHO ratios: {[f'{r:.3f}' for r in r_list]}")

        # ---- fixed-budget masks ----------------------------------------------
        records = []
        for j, (si, bi, vis) in enumerate(vis_list):
            r_i = r_list[j]
            H_w = vis['H_pad'] // vis['window_size']
            W_w = vis['W_pad'] // vis['window_size']
            N = H_w * W_w
            k_i = max(1, int(round(r_i * N)))
            scores = vis['kl_norm'].clone()
            top_idx = torch.topk(scores, k=min(k_i, len(scores))).indices
            fixed_mask = torch.zeros(N, dtype=torch.bool)
            fixed_mask[top_idx] = True

            state_fm = _build_state_map(
                fixed_mask.numpy(), H_w, W_w,
                vis['window_size'], vis['H'], vis['W'],
            )
            state_ori = _spatial_map_to_ori(state_fm, proc_H, proc_W, meta)

            pct = 100.0 * r_i
            records.append({
                'si': si, 'bi': bi,
                'state_ori': state_ori,
                'pct': pct, 'N': N,
            })
            print(f"  S{si}B{bi}: fixed F = {pct:.1f}%  ({k_i}/{N})")

        # ---- render ----------------------------------------------------------
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1×N row
        fig, axes = plt.subplots(
            1, len(records) + 1,
            figsize=(5 * (len(records) + 1), 5),
            gridspec_kw={'width_ratios': [0.3] + [1] * len(records)},
        )
        axes[0].imshow(orig_img) if orig_img is not None else None
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis("off")

        for j, rec in enumerate(records):
            ax = axes[j + 1]
            overlay = _state_to_overlay(rec['state_ori'], orig_img)
            ax.imshow(overlay)
            ax.set_title(
                f"S{rec['si']} B{rec['bi']}\nF = {rec['pct']:.1f}%", fontsize=11,
            )
            ax.axis("off")
        fig.tight_layout()

        # save individual blocks
        for rec in records:
            fig_s, ax_s = plt.subplots(1, 1, figsize=(6, 5))
            overlay = _state_to_overlay(rec['state_ori'], orig_img)
            ax_s.imshow(overlay)
            ax_s.set_title(
                f"S{rec['si']} B{rec['bi']}  F = {rec['pct']:.1f}%", fontsize=11,
            )
            ax_s.axis("off")
            fname = f"img{idx:02d}_S{rec['si']}_B{rec['bi']:02d}_fixed.png"
            fig_s.savefig(os.path.join(args.output_dir, fname),
                          dpi=150, bbox_inches="tight")
            plt.close(fig_s)

        # save row
        fname_row = f"img{idx:02d}_fixed_row.png"
        fig.savefig(os.path.join(args.output_dir, fname_row),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → img{idx:02d}")


if __name__ == "__main__":
    main()

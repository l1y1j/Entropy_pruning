#!/usr/bin/env python
"""Visualise ECHO execution states across blocks (2nd row of the 3x2 figure).

For selected W-MSA blocks, each window is coloured according to its execution
state in the current block:

    white  — skipped both Attention and FFN  (kl_keep=0)
    grey   — executed Attention, skipped FFN  (kl_keep=1, inc_keep=0)
    colour — executed both Attention and FFN  (kl_keep=1, inc_keep=1)

Usage::

  python tools/vis_execution_states.py \
      configs/dino/dino-4scale_swin_v3_kl_8xb2-36e_panda.py \  # 配置文件
      outputs/v3_kl_inc/13/epoch_24.pth \                       # ← 改这里换权重
      --target-stage 0,1,2,3 \                         # ← 改这里换 stage (0/1/2/3)
      --target-blocks 0 \                    # ← 改这里选 block
      --num-images 20 \                          # ← 改这里调图片数量
      --output-dir outputs/v3_kl_inc/13/exec_states  # 输出目录

  python tools/vis_execution_states.py configs/dino/dino-4scale_swin_v3_kl_8xb2-36e_panda.py outputs/v3_kl_inc/13/epoch_24.pth --target-stage 0,1,2,3 --target-blocks 0 --num-images 20 --output-dir outputs/exec_states
      
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
    p = argparse.ArgumentParser(description="ECHO execution-state visualisation")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--target-stage", type=int, default=2,
                   help="stage index to visualise (default: 2)")
    p.add_argument("--target-blocks", type=str, default="0,2,4",
                   help="comma-separated block indices (W-MSA only)")
    p.add_argument("--num-images", type=int, default=3)
    p.add_argument("--output-dir", default="outputs/exec_states")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def _metainfo(ds):
    mi = ds.metainfo if hasattr(ds, "metainfo") else {}
    return {
        "img_shape": mi.get("img_shape", None),
        "ori_shape": mi.get("ori_shape", None),
        "img_path": mi.get("img_path", None),
    }


def _state_to_overlay(state_map, orig_img):
    """Overlay semi-transparent coloured masks on the original image.

    State    Meaning                         Colour     Alpha
      0      skipped both Attention & FFN    dark blue  50%
      1      Attention only, skipped FFN     teal       45%
      2      both Attention & FFN            orange     55%

    The original image is always fully visible underneath.
    """
    H, W = state_map.shape
    if orig_img.shape[:2] != (H, W):
        orig_img = cv2.resize(orig_img, (W, H), interpolation=cv2.INTER_LINEAR)

    # colour definitions (RGB — PIL returns RGB arrays)
    COLOURS = {
        0: np.array([0x25, 0x3A, 0x70], dtype=np.float32),  # dark blue  #253A70
        1: np.array([0xF3, 0x9A, 0x38], dtype=np.float32),  # orange     #F39A38
        2: np.array([0xF3, 0x9A, 0x38], dtype=np.float32),  # orange     #F39A38
    }
    ALPHAS = {0: 0.50, 1: 0.45, 2: 0.45}

    overlay = orig_img.astype(np.float32)

    for s in (0, 1, 2):
        alpha = ALPHAS[s]
        mask = state_map == s
        if not mask.any():
            continue
        colour = COLOURS[s]
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * colour

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _build_state_map(kl_mask, inc_mask, H_w, W_w, ws, feat_H, feat_W):
    """Build a window-level state grid and broadcast to feature-map pixels.

    Returns
    -------
    state_fm : (feat_H, feat_W)  uint8
    """
    state = np.zeros(H_w * W_w, dtype=np.uint8)
    state[kl_mask.numpy()] = 1
    state[inc_mask.numpy()] = 2
    state = state.reshape(H_w, W_w)
    # broadcast window → feature map
    state_fm = np.repeat(np.repeat(state, ws, axis=0), ws, axis=1)
    # crop window-partition padding
    state_fm = state_fm[:feat_H, :feat_W]
    return state_fm


def _spatial_map_to_ori(state_fm, proc_H, proc_W, meta):
    """Resize a feature-map state grid to the original image coordinates."""
    # 1. feature map → preprocessed input size (NEAREST to keep discrete states)
    state_input = cv2.resize(
        state_fm, (proc_W, proc_H), interpolation=cv2.INTER_NEAREST,
    )
    # 2. remove data_preprocessor padding
    img_h, img_w = meta["img_shape"] if meta["img_shape"] else (proc_H, proc_W)
    state_input = state_input[:img_h, :img_w]
    # 3. map to original image
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

    target_blocks = [int(x) for x in args.target_blocks.split(",")]

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
    num_images = min(args.num_images, len(dataset))

    for idx in range(num_images):
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

        # ---- collect selected blocks -----------------------------------------
        records = []
        stage = model.backbone.stages[args.target_stage]
        for bi, block in enumerate(stage.blocks):
            if bi not in target_blocks:
                continue
            vis = block.selection_vis
            if vis is None or vis['shift_size'] != 0:
                print(f"  [SKIP] S{args.target_stage}B{bi}: "
                      f"vis={vis is not None} shift={vis['shift_size'] if vis else '?'}")
                continue

            H_w = vis['H_pad'] // vis['window_size']
            W_w = vis['W_pad'] // vis['window_size']
            N = H_w * W_w
            n_kl = int(vis['kl_mask'].sum())
            n_inc = int(vis['inc_mask'].sum())

            state_fm = _build_state_map(
                vis['kl_mask'], vis['inc_mask'],
                H_w, W_w, vis['window_size'],
                vis['H'], vis['W'],
            )
            state_ori = _spatial_map_to_ori(state_fm, proc_H, proc_W, meta)

            records.append({
                'block_idx': bi,
                'state_ori': state_ori,
                'n_kl': n_kl,
                'n_inc': n_inc,
                'N': N,
            })
            print(f"  S{args.target_stage}B{bi}: Attn={n_kl}/{N}={100*n_kl/N:.1f}%  "
                  f"FFN={n_inc}/{N}={100*n_inc/N:.1f}%")

        if not records:
            print(f"[img {idx}] no records — is stage {args.target_stage} "
                  f"block(s) {target_blocks} W-MSA?")
            continue

        # ---- render ----------------------------------------------------------
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            1, len(records) + 1,
            figsize=(5 * (len(records) + 1), 5),
            gridspec_kw={'width_ratios': [0.3] + [1] * len(records)},
        )

        # --- left panel: original image ---
        ax0 = axes[0]
        if orig_img is not None:
            ax0.imshow(orig_img)
        ax0.set_title("Original", fontsize=11)
        ax0.axis("off")

        # --- block panels ---
        for j, rec in enumerate(records):
            ax = axes[j + 1]
            overlay = _state_to_overlay(rec['state_ori'], orig_img)
            ax.imshow(overlay)
            a_pct = 100 * rec['n_kl'] / rec['N']
            f_pct = 100 * rec['n_inc'] / rec['N']
            ax.set_title(
                f"S{args.target_stage} B{rec['block_idx']}\n"
                f"Attn {a_pct:.1f}%   FFN {f_pct:.1f}%",
                fontsize=11,
            )
            ax.axis("off")

        fig.tight_layout()

        # save individual blocks
        for j, rec in enumerate(records):
            fig_single, ax_s = plt.subplots(1, 1, figsize=(6, 5))
            overlay = _state_to_overlay(rec['state_ori'], orig_img)
            ax_s.imshow(overlay)
            a_pct = 100 * rec['n_kl'] / rec['N']
            f_pct = 100 * rec['n_inc'] / rec['N']
            ax_s.set_title(
                f"S{args.target_stage} B{rec['block_idx']}  "
                f"Attn {a_pct:.1f}%  FFN {f_pct:.1f}%",
                fontsize=11,
            )
            ax_s.axis("off")
            fname_s = f"img{idx:02d}_S{args.target_stage}_B{rec['block_idx']:02d}.png"
            fig_single.savefig(os.path.join(args.output_dir, fname_s),
                               dpi=150, bbox_inches="tight")
            plt.close(fig_single)

        # save 1×N row
        fname_row = f"img{idx:02d}_row.png"
        fig.savefig(os.path.join(args.output_dir, fname_row),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → img{idx:02d}")

        # also save raw mask data for later use
        mask_data = {
            'img_idx': idx,
            'stage': args.target_stage,
            'proc_H': proc_H, 'proc_W': proc_W,
            'records': [{
                'block_idx': r['block_idx'],
                'n_kl': r['n_kl'], 'n_inc': r['n_inc'], 'N': r['N'],
            } for r in records],
        }
        np_path = os.path.join(args.output_dir, f"img{idx:02d}_masks.npy")
        np.save(np_path, mask_data)
        print(f"  mask data → {np_path}")


if __name__ == "__main__":
    main()

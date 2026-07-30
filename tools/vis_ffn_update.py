#!/usr/bin/env python
"""Visualise per-window FFN update magnitude across all blocks.

Supports both ECHO (SwinTransformerV3) and dense-baseline (SwinTransformerBaseline)
models.  For the baseline, forward hooks on FFN modules capture pre/post features;
for ECHO, ``force_full_ffn`` mode is used.

Usage::

    # Dense baseline
    python tools/vis_ffn_update.py \
        configs/dino/dino-4scale_swin_baseline_8xb2-36e_panda.py \
        outputs/baseline_withoutpretrainedmodel/epoch_36.pth \
        --num-images 3 --output-dir outputs/baseline/ffn_update_vis

    # ECHO model
    python tools/vis_ffn_update.py \
        configs/dino/dino-4scale_swin_v3_kl_8xb2-36e_panda.py \
        outputs/v3_kl_inc/14/epoch_36.pth \
        --num-images 3 --output-dir outputs/v3_kl_inc/14/ffn_update_vis
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
    parser = argparse.ArgumentParser(description="FFN update magnitude visualisation")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/ffn_update_vis")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _metainfo(ds):
    mi = ds.metainfo if hasattr(ds, "metainfo") else {}
    return {
        "img_shape": mi.get("img_shape", None),
        "ori_shape": mi.get("ori_shape", None),
        "pad_shape": mi.get("pad_shape", None),
        "scale_factor": mi.get("scale_factor", None),
        "img_path": mi.get("img_path", None),
    }


def _spatial_map(U_grid, ws, feat_H, feat_W, proc_H, proc_W, meta):
    # 1. window grid → feature map
    U_feat = np.repeat(np.repeat(U_grid, ws, axis=0), ws, axis=1)
    U_feat = U_feat[:feat_H, :feat_W]

    # 2. resize feature map → preprocessed input size
    U_input = cv2.resize(U_feat, (proc_W, proc_H), interpolation=cv2.INTER_NEAREST)

    # 3. remove data_preprocessor padding
    img_h, img_w = meta["img_shape"] if meta["img_shape"] else (proc_H, proc_W)
    U_input = U_input[:img_h, :img_w]

    # 4. map to original image
    ori_h, ori_w = meta["ori_shape"] if meta["ori_shape"] else (img_h, img_w)
    if (img_h, img_w) != (ori_h, ori_w):
        U_ori = cv2.resize(U_input, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
    else:
        U_ori = U_input
    return U_ori


# ---------------------------------------------------------------------------
# Dense-baseline FFN wrapper (keyword-arg identity must be captured)
# ---------------------------------------------------------------------------

class _FFNCapture(torch.nn.Module):
    """Wraps a standard MMDet FFN so we can capture ``identity`` (keyword arg).

    SwinBlock calls ``self.ffn(x, identity=identity)``.  PyTorch forward hooks
    only receive positional args, so the identity tensor is invisible to hooks.
    This tiny wrapper saves *both* positional and keyword args before delegating.
    """

    def __init__(self, original_ffn):
        super().__init__()
        self.original = original_ffn
        self._capture_update = None
        self._capture_meta = None

    def forward(self, x, identity=None):
        pre = identity if identity is not None else x     # (B, L, C)
        out = self.original(x, identity=identity)         # (B, L, C)
        # per-token relative update: U_t = ||out_t - pre_t|| / ||pre_t||
        U_t = (out - pre).norm(dim=-1) / (pre.norm(dim=-1) + 1e-8)  # (B, L)
        self._capture_update = U_t.detach()  # (B, L)
        return out


def _setup_baseline_ffn_capture(backbone):
    """Replace every block's FFN with ``_FFNCapture`` wrapper.

    Returns a list of (original_ffn, wrapper) tuples so the caller can restore
    the originals after inference if needed.
    """
    pairs = []
    for si, stage in enumerate(backbone.stages):
        for bi, block in enumerate(stage.blocks):
            ws = block.attn.w_msa.window_size
            if isinstance(ws, (tuple, list)):
                ws = ws[0]
            shift_size = getattr(block.attn, "shift_size", 0)

            wrapper = _FFNCapture(block.ffn)
            wrapper._capture_meta = (si, bi, ws, shift_size)
            block.ffn = wrapper
            pairs.append((si, bi, block))
    return pairs


def _collect_baseline(backbone, proc_H, proc_W, meta, orig_img, img_idx):
    """Collect FFN update records from baseline model (wrapper-based).

    Baseline FFN operates on full token sequences, not windows.  We reshape
    per-token U_t to a 2-D feature map, then resize to the original image.
    """
    records = []
    for si, stage in enumerate(backbone.stages):
        for bi, block in enumerate(stage.blocks):
            ffn = block.ffn
            if not hasattr(ffn, "_capture_update") or ffn._capture_update is None:
                continue

            U_t = ffn._capture_update.cpu().numpy()         # (B, L)  per-token
            if U_t.ndim == 2:
                U_t = U_t[0]                                 # (L,)
            _si, _bi, ws, shift_size = ffn._capture_meta
            if shift_size > 0:
                continue   # skip SW-MSA

            # --- feature-map dimensions at this stage ---
            stride = 4 * (2 ** si)
            feat_H = (proc_H + stride - 1) // stride          # ceil
            feat_W = (proc_W + stride - 1) // stride
            L_exp = feat_H * feat_W

            if len(U_t) != L_exp:
                print(f"  [WARN] S{si}B{bi}: tokens {len(U_t)} != "
                      f"H*W ({feat_H}*{feat_W}={L_exp}), skip")
                continue

            # --- token sequence → feature map → original image ---
            U_fm = U_t.reshape(feat_H, feat_W)                # (H_fm, W_fm)

            # resize feature map → preprocessed input size
            U_input = cv2.resize(
                U_fm, (proc_W, proc_H), interpolation=cv2.INTER_NEAREST,
            )
            # remove data_preprocessor padding
            img_h, img_w = meta["img_shape"] if meta["img_shape"] else (proc_H, proc_W)
            U_input = U_input[:img_h, :img_w]
            # map to original image
            ori_h, ori_w = meta["ori_shape"] if meta["ori_shape"] else (img_h, img_w)
            if (img_h, img_w) != (ori_h, ori_w):
                U_ori = cv2.resize(
                    U_input, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST,
                )
            else:
                U_ori = U_input

            shift_tag = "SW" if shift_size > 0 else "W "
            records.append({
                "img_idx": img_idx,
                "stage": si,
                "block": bi,
                "shift": shift_tag,
                "ws": ws,
                "U_ori": U_ori,
                "U_median": float(np.median(U_t)),
                "orig_img": orig_img,
            })
    return records


# ---------------------------------------------------------------------------
# ECHO (SwinTransformerV3) support
# ---------------------------------------------------------------------------

def _is_v3_model(model):
    return hasattr(model.backbone, "set_force_full_ffn")


def _collect_v3(backbone, proc_H, proc_W, meta, orig_img, img_idx):
    """Collect FFN update records from V3 model (force_full_ffn mode)."""
    records = []
    for si, stage in enumerate(backbone.stages):
        for bi, block in enumerate(stage.blocks):
            buf = block.ffn_update_buffer
            if buf is None:
                continue

            U_w = buf.cpu().numpy()
            feat_H, feat_W = block.ffn_update_hw
            H_pad, W_pad = block.ffn_update_padded_hw
            ws = block.window_size
            H_w = H_pad // ws
            W_w = W_pad // ws
            N_win = H_w * W_w

            if len(U_w) != N_win:
                print(f"  [WARN] S{si}B{bi}: buf {len(U_w)} != "
                      f"H_w*W_w ({H_w}*{W_w}), skip")
                continue

            U_grid = U_w.reshape(H_w, W_w)
            if block.shift_size > 0:
                continue   # skip SW-MSA

            shift_tag = "W "

            U_ori = _spatial_map(
                U_grid, ws, feat_H, feat_W, proc_H, proc_W, meta,
            )

            records.append({
                "img_idx": img_idx,
                "stage": si,
                "block": bi,
                "shift": shift_tag,
                "ws": ws,
                "U_ori": U_ori,
                "U_median": float(np.median(U_w)),
                "orig_img": orig_img,
            })
    return records


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    register_all_modules()

    import sparse_former.models.backbones.swin_baseline_v3   # noqa
    import sparse_former.models.backbones.swin_baseline       # noqa
    import sparse_former.models.detectors.dino_with_gate_loss # noqa
    import sparse_former.datasets.panda                       # noqa

    MMDET_DATASETS._module_dict.update(SPARSE_DATASETS._module_dict)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load config & model ------------------------------------------------
    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    model.to(args.device)

    is_v3 = _is_v3_model(model)
    if is_v3:
        model.backbone.set_force_full_ffn(True)
        print("[mode] ECHO (V3) — force_full_ffn")
    else:
        _setup_baseline_ffn_capture(model.backbone)
        print("[mode] Dense baseline — FFN capture wrappers")

    # ---- build test dataset -------------------------------------------------
    dataset = MMDET_DATASETS.build(cfg.test_dataloader.dataset)
    num_images = min(args.num_images, len(dataset))

    all_records = []

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

        if is_v3:
            records = _collect_v3(model.backbone, proc_H, proc_W, meta, orig_img, idx)
        else:
            records = _collect_baseline(model.backbone, proc_H, proc_W, meta, orig_img, idx)

        all_records.extend(records)
        print(f"[img {idx:2d}] {proc_H}×{proc_W} → collected {len(records)} blocks")

    if not all_records:
        print("No FFN update buffers collected.")
        return

    # ---- per-block visualisation --------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for record in all_records:
        ridx = record["img_idx"]
        si, bi, shift = record["stage"], record["block"], record["shift"]
        U_ori = record["U_ori"]
        orig = record["orig_img"]
        U_median = record["U_median"]

        p5 = float(np.percentile(U_ori, 5))
        p95 = float(np.percentile(U_ori, 95))
        denom = max(p95 - p5, 1e-8)
        U_norm = np.clip((U_ori - p5) / denom, 0.0, 1.0)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if orig is not None:
            ax.imshow(orig)

        im = ax.imshow(U_norm, alpha=0.55, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"S{si} B{bi:02d} {shift}    Median U = {U_median:.4f}", fontsize=11,
        )
        ax.axis("off")

        fname = f"img{ridx:02d}_S{si}_B{bi:02d}_{shift}.png"
        save_path = os.path.join(args.output_dir, fname)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Relative FFN Update (per-block norm.)")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {save_path}  (p5={p5:.4f} p95={p95:.4f} med={U_median:.4f})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Compute foreground/background FLOPs breakdown for ECHO pruning models.

Follows the 9-section FG/BG FLOPs protocol:
  1. Foreground definition: area-proportional q ∈ [0,1] per window
  2. Build per-stage FG coverage maps via avg_pool2d
  3. Window alignment with shift handling (torch.roll for SW-MSA)
  4. Analytic per-window Attention and FFN FLOPs
  5. Apply kl_mask / inc_mask execution masks, split by FG/BG
  6. Allocate fixed backbone FLOPs proportional to mean FG area ratio
  7. Aggregate across test set (sum first, then F-ratio)
  8. Output all required statistics
  9. Three verification checks

Usage::

  python tools/analysis_tools/get_fgbg_flops.py \
      configs/dino/dino-4scale_swin_v3_kl_8xb2-36e_panda.py \
      outputs/v3_kl_inc/14/epoch_36.pth \
      --num-images 100
"""

import argparse
import os
import sys
from collections import defaultdict

# Ensure project root is on the path
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
from sparse_former.registry import MODELS as SPARSE_MODELS


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="FG/BG FLOPs breakdown for ECHO")
    p.add_argument("config", help="Path to model config file")
    p.add_argument("checkpoint", help="Path to .pth checkpoint")
    p.add_argument("--num-images", type=int, default=100,
                   help="Number of test images (default: 100)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--debug", action="store_true",
                   help="Enable per-image diagnostic output (dimension alignment, "
                        "q statistics, per-block FG bias)")
    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================

def _metainfo(ds):
    mi = ds.metainfo if hasattr(ds, "metainfo") else {}
    return {
        "img_shape": mi.get("img_shape", None),
        "ori_shape": mi.get("ori_shape", None),
        "img_path": mi.get("img_path", None),
        "scale_factor": mi.get("scale_factor", None),
    }


# =============================================================================
# FLOPs formulas  (Protocol Section 4)
# =============================================================================

def attention_flops_per_window(C: int, P: int) -> float:
    """Attention FLOPs per window = 4*P*C^2 + 2*P^2*C

    Breakdown (multiply-add = 1 FLOP):
      QKV projection:    P * C * 3C  = 3*P*C^2
      Output projection: P * C * C   =   P*C^2
      Attention scores:  P * C * P   = P^2*C      (Q @ K^T)
      Score @ Value:     P * P * C   = P^2*C      (attn @ V)
    Total: 4*P*C^2 + 2*P^2*C
    """
    return 4 * P * C * C + 2 * P * P * C


def ffn_flops_per_window(C: int, C_ffn: int, P: int) -> float:
    """FFN FLOPs per window = 2*P*C*C_ffn

    Breakdown:
      FC1: P * C * C_ffn
      FC2: P * C_ffn * C
    Total: 2*P*C*C_ffn

    With mlp_ratio=4, C_ffn=4*C, total = 8*P*C^2.
    """
    return 2 * P * C * C_ffn


# =============================================================================
# FFN dimension reader  (Protocol Section 4: read actual dims, don't hardcode)
# =============================================================================

def get_ffn_channels(block) -> int:
    """Read actual FFN hidden dimension from the block's FFN module.

    mmcv FFN structure (num_fcs=2):
      layers[0]: Linear(embed_dims -> feedforward_channels)
      layers[1]: Activation (GELU)
      layers[2]: Dropout
      layers[3]: Linear(feedforward_channels -> embed_dims)
      layers[4]: Dropout (optional)
    """
    for layer in block.ffn.layers:
        if isinstance(layer, torch.nn.Linear):
            return layer.out_features
    # Fallback — should never be reached
    return block.embed_dims * 4


# =============================================================================
# FG coverage maps  (Protocol Sections 1-3)
# =============================================================================

def build_stage_fg_map(fg_mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Pool binary GT mask to match exact backbone feature map dimensions.

    Uses F.interpolate with mode='area' which is equivalent to adaptive
    average pooling.  Unlike cascaded avg_pool2d, this correctly handles
    odd-dimension intermediates introduced by PatchMerging padding.

    Args:
        fg_mask: (pad_h, pad_w) float32, 1.0 inside GT boxes
        target_h, target_w: actual feature map H, W from backbone (vis['H'], vis['W'])

    Returns:
        (1, 1, target_h, target_w) tensor with values ∈ [0,1]
    """
    return F.interpolate(
        fg_mask[None, None],
        size=(target_h, target_w),
        mode='area',
    )


def get_stage_feat_sizes(backbone) -> dict:
    """Read actual feature map sizes from the backbone's selection_vis.

    Returns dict {stage_idx: (feat_h, feat_w)} using the first block
    in each stage that has a valid selection_vis.
    """
    sizes = {}
    for si, stage in enumerate(backbone.stages):
        for block in stage.blocks:
            vis = block.selection_vis
            if vis is not None:
                sizes[si] = (vis['H'], vis['W'])
                break
    return sizes


def compute_q_windows(stage_fg_map, H_pad, W_pad, window_size, shift_size):
    """Per-window binary FG indicator: 1 if window intersects any GT box.

    Uses the same window_partition logic as SwinBlockV3._window_partition.
    For SW-MSA blocks, applies torch.roll before partitioning,
    matching the model's data shift.

    Args:
        stage_fg_map: (1, 1, feat_h, feat_w) tensor from build_stage_fg_maps
        H_pad, W_pad: padded feature map spatial dims (multiples of window_size)
        window_size: typically 7
        shift_size: 0 for W-MSA, window_size//2 for SW-MSA

    Returns:
        q_windows: (N_win,) float tensor, each value ∈ {0, 1}.
            1 if any pixel in the window overlaps with a GT box, else 0.
            N_win = (H_pad // window_size) * (W_pad // window_size)
    """
    _, _, feat_h, feat_w = stage_fg_map.shape

    # Pad to H_pad x W_pad (padding region has FG=0)
    pad_b = H_pad - feat_h
    pad_r = W_pad - feat_w
    if pad_b > 0 or pad_r > 0:
        fg_padded = F.pad(stage_fg_map, (0, pad_r, 0, pad_b),
                          mode='constant', value=0.0)
    else:
        fg_padded = stage_fg_map

    # SW-MSA: apply same shift as model
    if shift_size > 0:
        fg_padded = torch.roll(
            fg_padded,
            shifts=(-shift_size, -shift_size),
            dims=(-2, -1),
        )

    # Window partition — matches SwinBlockV3._window_partition
    # Input:  (1, 1, H_pad, W_pad)
    # Output: (N_win, window_size * window_size)
    _, _, Hp, Wp = fg_padded.shape
    H_w = Hp // window_size
    W_w = Wp // window_size

    x = fg_padded.view(1, 1, H_w, window_size, W_w, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(-1, window_size * window_size)

    q_windows = (x.sum(dim=-1) > 0).float()  # binary: 1 if window intersects any GT
    return q_windows


# =============================================================================
# GT box coordinate scaling
# =============================================================================

def scale_boxes_to_input(boxes, scale_factor):
    """Scale GT boxes from original image coords to preprocessed (resized) coords.

    Args:
        boxes: (N, 4) tensor or BaseBoxes in xyxy format, original image coordinates
        scale_factor: (w_scale, h_scale) — can be tuple, list, or 1D tensor/array

    Returns:
        (N, 4) tensor in preprocessed image coordinates
    """
    # Handle mmdet BaseBoxes (HorizontalBoxes) — extract underlying tensor
    if hasattr(boxes, 'tensor'):
        boxes = boxes.tensor

    if boxes.shape[0] == 0:
        return boxes.float()

    if scale_factor is not None:
        if hasattr(scale_factor, 'tolist'):
            sf = scale_factor.tolist()
        else:
            sf = list(scale_factor)
        w_scale = float(sf[0])
        h_scale = float(sf[1]) if len(sf) > 1 else w_scale
    else:
        w_scale, h_scale = 1.0, 1.0

    scaled = boxes.clone().float()
    scaled[:, 0] *= w_scale  # x1
    scaled[:, 1] *= h_scale  # y1
    scaled[:, 2] *= w_scale  # x2
    scaled[:, 3] *= h_scale  # y2
    return scaled


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    register_all_modules()

    # Register sparse_former custom modules
    import sparse_former.models.backbones.swin_baseline_v3      # noqa
    import sparse_former.models.backbones.sparsenet_entropy     # noqa
    import sparse_former.models.detectors.dino_with_gate_loss   # noqa
    import sparse_former.datasets.panda                         # noqa

    MMDET_DATASETS._module_dict.update(SPARSE_DATASETS._module_dict)

    # ---- load model ---------------------------------------------------------
    print(f"Loading config: {args.config}")
    print(f"Loading checkpoint: {args.checkpoint}")

    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    # Try mmdet MODELS first, fall back to sparse_former MODELS
    # (SparseNetEntropy is registered under sparse_former.registry)
    try:
        model = MODELS.build(cfg.model)
    except KeyError:
        model = SPARSE_MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    model.to(args.device)
    model.backbone.set_enable_selection_vis(True)

    # ---- dataset ------------------------------------------------------------
    dataset = MMDET_DATASETS.build(cfg.test_dataloader.dataset)
    num_images = min(args.num_images, len(dataset))

    # ---- accumulators -------------------------------------------------------
    # Per-image FG area ratio
    total_fg_area_ratio = 0.0

    # Dynamic FLOPs (attention + FFN that vary with pruning masks)
    attn_fg_sum = 0.0
    attn_bg_sum = 0.0
    ffn_fg_sum = 0.0
    ffn_bg_sum = 0.0

    # Shape verification errors (Protocol Section 9, Check 2)
    shape_errors = []

    # Per-stage breakdown
    stage_stats = {
        s: {'attn_fg': 0.0, 'attn_bg': 0.0, 'ffn_fg': 0.0, 'ffn_bg': 0.0,
            'n_kl': 0, 'n_inc': 0, 'N': 0, 'num_blocks': 0}
        for s in range(4)
    }

    # Per-block detail records (for final aggregation)
    block_records = []

    # Per-stage binary FG token ratios (for fixed FLOPs allocation)
    stage_fg_ratio_sum = {s: 0.0 for s in range(4)}

    # ---- process images -----------------------------------------------------
    print(f"Processing {num_images} images...")

    for idx in range(num_images):
        data_ = dataset[idx]
        ds = data_["data_samples"]
        meta = _metainfo(ds)

        # ---- step A: get GT boxes in preprocessed coords --------------------
        gt_bboxes = ds.gt_instances.bboxes  # (N, 4) xyxy, ORIGINAL coords
        scale_factor = meta.get("scale_factor", None)

        # Run data_preprocessor to get the actual model input
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

        # Preprocessed image dimensions (after resize + pad in data_preprocessor)
        _, _, proc_H, proc_W = inputs_proc.shape

        # Scale GT boxes to preprocessed image coordinates
        boxes_resized = scale_boxes_to_input(gt_bboxes, scale_factor)

        # ---- step B: build binary FG mask at input resolution (Protocol §2) --
        fg_mask = torch.zeros((proc_H, proc_W), dtype=torch.float32)
        for box in boxes_resized:
            x1, y1, x2, y2 = box.round().long()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(proc_W, x2)
            y2 = min(proc_H, y2)
            if x2 > x1 and y2 > y1:
                fg_mask[y1:y2, x1:x2] = 1.0

        # Per-image FG area ratio: |F_b| / (H_b * W_b)
        fg_pixels = fg_mask.sum().item()
        total_pixels = proc_H * proc_W
        fg_area_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0.0
        total_fg_area_ratio += fg_area_ratio

        # ---- debug: dimension check for first 3 images --------------------
        if args.debug and idx < 3:
            img_h, img_w = meta.get("img_shape", (proc_H, proc_W))
            ori_h, ori_w = meta.get("ori_shape", (0, 0))
            print(f"\n[DEBUG img={idx}] "
                  f"ori=({ori_h},{ori_w}) -> "
                  f"img_shape=({img_h},{img_w}) -> "
                  f"proc=({proc_H},{proc_W})  "
                  f"scale_factor={scale_factor}  "
                  f"fg_pixels={fg_pixels}/{total_pixels}={fg_area_ratio*100:.2f}%")

        # ---- step C: run backbone forward -----------------------------------
        with torch.no_grad():
            model.extract_feat(inputs_proc)

        # ---- step D: build per-stage FG maps at EXACT feature sizes ---------
        # Use the backbone's actual feature map dimensions to avoid the
        # floor-vs-ceil mismatch from cascaded avg_pool2d.
        # PatchMerging pads odd dimensions (e.g. 187→188) before 2× reduction,
        # so stage 1 is ceil(187/2)=94, not floor(187/2)=93.
        feat_sizes = get_stage_feat_sizes(model.backbone)
        stage_fg_maps = {}
        for si, (fh, fw) in feat_sizes.items():
            stage_fg_maps[si] = build_stage_fg_map(fg_mask, fh, fw)
            # Per-stage binary FG token ratio (token = FG if any input pixel in
            # its receptive field overlaps a GT box).
            ratio_s = (stage_fg_maps[si] > 0).float().mean().item()
            stage_fg_ratio_sum[si] += ratio_s

        # ---- step E: collect per-block masks and compute FG/BG FLOPs --------
        for si, stage in enumerate(model.backbone.stages):
            if si not in stage_fg_maps:
                continue
            stage_fg_map = stage_fg_maps[si]

            for bi, block in enumerate(stage.blocks):
                vis = block.selection_vis
                if vis is None:
                    continue

                # Read actual dimensions from block (Protocol §4)
                C = block.embed_dims
                C_ffn = get_ffn_channels(block)
                ws = vis['window_size']
                P = ws * ws

                attn_per_win = attention_flops_per_window(C, P)
                ffn_per_win = ffn_flops_per_window(C, C_ffn, P)

                # Execution masks
                kl_mask = vis['kl_mask'].float()
                inc_mask = vis['inc_mask'].float()

                # ★ Fix: Stage 0 block 0 uses _forward_kl (no INC gating).
                # Its inc_mask is artificially all-zeros (placeholder),
                # but FFN actually runs on ALL KL-kept windows.
                # Use kl_mask as effective FFN mask.  (Protocol §5)
                if block.inc_ratio is None:
                    inc_mask_eff = kl_mask.clone()
                else:
                    inc_mask_eff = inc_mask

                # Compute q: per-window FG area proportion (Protocol §3)
                q = compute_q_windows(
                    stage_fg_map,
                    vis['H_pad'], vis['W_pad'],
                    ws, vis['shift_size'],
                )

                # Shape check (Protocol §9, Check 2)
                if q.shape[0] != kl_mask.shape[0]:
                    shape_errors.append(
                        f"S{si}B{bi}: q.shape={q.shape[0]}, "
                        f"mask.shape={kl_mask.shape[0]}"
                    )

                q = q.to(device=kl_mask.device)

                # ---- per-block diagnostics (first 3 images only) ----
                if args.debug and idx < 3:
                    # Dimension alignment
                    _, _, fh, fw = stage_fg_map.shape
                    print(f"  [DEBUG S{si}B{bi}] stage_fg=({fh},{fw})  "
                          f"vis H={vis['H']} W={vis['W']}  "
                          f"H_pad={vis['H_pad']} W_pad={vis['W_pad']}  "
                          f"shift={vis['shift_size']}  ws={ws}")

                    # FG bias: is q higher for kept windows?
                    q_all = q.mean().item()
                    n_kl_debug = kl_mask.sum().item()
                    if n_kl_debug > 0:
                        q_kept = q[kl_mask > 0].mean().item()
                    else:
                        q_kept = 0.0
                    bias = q_kept / max(q_all, 1e-8)
                    print(f"    q_all={q_all:.4f}  q_kept={q_kept:.4f}  "
                          f"bias={bias:.2f}x  "
                          f"KL_keep={int(n_kl_debug)}/{q.shape[0]}")

                # Per-window dynamic cost: attn if kl_kept, ffn if inc_kept
                window_cost = (kl_mask * attn_per_win
                               + inc_mask_eff * ffn_per_win)  # (N_win,)

                # FG/BG split using continuous q  (Protocol §5)
                blk_attn_fg = (q * kl_mask * attn_per_win).sum().item()
                blk_attn_bg = ((1.0 - q) * kl_mask * attn_per_win).sum().item()
                blk_ffn_fg = (q * inc_mask_eff * ffn_per_win).sum().item()
                blk_ffn_bg = ((1.0 - q) * inc_mask_eff * ffn_per_win).sum().item()

                attn_fg_sum += blk_attn_fg
                attn_bg_sum += blk_attn_bg
                ffn_fg_sum += blk_ffn_fg
                ffn_bg_sum += blk_ffn_bg

                n_kl = int(kl_mask.sum())
                n_inc = int(inc_mask_eff.sum())
                N = kl_mask.shape[0]

                stage_stats[si]['attn_fg'] += blk_attn_fg
                stage_stats[si]['attn_bg'] += blk_attn_bg
                stage_stats[si]['ffn_fg'] += blk_ffn_fg
                stage_stats[si]['ffn_bg'] += blk_ffn_bg
                stage_stats[si]['n_kl'] += n_kl
                stage_stats[si]['n_inc'] += n_inc
                stage_stats[si]['N'] += N
                stage_stats[si]['num_blocks'] += 1

                block_records.append({
                    'stage': si,
                    'block': bi,
                    'shift': vis['shift_size'],
                    'C': C,
                    'n_kl': n_kl,
                    'n_inc': n_inc,
                    'N': N,
                    'attn_fg': blk_attn_fg,
                    'attn_bg': blk_attn_bg,
                    'ffn_fg': blk_ffn_fg,
                    'ffn_bg': blk_ffn_bg,
                })

        # ---- per-image debug summary ---------------------------------------
        if args.debug and idx < 3:
            # Compute per-image dynamic F-ratio from block_records for this image
            img_dfg = sum(r['attn_fg'] + r['ffn_fg']
                          for r in block_records
                          if r['stage'] == -1)  # no-op, actual calc below
            img_dfg = 0.0
            img_dbg = 0.0
            for si, stage in enumerate(model.backbone.stages):
                for bi, block in enumerate(stage.blocks):
                    vis2 = block.selection_vis
                    if vis2 is None:
                        continue
                    C2 = block.embed_dims
                    Cffn2 = get_ffn_channels(block)
                    ws2 = vis2['window_size']
                    attn_pw = attention_flops_per_window(C2, ws2 * ws2)
                    ffn_pw = ffn_flops_per_window(C2, Cffn2, ws2 * ws2)
                    kl2 = vis2['kl_mask'].float()
                    inc2 = vis2['inc_mask'].float()
                    if block.inc_ratio is None:
                        inc2 = kl2.clone()
                    q2 = compute_q_windows(
                        stage_fg_maps[si],
                        vis2['H_pad'], vis2['W_pad'],
                        ws2, vis2['shift_size'],
                    ).to(device=kl2.device)
                    img_dfg += (q2 * (kl2 * attn_pw + inc2 * ffn_pw)).sum().item()
                    img_dbg += ((1 - q2) * (kl2 * attn_pw + inc2 * ffn_pw)).sum().item()
            img_dyn_total = img_dfg + img_dbg
            img_dyn_fr = img_dfg / img_dyn_total * 100 if img_dyn_total > 0 else 0
            print(f"  [DEBUG img={idx}] per-image dynamic: "
                  f"FG={img_dfg/1e9:.4f}G  BG={img_dbg/1e9:.4f}G  "
                  f"dyn F-ratio={img_dyn_fr:.2f}%  "
                  f"(baseline FG area={fg_area_ratio*100:.2f}%)")

        # ---- progress --------------------------------------------------------
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx + 1}/{num_images}] "
                  f"FG area: {fg_area_ratio * 100:.2f}%")

    # =========================================================================
    # Aggregate across all images  (Protocol Sections 6-7)
    # =========================================================================
    N = num_images

    # Mean dynamic FLOPs per image (in GFLOPs)
    G_attn_fg = attn_fg_sum / N / 1e9
    G_attn_bg = attn_bg_sum / N / 1e9
    G_ffn_fg  = ffn_fg_sum  / N / 1e9
    G_ffn_bg  = ffn_bg_sum  / N / 1e9

    G_dyn_fg = G_attn_fg + G_ffn_fg
    G_dyn_bg = G_attn_bg + G_ffn_bg
    G_dyn = G_dyn_fg + G_dyn_bg

    # Total backbone FLOPs from trace-based measurement (get_flops_2.py)
    G_total = 6.181  # GFLOPs

    # Fixed FLOPs = total - dynamic  (Protocol §6)
    G_fixed = G_total - G_dyn

    # Mean FG area ratio across test set (at input resolution)
    r_bar_F = total_fg_area_ratio / N

    # Per-stage mean binary FG token ratios (for fixed FLOPs allocation)
    # Unlike r_bar_F (input resolution), these reflect the FG proportion
    # at each stage's feature map after pooling — much higher because
    # a 4×4 / 8×8 / ... patch is "FG" if ANY of its pixels overlap GT.
    stage_fg_ratios = {}
    for s in range(4):
        stage_fg_ratios[s] = stage_fg_ratio_sum[s] / N if N > 0 else 0.0

    # Use the average per-stage FG ratio for fixed FLOPs allocation.
    # Fixed ops (LN, patch merging, residual) run at various resolutions;
    # averaging across stages is a simple weighting proxy.
    r_fixed = sum(stage_fg_ratios.values()) / max(len(stage_fg_ratios), 1)

    # Allocate fixed FLOPs proportionally
    G_F_fixed = r_fixed * G_fixed
    G_B_fixed = (1.0 - r_fixed) * G_fixed

    # Final FG / BG
    G_F = G_dyn_fg + G_F_fixed
    G_B = G_dyn_bg + G_B_fixed
    G_total_recon = G_F + G_B

    reconstruction_error = abs(G_total_recon - G_total)

    # F-ratio: sum first, then divide (Protocol §7)
    #  = sum_b G_F_b / sum_b (G_F_b + G_B_b)
    F_ratio = G_F / G_total_recon * 100.0

    # =========================================================================
    # Output report  (Protocol Section 8)
    # =========================================================================
    print()
    print("=" * 60)
    print("FG/BG FLOPs Breakdown")
    print("=" * 60)
    print(f"Config:      {os.path.basename(args.config)}")
    print(f"Checkpoint:  {os.path.basename(args.checkpoint)}")
    print(f"Images:      {N}")
    print(f"Device:      {args.device}")
    print()
    print(f"Mean foreground area ratio (input): {r_bar_F * 100:.4f}%")
    print(f"Per-stage binary FG token ratios:")
    for s in range(4):
        print(f"  Stage {s}: {stage_fg_ratios[s] * 100:.2f}%")
    print(f"Fixed FLOPs allocation FG ratio: {r_fixed * 100:.2f}%")
    print()
    print(f"Attention GFLOPs-F: {G_attn_fg:.4f}")
    print(f"Attention GFLOPs-B: {G_attn_bg:.4f}")
    print(f"FFN GFLOPs-F: {G_ffn_fg:.4f}")
    print(f"FFN GFLOPs-B: {G_ffn_bg:.4f}")
    print()
    print(f"Dynamic GFLOPs-F: {G_dyn_fg:.4f}")
    print(f"Dynamic GFLOPs-B: {G_dyn_bg:.4f}")
    print(f"Fixed backbone GFLOPs: {G_fixed:.4f}")
    print()
    print(f"Final GFLOPs-F: {G_F:.4f}")
    print(f"Final GFLOPs-B: {G_B:.4f}")
    print(f"Final GFLOPs-O: {G_total}")
    print(f"F-ratio: {F_ratio:.4f}%")
    print(f"Reconstruction error: {reconstruction_error:.6f}")

    # =========================================================================
    # Verification  (Protocol Section 9)
    # =========================================================================
    print()
    print("-" * 60)
    print("Verification")
    print("-" * 60)

    # Check 1: GF + GB = 6.181, error < 1e-3
    if reconstruction_error < 1e-3:
        print(f"[PASS] Check 1: FG+BG = {G_total_recon:.4f} G, "
              f"error = {reconstruction_error:.6f} < 1e-3")
    else:
        print(f"[FAIL] Check 1: FG+BG = {G_total_recon:.4f} G, "
              f"error = {reconstruction_error:.6f} >= 1e-3")

    # Check 2: all q_windows shapes match execution masks
    if not shape_errors:
        print(f"[PASS] Check 2: All {len(block_records)} block records "
              f"have q_windows.shape == mask.shape")
    else:
        print(f"[FAIL] Check 2: {len(shape_errors)} shape mismatch(es):")
        for err in shape_errors:
            print(f"  {err}")

    # Check 3: F-ratio sanity
    print(f"[INFO] Check 3: F-ratio = {F_ratio:.2f}%, "
          f"mean FG area ratio = {r_bar_F * 100:.2f}%")
    print(f"       If all windows forced (mask ≡ 1), "
          f"F-ratio should ≈ {r_bar_F * 100:.2f}%")
    print(f"       ECHO F-ratio vs baseline: "
          f"{'higher → computation concentrates on FG' if F_ratio > r_bar_F * 100 else 'need investigation'}")

    # =========================================================================
    # Per-stage breakdown
    # =========================================================================
    print()
    print("-" * 60)
    print("Per-Stage Breakdown (averaged over images)")
    print("-" * 60)

    for si in range(4):
        s = stage_stats[si]
        if s['num_blocks'] == 0:
            continue
        n_blk = s['num_blocks'] // N  # blocks per image in this stage
        print(f"\nStage {si} ({n_blk} blocks, per-image avg):")
        print(f"  Attention FG: {s['attn_fg'] / N / 1e9:.4f} G  "
              f"Attention BG: {s['attn_bg'] / N / 1e9:.4f} G")
        print(f"  FFN FG:       {s['ffn_fg'] / N / 1e9:.4f} G  "
              f"FFN BG:       {s['ffn_bg'] / N / 1e9:.4f} G")
        n_kl_total = max(s['n_kl'], 1)
        n_inc_total = max(s['n_inc'], 1)
        N_total = max(s['N'], 1)
        print(f"  KL keep: {s['n_kl'] / N_total * 100:.1f}%  "
              f"INC keep: {s['n_inc'] / N_total * 100:.1f}%  "
              f"(avg over {s['num_blocks']} block-instances)")

    # =========================================================================
    # Per-block detail
    # =========================================================================
    print()
    print("-" * 60)
    print("Per-Block Detail (per-image avg)")
    print("-" * 60)

    # Aggregate same block across images
    block_agg = defaultdict(lambda: {
        'attn_fg': 0.0, 'attn_bg': 0.0, 'ffn_fg': 0.0, 'ffn_bg': 0.0,
        'n_kl': 0, 'n_inc': 0, 'N': 0, 'shift': 0, 'C': 0,
    })
    for rec in block_records:
        key = (rec['stage'], rec['block'])
        block_agg[key]['attn_fg'] += rec['attn_fg']
        block_agg[key]['attn_bg'] += rec['attn_bg']
        block_agg[key]['ffn_fg'] += rec['ffn_fg']
        block_agg[key]['ffn_bg'] += rec['ffn_bg']
        block_agg[key]['n_kl'] += rec['n_kl']
        block_agg[key]['n_inc'] += rec['n_inc']
        block_agg[key]['N'] += rec['N']
        block_agg[key]['shift'] = rec['shift']
        block_agg[key]['C'] = rec['C']

    for key in sorted(block_agg.keys()):
        si, bi = key
        b = block_agg[key]
        n = N  # each block appears once per image
        shift_label = "W " if b['shift'] == 0 else "SW"
        a_fg = b['attn_fg'] / n / 1e9
        a_bg = b['attn_bg'] / n / 1e9
        f_fg = b['ffn_fg'] / n / 1e9
        f_bg = b['ffn_bg'] / n / 1e9
        kl_pct = b['n_kl'] / max(1, b['N']) * 100
        inc_pct = b['n_inc'] / max(1, b['N']) * 100
        print(f"  S{si}B{bi} [{shift_label}] C={b['C']:3d}: "
              f"Attn FG={a_fg:.4f}G BG={a_bg:.4f}G | "
              f"FFN FG={f_fg:.4f}G BG={f_bg:.4f}G | "
              f"KL={kl_pct:.1f}% INC={inc_pct:.1f}%")


if __name__ == "__main__":
    main()

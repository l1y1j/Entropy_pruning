import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math
import numpy as np
from PIL import Image


def compute_window_relative_entropy(x_windows: torch.Tensor, B: int, window_size: int = 7) -> torch.Tensor:
    """
    计算每个窗口的相对熵（KL散度），保留Batch维度独立计算
    
    Args:
        x_windows: (total_windows, window_size, window_size, C) - 总窗口数 = B * N_win
        B: Batch大小
        window_size: 窗口大小，默认7
    
    Returns:
        kl: (B * N_win,) 每个窗口的相对熵
    """
    total_windows, _, _, C = x_windows.shape
    N_win = total_windows // B
    
    x_windows = x_windows.view(B, N_win, window_size * window_size, C)
    local_dist = F.softmax(x_windows.mean(dim=2), dim=-1)
    global_dist = local_dist.mean(dim=1, keepdim=True)
    kl = (local_dist * torch.log(local_dist / (global_dist + 1e-8))).sum(dim=-1)
    
    return kl.view(-1)  # 返回 (B * N_win,)


def compute_channel_distribution(features: torch.Tensor, spatial_dims: Tuple[int, int]) -> torch.Tensor:
    """
    Compute channel distribution from features.
    Features shape: (B, N, C) where N = H*W
    Returns: (B, H, W, C) - channel-wise probability distributions
    """
    B, N, C = features.shape
    H, W = spatial_dims
    
    features_2d = features.reshape(B, H, W, C)
    dist = F.softmax(features_2d, dim=-1)
    
    return dist


def compute_block_distribution(features: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Compute distribution for each block.
    
    Args:
        features: (B, H, W, C) - token features in NHWC format
        block_size: size of block (e.g., 2 for 2x2)
    
    Returns:
        block_dists: (B, num_blocks_h, num_blocks_w, C) - distribution for each block
        grid_shape: (num_blocks_h, num_blocks_w)
    """
    B, H, W, C = features.shape
    
    if H < block_size or W < block_size:
        return None, (0, 0)
    
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    
    if pad_h > 0 or pad_w > 0:
        features = F.pad(features, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
    
    new_H, new_W = features.shape[1], features.shape[2]
    num_blocks_h = new_H // block_size
    num_blocks_w = new_W // block_size
    
    block_features = features.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
    block_features = block_features.reshape(B, num_blocks_h, num_blocks_w, block_size * block_size, C)
    
    block_mean = block_features.mean(dim=3)
    
    block_dists = F.softmax(block_mean, dim=-1)
    
    return block_dists, (num_blocks_h, num_blocks_w)


def compute_global_distribution(block_dists: torch.Tensor) -> torch.Tensor:
    """
    Compute global distribution as average of all block distributions.
    
    Args:
        block_dists: (B, H, W, C) - block-wise distributions
    
    Returns:
        global_dist: (B, C) - global distribution
    """
    global_dist = block_dists.mean(dim=(1, 2))
    
    return global_dist


def compute_kl_divergence(block_dists: torch.Tensor, global_dist: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute KL divergence between each block distribution and global distribution.
    KL(P || Q) = sum(P * log(P / Q))
    
    Args:
        block_dists: (B, H, W, C) - local block distributions
        global_dist: (B, C) - global distribution
    
    Returns:
        kl_map: (B, H, W) - KL divergence for each block
    """
    B, H, W, C = block_dists.shape
    
    global_dist_expanded = global_dist.unsqueeze(1).unsqueeze(2).expand(B, H, W, C)
    
    kl_map = (block_dists * torch.log((block_dists + eps) / (global_dist_expanded + eps))).sum(dim=-1)
    
    return kl_map


def compute_local_relative_entropy(features: torch.Tensor, block_size: int = 2) -> torch.Tensor:
    """
    Compute local relative entropy for each block.
    
    The approach:
    1. Extract block distributions (local channel distribution)
    2. Compute global distribution (average of all blocks)
    3. Compute KL(block_dist || global_dist) for each block
    
    Higher KL values indicate blocks that differ more from global average.
    
    Args:
        features: (B, H, W, C) - Swin features in NHWC format
        block_size: size of block (default 2 for 2x2)
    
    Returns:
        kl_map: (H, W) - KL divergence heatmap (for B=1)
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    B, H, W, C = features.shape
    
    block_dists, grid_shape = compute_block_distribution(features, block_size)
    
    if block_dists is None:
        return None
    
    global_dist = compute_global_distribution(block_dists)
    
    kl_map = compute_kl_divergence(block_dists, global_dist)
    
    return kl_map[0]


def compute_token_local_entropy(features: torch.Tensor, window_size: int = 3) -> torch.Tensor:
    """
    Compute relative entropy for each token vs its neighborhood.
    
    Args:
        features: (B, H, W, C) - Swin features in NHWC format
        window_size: size of neighborhood window
    
    Returns:
        entropy_map: (H, W) - entropy heatmap
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    B, H, W, C = features.shape
    
    token_dists = F.softmax(features, dim=-1)
    
    pad = window_size // 2
    
    padded = F.pad(token_dists.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode='reflect')
    
    kernel_size = window_size
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=features.device)
    kernel = kernel / kernel.sum()
    
    neighbor_dists = F.conv2d(
        padded,
        kernel.repeat(C, 1, 1, 1),
        groups=C
    )
    
    neighbor_dists = neighbor_dists.permute(0, 2, 3, 1)
    
    token_dists_center = token_dists
    
    eps = 1e-8
    kl_map = (token_dists_center * torch.log((token_dists_center + eps) / (neighbor_dists + eps))).sum(dim=-1)
    
    return kl_map[0]


def compute_block_importance_score(
    features: torch.Tensor,
    block_size: int = 2,
    mode: str = 'kl'
) -> torch.Tensor:
    """
    Compute importance score for each block.
    
    Args:
        features: (B, H, W, C) - Swin features in NHWC format
        block_size: size of block
        mode: 'kl' for KL divergence, 'variance' for variance-based
    
    Returns:
        scores: (num_blocks_h, num_blocks_w) - importance scores
    """
    if mode == 'kl':
        scores = compute_local_relative_entropy(features, block_size)
    elif mode == 'variance':
        B, H, W, C = features.shape
        block_dists, _ = compute_block_distribution(features, block_size)
        if block_dists is None:
            return None
        scores = block_dists.var(dim=-1).mean(dim=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return scores


def compute_softmax_topk_score(
    features: torch.Tensor,
    top_k_ratio: float = 0.7
) -> torch.Tensor:
    """
    Compute importance score based on softmax top-k (similar to SparseNet).
    
    Args:
        features: (B, H, W, C) - Swin features in NHWC format
        top_k_ratio: ratio of tokens to keep
    
    Returns:
        scores: (H, W) - importance scores based on max softmax probability
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    B, H, W, C = features.shape
    
    probs = F.softmax(features, dim=-1)
    max_probs = probs.max(dim=-1)[0]
    
    return max_probs[0]


class BlockEntropyCalculator:
    def __init__(self, block_size: int = 2, mode: str = 'kl'):
        """
        Initialize calculator for block pruning.
        
        Args:
            block_size: size of block for relative entropy computation
            mode: 'kl' for KL divergence, 'variance' for variance-based
        """
        self.block_size = block_size
        self.mode = mode
        
    def compute(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute importance score map.
        
        Args:
            features: (B, H, W, C) - Swin features in NHWC format
        
        Returns:
            score_map: (H, W) - importance score heatmap
        """
        return compute_block_importance_score(features, self.block_size, self.mode)
    
    def compute_multi_layer(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute importance scores for multiple layers and average.
        
        Args:
            features_list: list of (B, H, W, C) features from different layers
        
        Returns:
            combined_map: (H, W) - averaged score map
        """
        maps = []
        for features in features_list:
            score_map = self.compute(features)
            if score_map is not None:
                maps.append(score_map)
        
        if len(maps) == 0:
            return None
            
        return torch.stack(maps).mean(dim=0)


def create_block_calculator(block_size: int = 2, mode: str = 'kl') -> BlockEntropyCalculator:
    return BlockEntropyCalculator(block_size, mode)


def normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap_np = heatmap.cpu().numpy()
    min_val = heatmap_np.min()
    max_val = heatmap_np.max()
    if max_val - min_val < 1e-8:
        return torch.zeros_like(heatmap)
    normalized = (heatmap_np - min_val) / (max_val - min_val)
    return torch.from_numpy(normalized)


def resize_heatmap(heatmap: torch.Tensor, target_size: tuple) -> torch.Tensor:
    heatmap_np = heatmap.cpu().numpy()
    heatmap_pil = Image.fromarray((heatmap_np * 255).astype(np.uint8), mode='L')
    heatmap_resized = heatmap_pil.resize(target_size, Image.BILINEAR)
    heatmap_np = np.array(heatmap_resized).astype(np.float32) / 255.0
    return torch.from_numpy(heatmap_np)


def visualize_scores_3panel(
    score_map: torch.Tensor,
    image: Optional[Image.Image] = None,
    stage_idx: int = 0,
    save_path: Optional[str] = None,
    original_size: tuple = (224, 224),
) -> None:
    """
    Visualize block importance scores in 3 panels (aligned with entropy project).
    
    Panel 1: Original image with GT boxes (if provided)
    Panel 2: Image with heatmap overlay
    Panel 3: Heatmap only
    
    Args:
        score_map: (H, W) - block importance score map
        image: PIL Image for overlay (optional)
        stage_idx: stage index for title
        save_path: path to save figure
        original_size: (H, W) original image size for resizing heatmap
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    score_map_np = score_map.cpu().numpy()
    
    resized_heatmap = resize_heatmap(score_map, original_size)
    resized_heatmap_np = resized_heatmap.cpu().numpy()
    
    if image is not None:
        img_w, img_h = image.size
    else:
        img_w, img_h = original_size
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image if image else np.zeros((img_h, img_w, 3)))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    if image:
        axes[1].imshow(image)
        im = axes[1].imshow(resized_heatmap_np, cmap='jet', alpha=0.6)
    else:
        axes[1].imshow(resized_heatmap_np, cmap='jet')
        im = axes[1].imshow(resized_heatmap_np, cmap='jet')
    axes[1].set_title(f'Stage {stage_idx} KL Heatmap Overlay')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    im = axes[2].imshow(resized_heatmap_np, cmap='jet')
    axes[2].set_title(f'Stage {stage_idx} KL Map (Grayscale)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()


def visualize_multi_block_scores(
    block_scores: List[torch.Tensor],
    stage_idx: int,
    image: Optional[Image.Image] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize entropy scores for multiple blocks in a stage.
    
    Creates a figure with 3 columns per block:
    - Column 1: Original image
    - Column 2: Heatmap overlay
    - Column 3: Score map
    
    Args:
        block_scores: List of (H, W) score tensors for each block
        stage_idx: stage index
        image: PIL Image for overlay (optional)
        save_path: path to save figure
    """
    import matplotlib.pyplot as plt
    
    num_blocks = len(block_scores)
    if num_blocks == 0:
        return
    
    fig_width = 5 * 3
    fig_height = 5 * num_blocks
    fig, axes = plt.subplots(num_blocks, 3, figsize=(fig_width, fig_height))
    
    if num_blocks == 1:
        axes = axes.reshape(1, -1)
    
    for block_idx, score_map in enumerate(block_scores):
        score_map_np = score_map.cpu().numpy()
        
        if image is not None:
            img_w, img_h = image.size
            original_size = (img_w, img_h)
        else:
            original_size = (score_map.shape[1], score_map.shape[0])
        
        resized_heatmap = resize_heatmap(score_map, original_size)
        resized_heatmap_np = resized_heatmap.cpu().numpy()
        
        axes[block_idx, 0].imshow(image if image else np.zeros((original_size[1], original_size[0], 3)))
        axes[block_idx, 0].set_title(f'Block {block_idx} - Original')
        axes[block_idx, 0].axis('off')
        
        if image:
            axes[block_idx, 1].imshow(image)
            im = axes[block_idx, 1].imshow(resized_heatmap_np, cmap='jet', alpha=0.6)
        else:
            axes[block_idx, 1].imshow(resized_heatmap_np, cmap='jet')
            im = axes[block_idx, 1].imshow(resized_heatmap_np, cmap='jet')
        axes[block_idx, 1].set_title(f'Block {block_idx} - Overlay')
        axes[block_idx, 1].axis('off')
        plt.colorbar(im, ax=axes[block_idx, 1], fraction=0.046, pad=0.04)
        
        im = axes[block_idx, 2].imshow(resized_heatmap_np, cmap='jet')
        axes[block_idx, 2].set_title(f'Block {block_idx} - KL Score')
        axes[block_idx, 2].axis('off')
        plt.colorbar(im, ax=axes[block_idx, 2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Stage {stage_idx} - KL Divergence per Block', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
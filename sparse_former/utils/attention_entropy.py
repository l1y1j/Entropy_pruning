import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def compute_attention_entropy(attn: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute attention entropy for each token.
    
    Args:
        attn: Attention probabilities, shape (B * num_windows, num_heads, window_area, window_area)
              e.g., (B*nW, heads, 49, 49)
    
    Returns:
        entropy: Token-level entropy, shape (B * num_windows, window_area)
    """
    # attn shape: (B * nW, num_heads, N, N)
    # Compute entropy along the last dimension (key dimension)
    # H = -sum(p * log(p)) for each query
    
    H_heads = -torch.sum(attn * torch.log(attn + epsilon), dim=-1)  # (B*nW, num_heads, N)
    
    # Average over heads to get token-level entropy
    H_token = H_heads.mean(dim=1)  # (B*nW, N)
    
    return H_token


def pool_entropy_map(entropy: torch.Tensor, 
                     original_shape: Tuple[int, int],
                     target_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Pool entropy map to match target shape using 2x2 average pooling.
    
    Args:
        entropy: Entropy tensor, shape (B, H*W) or (B*nW, window_area)
        original_shape: Original spatial shape (H, W)
        target_shape: Target spatial shape (H', W')
    
    Returns:
        pooled_entropy: Pooled entropy, shape (B, H'*W')
    """
    B = entropy.shape[0]
    H, W = original_shape
    H_target, W_target = target_shape
    
    # Reshape to spatial format (B, H, W)
    if entropy.dim() == 2:
        entropy_spatial = entropy.view(B, H, W)
    else:
        entropy_spatial = entropy
    
    # 2x2 average pooling
    if H > H_target and W > W_target:
        kernel_size = 2
        stride = 2
        padding = 0
        
        # Ensure dimensions are divisible by 2
        H_pad = (H // kernel_size) * kernel_size
        W_pad = (W // kernel_size) * kernel_size
        
        if H_pad != H or W_pad != W:
            entropy_spatial = entropy_spatial[:, :H_pad, :W_pad]
        
        # Apply 2D average pooling
        pooled = F.avg_pool2d(entropy_spatial.float(), 
                              kernel_size=kernel_size, 
                              stride=stride,
                              padding=0)  # (B, H//2, W//2)
        
        H_pooled = pooled.shape[1]
        W_pooled = pooled.shape[2]
    else:
        pooled = entropy_spatial
        H_pooled = H
        W_pooled = W
    
    # Flatten back to (B, H*W)
    pooled_flat = pooled.view(B, H_pooled * W_pooled)
    
    # If target is smaller, we might need to handle edge case
    if H_pooled > H_target or W_pooled > W_target:
        # Take the first H_target * W_target values
        pooled_flat = pooled_flat[:, :H_target * W_target]
    elif H_pooled < H_target or W_pooled < W_target:
        # Pad to target size
        target_size = H_target * W_target
        if pooled_flat.shape[1] < target_size:
            pad_size = target_size - pooled_flat.shape[1]
            pooled_flat = F.pad(pooled_flat, (0, pad_size), value=0)
    
    return pooled_flat


def align_entropy_for_comparison(entropy_before: torch.Tensor,
                                  hw_before: Tuple[int, int],
                                  entropy_after: torch.Tensor,
                                  hw_after: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align entropy maps for comparison across stages.
    
    Args:
        entropy_before: Entropy from previous stage, shape (B, H_before*W_before)
        hw_before: Spatial shape before (H, W)
        entropy_after: Entropy from current stage, shape (B, H_after*W_after)
        hw_after: Spatial shape after (H', W')
    
    Returns:
        Tuple of aligned (entropy_before_aligned, entropy_after)
    """
    B = entropy_before.shape[0]
    
    # If shapes are the same, no alignment needed
    if hw_before == hw_after:
        return entropy_before, entropy_after
    
    # If before is larger, pool it to match after
    if hw_before[0] > hw_after[0] and hw_before[1] > hw_after[1]:
        entropy_before_aligned = pool_entropy_map(entropy_before, hw_before, hw_after)
        return entropy_before_aligned, entropy_after
    else:
        # If before is smaller or same, return as is
        return entropy_before, entropy_after


def compute_entropy_diff(entropy_before: torch.Tensor, 
                         entropy_after: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy difference between two stages.
    
    Args:
        entropy_before: Entropy before, shape (B, N)
        entropy_after: Entropy after, shape (B, N)
    
    Returns:
        entropy_diff: Absolute entropy difference, shape (B, N)
    """
    entropy_diff = torch.abs(entropy_before - entropy_after)
    return entropy_diff


def select_top_ratio(entropy_diff: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k tokens based on entropy difference ratio.
    
    Args:
        entropy_diff: Entropy difference, shape (B, N)
        ratio: Ratio of tokens to keep (0-1)
    
    Returns:
        Tuple of (mask_keep, mask_frozen):
            mask_keep: Binary mask for tokens to keep, shape (B, N)
            mask_frozen: Binary mask for tokens to freeze, shape (B, N)
    """
    B, N = entropy_diff.shape
    k = max(1, int(N * ratio))
    
    # Get threshold for top-k
    threshold, _ = torch.kthvalue(entropy_diff.flatten(), N - k)
    
    # Create masks
    mask_keep = (entropy_diff >= threshold).float()
    mask_frozen = 1.0 - mask_keep
    
    return mask_keep, mask_frozen


class AttentionEntropyCalculator:
    """Calculator for attention entropy and entropy difference"""
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def compute_entropy(self, attn: torch.Tensor) -> torch.Tensor:
        """Compute entropy from attention matrix"""
        return compute_attention_entropy(attn, self.epsilon)
    
    def align_and_diff(self, 
                      entropy_before: torch.Tensor,
                      hw_before: Tuple[int, int],
                      entropy_after: torch.Tensor,
                      hw_after: Tuple[int, int]) -> torch.Tensor:
        """Align entropy maps and compute difference"""
        ent_before_aligned, ent_after = align_entropy_for_comparison(
            entropy_before, hw_before, entropy_after, hw_after
        )
        return compute_entropy_diff(ent_before_aligned, ent_after)
    
    def select_frozen_tokens(self, entropy_diff: torch.Tensor, ratio: float) -> torch.Tensor:
        """Select tokens to freeze based on entropy difference ratio"""
        _, mask_frozen = select_top_ratio(entropy_diff, ratio)
        return mask_frozen


def create_attention_entropy_calculator() -> AttentionEntropyCalculator:
    """Factory function to create attention entropy calculator"""
    return AttentionEntropyCalculator()
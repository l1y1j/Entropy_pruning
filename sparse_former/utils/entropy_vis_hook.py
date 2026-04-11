import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, List
from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class EntropyVisualizationHook(Hook):
    """Hook for visualizing entropy scores during training
    
    This hook triggers visualization of entropy scores after each epoch
    for the specified stages and blocks.
    
    Args:
        vis_interval (int): Visualization interval in epochs (default: 1)
        vis_max_images (int): Max images to visualize per epoch (default: 2)
        save_dir (str): Directory to save visualizations (default: './entropy_vis')
        entropy_diff_dir (str): Directory to save entropy diff files (default: './entropy_diff')
    """
    
    def __init__(self,
                 vis_interval: int = 1,
                 vis_max_images: int = 2,
                 save_dir: str = './entropy_vis',
                 entropy_diff_dir: str = './entropy_diff',
                 **kwargs):
        self.vis_interval = vis_interval
        self.vis_max_images = vis_max_images
        self.save_dir = save_dir
        self.entropy_diff_dir = entropy_diff_dir
        self._vis_count = 0
    
    def after_train_epoch(self, runner):
        """Trigger visualization and entropy diff output after each training epoch"""
        current_epoch = runner.epoch
        if (current_epoch + 1) % self.vis_interval != 0:
            return
        
        # Get model
        model = runner.model
        
        # Find the backbone
        if hasattr(model, 'module'):
            model = model.module
        
        if hasattr(model, 'backbone'):
            backbone = model.backbone
        else:
            return
        
        # Get entropy config
        entropy_config = getattr(backbone, 'entropy_strategy', 'original')
        
        # Handle KL and original strategies
        if hasattr(backbone, 'get_entropy_scores'):
            entropy_scores = backbone.get_entropy_scores()
            
            if entropy_scores:
                # Create save directory for heatmaps
                os.makedirs(self.save_dir, exist_ok=True)
                
                # Visualize each block for each stage
                total_saved = 0
                for stage_idx, block_scores in entropy_scores.items():
                    for block_idx, score in enumerate(block_scores):
                        if score is None:
                            continue
                        
                        # Save heatmap for up to vis_max_images
                        for img_idx in range(self.vis_max_images):
                            save_path = os.path.join(
                                self.save_dir,
                                f'epoch_{current_epoch}_img{img_idx}_stage{stage_idx}_block{block_idx}_entropy.png'
                            )
                            
                            # Save heatmap
                            self._save_heatmap(score, save_path)
                            total_saved += 1
                
                if total_saved > 0:
                    print(f"[EntropyVisHook] Saved {total_saved} entropy visualizations to {self.save_dir}")
        
        # Handle entropy_diff strategy - output token-level entropy diff values
        if entropy_config == 'entropy_diff':
            self._output_entropy_diff(backbone, current_epoch)
    
    def _output_entropy_diff(self, backbone, current_epoch: int):
        """Output entropy difference values for each token"""
        if not hasattr(backbone, '_entropy_diff'):
            return
        
        entropy_diff_data = backbone._entropy_diff
        
        if not entropy_diff_data:
            return
        
        # Create directory
        os.makedirs(self.entropy_diff_dir, exist_ok=True)
        
        # Output file
        output_file = os.path.join(
            self.entropy_diff_dir,
            f'entropy_diff_epoch_{current_epoch}.txt'
        )
        
        with open(output_file, 'w') as f:
            f.write(f"Entropy Difference - Epoch {current_epoch}\n")
            f.write("=" * 50 + "\n\n")
            
            total_tokens = 0
            all_diffs = []
            
            for stage_idx, block_diffs in entropy_diff_data.items():
                f.write(f"Stage {stage_idx}:\n")
                for block_idx, diff in enumerate(block_diffs):
                    if diff is None:
                        continue
                    
                    # Compute statistics
                    diff_cpu = diff.detach().cpu()
                    mean_val = diff_cpu.mean().item()
                    std_val = diff_cpu.std().item()
                    min_val = diff_cpu.min().item()
                    max_val = diff_cpu.max().item()
                    num_tokens = diff_cpu.numel()
                    
                    f.write(f"  Block {block_idx}: ")
                    f.write(f"mean={mean_val:.6f}, std={std_val:.6f}, ")
                    f.write(f"min={min_val:.6f}, max={max_val:.6f}, ")
                    f.write(f"num_tokens={num_tokens}\n")
                    
                    # Store all token values for detailed output
                    all_diffs.append((stage_idx, block_idx, diff_cpu))
                    total_tokens += num_tokens
            
            f.write(f"\nTotal tokens processed: {total_tokens}\n")
            
            # Output detailed token values (first 100 per block for reference)
            f.write("\n" + "=" * 50 + "\n")
            f.write("Detailed token values (first 100 per block):\n")
            f.write("=" * 50 + "\n\n")
            
            for stage_idx, block_idx, diff_cpu in all_diffs:
                f.write(f"Stage {stage_idx}, Block {block_idx}:\n")
                flat_diff = diff_cpu.flatten()[:100]
                vals_str = ", ".join([f"{v:.6f}" for v in flat_diff.tolist()])
                f.write(f"  {vals_str}\n")
        
        print(f"[EntropyVisHook] Saved entropy diff to {output_file}")
    
    def _save_heatmap(self, score_map, save_path):
        """Save heatmap visualization"""
        # Normalize score map
        score_np = score_map.cpu().numpy()
        if score_np.max() > score_np.min():
            score_np = (score_np - score_np.min()) / (score_np.max() - score_np.min() + 1e-8)
        else:
            score_np = np.zeros_like(score_np)
        
        score_np = (score_np * 255).astype(np.uint8)
        
        # Create heatmap with colormap
        heatmap = Image.fromarray(score_np, mode='L')
        
        # Save
        heatmap.save(save_path)
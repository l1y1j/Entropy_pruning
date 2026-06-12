import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse


def plot_kl_histograms(data_list, output_dir, stage_filter=None, threshold_data=None):
    """Plot KL histograms and cumulative curves from collected KL scores.

    Args:
        data_list: List of dicts with keys: data, shape, stage_idx, block_idx, epoch, strategy
        output_dir: Directory to save plots
        stage_filter: tuple (stage_idx, block_idx) to filter data, or None for all
        threshold_data: dict with (stage_idx, block_idx, 'kl') -> threshold value, or None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stage_filter is not None:
        stage_idx, block_idx = stage_filter
        filtered_data = [
            item for item in data_list
            if item.get('stage_idx') == stage_idx and item.get('block_idx') == block_idx
        ]
        data_label = f'stage{stage_idx}_block{block_idx}'
        # Get threshold for this stage/block (stored as list of records)
        threshold_key = (stage_idx, block_idx, 'kl')
        threshold_value = None
        if threshold_data is not None and threshold_key in threshold_data:
            records = threshold_data[threshold_key]
            if records:
                # Use mean threshold across all records
                threshold_value = np.mean([r['threshold'] for r in records])
    else:
        filtered_data = data_list
        data_label = 'all_stages'
        threshold_value = None

    # Group by shape since different blocks have different window counts
    from collections import defaultdict
    groups = defaultdict(list)
    for item in filtered_data:
        shape_key = tuple(item['shape'])
        groups[shape_key].append(item)

    for shape_key, items in groups.items():
        print(f"\nProcessing shape {shape_key}: {len(items)} batches")
        all_data = np.concatenate([item['data'] for item in items], axis=0)
        print(f"  Merged data shape: {all_data.shape}")

        num_images = all_data.shape[0]
        tokens_per_image = all_data.shape[1] if len(all_data.shape) > 1 else all_data.shape[0]
        print(f"  Images: {num_images}, Tokens per image: {tokens_per_image}")

        shape_label = f's{shape_key[0]}x{shape_key[1]}' if len(shape_key) >= 2 else str(shape_key)
        data_label = f'{stage_filter[0]}_{stage_filter[1]}_{shape_label}' if stage_filter else f'all_{shape_label}'

        for img_idx in range(min(num_images, 10)):
            img_data = all_data[img_idx]  # each row is one image/batch
            kl_flat = img_data.flatten()

            # Normalize KL scores
            kl_min, kl_max = kl_flat.min(), kl_flat.max()
            normalized_kl = (kl_flat - kl_min) / (kl_max - kl_min + 1e-8)

            # Power penalty to sharpen
            info_mass = normalized_kl ** 1

            # Sort by info mass descending
            mass_sorted = np.sort(info_mass)[::-1]

            # Cumulative info mass
            cumulative_mass = np.cumsum(mass_sorted)
            total_mass = cumulative_mass[-1]

            # Convert to percentage
            cumulative_mass_pct = (cumulative_mass / total_mass) * 100
            token_pct = (np.arange(1, len(mass_sorted) + 1) / len(mass_sorted)) * 100

            # Find 95% threshold
            idx_95 = np.argmax(cumulative_mass_pct >= 95.0)
            tokens_needed_95 = token_pct[idx_95]

            print(f"\n  Image {img_idx}:")
            print(f"    Total Tokens: {len(kl_flat)}")
            print(f"    To preserve 95% of KL Mass, we only need {tokens_needed_95:.2f}% of Tokens!")

            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(info_mass, bins=50, alpha=0.7, color='royalblue', edgecolor='black')
            axes[0].set_title(f'Image {img_idx} - Info Mass Distribution (Normalized KL)', fontsize=12)
            axes[0].set_xlabel('Information Mass [0, 1]')
            axes[0].set_ylabel('Token Count')
            axes[0].grid(axis='y', linestyle='--', alpha=0.6)

            # Add threshold line if available
            if threshold_value is not None:
                # Convert threshold to info_mass space (normalized_kl is 0-1, info_mass is normalized_kl^2)
                threshold_mass = threshold_value ** 1
                axes[0].axvline(x=threshold_mass, color='red', linestyle='--', linewidth=2,
                                label=f'Learned Threshold: {threshold_mass:.4f}')
                axes[0].legend(loc='upper right', fontsize=10)

            axes[1].plot(token_pct, cumulative_mass_pct, color='darkorange', linewidth=3, label='Cumulative Info Mass')
            axes[1].axhline(y=95, color='red', linestyle='--', linewidth=1.5, label='95% Mass Target')
            axes[1].axvline(x=tokens_needed_95, color='green', linestyle='--', linewidth=1.5,
                            label=f'Tokens Needed: {tokens_needed_95:.1f}%')

            # Add learned threshold as vertical line in cumulative curve
            if threshold_value is not None:
                # threshold_value is in [0, 1] normalized space
                # On cumulative curve, we need to find what token percentage corresponds to this threshold
                # Tokens are sorted by info_mass descending, so threshold corresponds to
                # the percentage of tokens with info_mass >= threshold^2
                above_threshold = mass_sorted >= (threshold_value ** 1)
                token_pct_at_threshold = (np.sum(above_threshold) / len(mass_sorted)) * 100
                axes[1].axvline(x=token_pct_at_threshold, color='purple', linestyle='-.', linewidth=2,
                                label=f'Learned Threshold: {token_pct_at_threshold:.1f}% tokens')

            axes[1].scatter([tokens_needed_95], [95], color='red', s=100, zorder=5)

            axes[1].set_title(f'Image {img_idx} - Information Mass Preservation (KL)', fontsize=12)
            axes[1].set_xlabel('Percentage of Retained Tokens (%) - [Compute Cost]')
            axes[1].set_ylabel('Percentage of Retained Info Mass (%)')
            axes[1].set_xlim(0, 100)
            axes[1].set_ylim(0, 105)
            axes[1].legend(loc='lower right')
            axes[1].grid(linestyle='--', alpha=0.6)

            plt.tight_layout()

            save_path = output_dir / f'{data_label}_image_{img_idx:02d}_kl_mass_preservation.png'
            plt.savefig(save_path, dpi=150)
            print(f"    Saved plot to {save_path}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot KL score histograms and cumulative curves')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Base directory containing epoch_* folders with scores (default: auto-detect)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Full path to kl_scores.pkl file (overrides --base_dir)')
    parser.add_argument('--epoch', type=str, default='latest',
                        help='Epoch to visualize (e.g., "001", "latest")')
    parser.add_argument('--stage', type=int, default=None,
                        help='Stage index to filter (0-3)')
    parser.add_argument('--block', type=int, default=None,
                        help='Block index to filter (0-5)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots')
    args = parser.parse_args()

    # Determine data path
    if args.data_path:
        data_path = Path(args.data_path)
    elif args.base_dir:
        base_dir = Path(args.base_dir)
        if args.epoch == 'latest':
            epoch_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('epoch_')],
                                key=lambda x: int(x.name.split('_')[1]))
            if epoch_dirs:
                data_path = epoch_dirs[-1] / 'kl_scores.pkl'
            else:
                print(f"No epoch directories found in {base_dir}")
                return
        else:
            data_path = base_dir / f'epoch_{args.epoch}' / 'kl_scores.pkl'
    else:
        # Auto-detect from kl_scores_export directory
        base_dir = Path(__file__).parent / 'kl_scores_export'
        if args.epoch == 'latest':
            # Find the highest epoch directory
            epoch_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('epoch_')],
                                key=lambda x: int(x.name.split('_')[1]))
            if epoch_dirs:
                data_path = epoch_dirs[-1] / 'kl_scores.pkl'
            else:
                print(f"No epoch directories found in {base_dir}")
                return
        else:
            data_path = base_dir / f'epoch_{args.epoch}' / 'kl_scores.pkl'

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    # Load data
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"Loaded {len(data_list)} batches from {data_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_path.parent / 'plots'

    # Build stage filter
    stage_filter = None
    if args.stage is not None and args.block is not None:
        stage_filter = (args.stage, args.block)

    # Load threshold data if available
    threshold_data = None
    threshold_path = data_path.parent / 'kl_thresholds.pkl'
    if threshold_path.exists():
        with open(threshold_path, 'rb') as f:
            threshold_data = pickle.load(f)
        print(f"Loaded threshold data: {len(threshold_data)} entries")

    # Plot
    plot_kl_histograms(data_list, output_dir, stage_filter, threshold_data)


if __name__ == '__main__':
    main()
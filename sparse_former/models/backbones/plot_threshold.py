import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse


def plot_thresholds(data_dict, output_dir, strategy='kl'):
    """Plot threshold evolution across epochs.

    Args:
        data_dict: Dict with keys (stage_idx, block_idx, strategy) -> list of threshold records
        output_dir: Directory to save plots
        strategy: 'kl' or 'inc'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by stage_idx, block_idx
    from collections import defaultdict
    groups = defaultdict(list)
    for key, values in data_dict.items():
        if len(key) == 3 and key[2] == strategy:
            stage_idx, block_idx, _ = key
            groups[(stage_idx, block_idx)].extend(values)

    if not groups:
        print(f"No {strategy.upper()} threshold data found")
        return

    # Plot each (stage, block) combination
    for (stage_idx, block_idx), values in groups.items():
        if not values:
            continue

        # Sort by epoch
        values_sorted = sorted(values, key=lambda x: x['epoch'])

        epochs = [v['epoch'] for v in values_sorted]
        thresholds = [v['threshold'] for v in values_sorted]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, thresholds, 'b-o', linewidth=2, markersize=8)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Threshold', fontsize=12)
        ax.set_title(f'{strategy.upper()} Threshold Evolution - Stage {stage_idx}, Block {block_idx}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Add value annotations
        for i, (ep, th) in enumerate(zip(epochs, thresholds)):
            if i % 2 == 0 or len(epochs) <= 5:  # annotate every 2nd point or all if few points
                ax.annotate(f'{th:.4f}', (ep, th), textcoords='offset points',
                           xytext=(0, 10), ha='center', fontsize=9)

        plt.tight_layout()

        save_path = output_dir / f'{strategy}_threshold_s{stage_idx}_b{block_idx}.png'
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot threshold evolution across epochs')
    parser.add_argument('--epoch', type=str, default='latest',
                        help='Epoch to visualize (e.g., "001", "latest")')
    parser.add_argument('--strategy', type=str, default='kl', choices=['kl', 'inc'],
                        help='Strategy to visualize (kl or inc)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots')
    args = parser.parse_args()

    # Determine data path
    base_dir = Path(__file__).parent / 'kl_scores_export'

    if args.epoch == 'latest':
        epoch_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('epoch_')],
                            key=lambda x: int(x.name.split('_')[1]))
        if epoch_dirs:
            epoch_path = epoch_dirs[-1]
        else:
            print(f"No epoch directories found in {base_dir}")
            return
    else:
        epoch_path = base_dir / f'epoch_{args.epoch}'

    if not epoch_path.exists():
        print(f"Epoch directory not found: {epoch_path}")
        return

    # Load thresholds
    if args.strategy == 'kl':
        data_path = epoch_path / 'kl_thresholds.pkl'
    else:
        data_path = epoch_path / 'inc_thresholds.pkl'

    if not data_path.exists():
        print(f"Threshold file not found: {data_path}")
        return

    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)

    print(f"Loaded threshold data: {len(data_dict)} entries")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = epoch_path / 'plots'

    # Plot
    plot_thresholds(data_dict, output_dir, args.strategy)


if __name__ == '__main__':
    main()
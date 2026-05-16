"""
Calibration set diagnostics.

Usage:
    python diagnostics.py [dataset] [--n_seeds 20] [--cal_frac 0.3]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from data import load_inputs, random_cal_test_split

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_diagnostics(dataset='plantnet', n_seeds=20, cal_frac=0.3, show=True):
    """
    Compute calibration-set diagnostics, save plot + metrics file, and return stats.

    Returns
    -------
    dict with keys n_cal, n_zero, n_one, n_le5, each a (mean, se) tuple.
    """
    data = load_inputs(dataset=dataset)
    softmax = data['softmax']
    labels = data['labels']
    num_classes = data['num_classes']

    n_cal_list = []
    n_zero_list = []
    n_one_list = []
    n_le5_list = []
    counts_per_seed = []

    for seed in range(n_seeds):
        _, cal_labels, _, _ = random_cal_test_split(
            softmax, labels, cal_frac=cal_frac, seed=seed
        )
        counts = np.bincount(cal_labels, minlength=num_classes)
        n_cal_list.append(len(cal_labels))
        n_zero_list.append((counts == 0).sum())
        n_one_list.append((counts == 1).sum())
        n_le5_list.append((counts <= 5).sum())
        counts_per_seed.append(np.sort(counts)[::-1])

    def agg(vals):
        a = np.array(vals, dtype=float)
        mean = a.mean()
        se = a.std(ddof=1) / np.sqrt(len(a))
        return float(mean), float(se)

    def fmt(mean, se):
        return f'{mean:.1f} ± {se:.2f}'

    stats = dict(
        n_cal=agg(n_cal_list),
        n_zero=agg(n_zero_list),
        n_one=agg(n_one_list),
        n_le5=agg(n_le5_list),
    )

    print(f'\nCalibration diagnostics  |  dataset={dataset}  '
          f'n_seeds={n_seeds}  cal_frac={cal_frac}')
    print('-' * 55)
    print(f'  # calibration points              {fmt(*stats["n_cal"])}')
    print(f'  # classes with 0 cal points       {fmt(*stats["n_zero"])}')
    print(f'  # classes with 1 cal point        {fmt(*stats["n_one"])}')
    print(f'  # classes with <= 5 cal points    {fmt(*stats["n_le5"])}')
    print()

    # ------------------------------------------------------------------ #
    # Plot: sorted per-class counts (rank vs. count), mean across seeds
    # ------------------------------------------------------------------ #
    counts_matrix = np.stack(counts_per_seed, axis=0)
    mean_counts = counts_matrix.mean(axis=0)
    ranks = np.arange(1, num_classes + 1)

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for counts in counts_per_seed:
        ax.plot(ranks, counts, linewidth=0.7, color='steelblue', alpha=0.15)
    ax.plot(ranks, mean_counts, linewidth=1.5, color='steelblue', alpha=0.9, label='mean')
    ax.set_xlabel('Class rank (sorted by # calibration examples)')
    ax.set_ylabel('# calibration examples')
    ax.set_yscale('log')
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)
    ax.set_xlim(1, num_classes)

    out_dir = os.path.join(_THIS_DIR, 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)

    for ext in ('png', 'pdf'):
        plot_path = os.path.join(out_dir, f'cal_counts_{dataset}.{ext}')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {plot_path}')
    if show:
        plt.show()
    plt.close(fig)

    # Save metrics
    metrics_path = os.path.join(out_dir, f'cal_metrics_{dataset}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'dataset={dataset}  n_seeds={n_seeds}  cal_frac={cal_frac}\n')
        f.write(f'# calibration points              {fmt(*stats["n_cal"])}\n')
        f.write(f'# classes with 0 cal points       {fmt(*stats["n_zero"])}\n')
        f.write(f'# classes with 1 cal point        {fmt(*stats["n_one"])}\n')
        f.write(f'# classes with <= 5 cal points    {fmt(*stats["n_le5"])}\n')
    print(f'Metrics saved to {metrics_path}')

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', default='plantnet')
    parser.add_argument('--n_seeds', type=int, default=20)
    parser.add_argument('--cal_frac', type=float, default=0.2)
    args = parser.parse_args()
    run_diagnostics(args.dataset, args.n_seeds, args.cal_frac)


if __name__ == '__main__':
    main()

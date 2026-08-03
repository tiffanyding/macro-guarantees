"""
2D coverage-region plot for plantnet-trunc.

X-axis: 1 - alpha_1  (MacroCov target, LWCP + PAS)
Y-axis: 1 - alpha_2  (MarginalCov target, standard CP + softmax)

For each cell (alpha_1, alpha_2):
  Red    : LWCP+PAS (targeting MacroCov >= 1-alpha_1) also achieves MarginalCov >= 1-alpha_2
  Blue   : StdCP+softmax (targeting MarginalCov >= 1-alpha_2) also achieves MacroCov >= 1-alpha_1
  Purple : both hold simultaneously

Transparency scales with the fraction of random seeds where the condition is met.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data import load_inputs, random_cal_test_split
from conformal import get_conformal_scores, macro_weights, compute_weighted_qhat, standard_qhat
from experiment_setup import _get_train_class_distr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET       = 'plantnet-trunc'
N_SEEDS       = 20
CAL_FRAC      = 0.2
# TARGET_LEVELS = np.linspace(0.90, 0.99, 10)   # 1 - alpha values
# TARGET_LEVELS = np.linspace(0.90, 0.99, 19)   # 1 - alpha values
TARGET_LEVELS = np.linspace(0.80, 0.99, 39)   # 1 - alpha values
ALPHA_VALS    = 1.0 - TARGET_LEVELS            # alpha_1 / alpha_2 values


# ---------------------------------------------------------------------------
# Fast coverage helpers
# ---------------------------------------------------------------------------

def _marginal_cov(true_test_scores, q):
    return float((true_test_scores <= q).mean())


def _macro_cov(true_test_scores, q, test_labels, num_classes):
    covered       = (true_test_scores <= q).astype(float)
    class_correct = np.bincount(test_labels, weights=covered, minlength=num_classes)
    class_count   = np.bincount(test_labels, minlength=num_classes).astype(float)
    valid         = class_count > 0
    class_cov     = np.where(valid, class_correct / np.maximum(class_count, 1.0), np.nan)
    return float(class_cov[valid].mean()) if valid.any() else np.nan


# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------

def compute_coverage_grid(data, n_seeds=N_SEEDS, cal_frac=CAL_FRAC):
    """
    Returns a dict with:
      red_frac   (n_a2, n_a1) : fraction of seeds LWCP+PAS achieves marginal_cov >= 1-alpha_2
      blue_frac  (n_a2, n_a1) : fraction of seeds StdCP+softmax achieves macro_cov >= 1-alpha_1
      m1_marg    (n_a1,)      : mean marginal_cov of LWCP+PAS    (bonus metric)
      m1_macro   (n_a1,)      : mean macro_cov   of LWCP+PAS    (guaranteed metric)
      m2_marg    (n_a2,)      : mean marginal_cov of StdCP+sfx  (guaranteed metric)
      m2_macro   (n_a2,)      : mean macro_cov   of StdCP+sfx  (bonus metric)
    """
    n           = len(ALPHA_VALS)
    num_classes = data['num_classes']

    red_counts  = np.zeros((n, n))   # [alpha_2_idx, alpha_1_idx]
    blue_counts = np.zeros((n, n))
    m1_marg_acc  = np.zeros(n)
    m1_macro_acc = np.zeros(n)
    m2_marg_acc  = np.zeros(n)
    m2_macro_acc = np.zeros(n)

    for seed in range(n_seeds):
        print(f'  seed {seed + 1}/{n_seeds}', end='\r', flush=True)
        cal_sm, cal_labels, test_sm, test_labels = random_cal_test_split(
            data['softmax'], data['labels'], cal_frac=cal_frac, seed=seed
        )
        train_class_distr = _get_train_class_distr(data, cal_labels)

        arange_cal  = np.arange(len(cal_labels))
        arange_test = np.arange(len(test_labels))

        cal_true_pas  = get_conformal_scores(cal_sm,  'PAS', train_class_distr)[arange_cal,  cal_labels]
        cal_true_sfx  = get_conformal_scores(cal_sm,  'softmax'               )[arange_cal,  cal_labels]
        test_true_pas = get_conformal_scores(test_sm, 'PAS', train_class_distr)[arange_test, test_labels]
        test_true_sfx = get_conformal_scores(test_sm, 'softmax'               )[arange_test, test_labels]

        w_macro = macro_weights(cal_labels, num_classes)

        # Method 1: LWCP+PAS — threshold set by alpha_1
        for i, a1 in enumerate(ALPHA_VALS):
            q    = compute_weighted_qhat(cal_true_pas, a1, w_macro)
            marg = _marginal_cov(test_true_pas, q)
            mac  = _macro_cov(test_true_pas, q, test_labels, num_classes)
            m1_marg_acc[i]  += marg
            m1_macro_acc[i] += mac
            red_counts[:, i] += (marg >= TARGET_LEVELS).astype(float)

        # Method 2: StdCP+softmax — threshold set by alpha_2
        for j, a2 in enumerate(ALPHA_VALS):
            q    = standard_qhat(cal_true_sfx, a2)
            marg = _marginal_cov(test_true_sfx, q)
            mac  = _macro_cov(test_true_sfx, q, test_labels, num_classes)
            m2_marg_acc[j]  += marg
            m2_macro_acc[j] += mac
            blue_counts[j, :] += (mac >= TARGET_LEVELS).astype(float)

    print()

    return dict(
        red_frac  = red_counts   / n_seeds,
        blue_frac = blue_counts  / n_seeds,
        m1_marg   = m1_marg_acc  / n_seeds,
        m1_macro  = m1_macro_acc / n_seeds,
        m2_marg   = m2_marg_acc  / n_seeds,
        m2_macro  = m2_macro_acc / n_seeds,
    )


# ---------------------------------------------------------------------------
# Main coverage-region plot
# ---------------------------------------------------------------------------

def _imshow_layer(ax, x_edges, y_edges, frac, color_rgb, max_alpha=0.9):
    """Draw one semitransparent color layer using imshow (alpha respected per pixel)."""
    n_y, n_x = frac.shape
    rgba = np.zeros((n_y, n_x, 4))
    rgba[:, :, :3] = color_rgb
    rgba[:, :, 3]  = frac * max_alpha
    ax.imshow(rgba, origin='lower', interpolation='nearest', aspect='auto',
              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])


def _set_grid_axes(ax, xlabel=True, ylabel=True):
    step    = TARGET_LEVELS[1] - TARGET_LEVELS[0]
    x_edges = np.r_[TARGET_LEVELS - step / 2, TARGET_LEVELS[-1] + step / 2]
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(x_edges[0], x_edges[-1])
    ax.set_xticks(TARGET_LEVELS)
    ax.set_yticks(TARGET_LEVELS)
    ax.set_xticklabels(
        [f'{v:.3f}' for v in TARGET_LEVELS], rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels([f'{v:.3f}' for v in TARGET_LEVELS], fontsize=6)
    if xlabel:
        ax.set_xlabel(r'$1 - \alpha_1$  (MacroCov target)', fontsize=10)
    if ylabel:
        ax.set_ylabel(r'$1 - \alpha_2$  (MarginalCov target)', fontsize=10)


def plot_coverage_grid(results, out_path=None):
    red_frac  = results['red_frac']
    blue_frac = results['blue_frac']

    step    = TARGET_LEVELS[1] - TARGET_LEVELS[0]
    x_edges = np.r_[TARGET_LEVELS - step / 2, TARGET_LEVELS[-1] + step / 2]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    _imshow_layer(ax, x_edges, x_edges, red_frac,  (1.0, 0.0, 0.0))
    _imshow_layer(ax, x_edges, x_edges, blue_frac, (0.0, 0.0, 1.0))
    ax.plot([TARGET_LEVELS[0], TARGET_LEVELS[-1]],
            [TARGET_LEVELS[0], TARGET_LEVELS[-1]],
            'k--', lw=0.9, alpha=0.45, zorder=3)
    _set_grid_axes(ax)

    red_patch    = mpatches.Patch(color=(1.0, 0.0, 0.0), alpha=0.9,
                                  label=r'LWCP+PAS also achieves MarginalCov $\geq 1{-}\alpha_2$')
    blue_patch   = mpatches.Patch(color=(0.0, 0.0, 1.0), alpha=0.9,
                                  label=r'StdCP+softmax also achieves MacroCov $\geq 1{-}\alpha_1$')
    # purple_patch = mpatches.Patch(color=(0.5, 0.0, 0.5), alpha=0.7, label='Both simultaneously')
    ax.legend(handles=[red_patch, blue_patch],
              loc='lower right', fontsize=7, framealpha=0.9)

    fig.suptitle(f'Coverage regions — {DATASET}  ({N_SEEDS} seeds)', fontsize=10, y=1.01)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Debug 2x2 grid
# ---------------------------------------------------------------------------

def plot_debug_grids(results, out_path=None):
    """
    2x2 figure showing the mean coverage value in each cell.

    Layout:
      [0,0] LWCP+PAS     | macro_cov    (guaranteed — should be >= 1-alpha_1)
      [0,1] LWCP+PAS     | marginal_cov (bonus — determines red shading)
      [1,0] StdCP+softmax | marginal_cov (guaranteed — should be >= 1-alpha_2)
      [1,1] StdCP+softmax | macro_cov   (bonus — determines blue shading)

    method1 metrics vary with alpha_1 only (cols), so each row is identical.
    method2 metrics vary with alpha_2 only (rows), so each col is identical.
    """
    n = len(ALPHA_VALS)

    # Broadcast 1D metric vectors to (n_alpha2, n_alpha1) grids
    m1_macro = np.broadcast_to(results['m1_macro'][np.newaxis, :], (n, n))
    m1_marg  = np.broadcast_to(results['m1_marg'] [np.newaxis, :], (n, n))
    m2_marg  = np.broadcast_to(results['m2_marg'] [:, np.newaxis], (n, n))
    m2_macro = np.broadcast_to(results['m2_macro'][:, np.newaxis], (n, n))

    panels = [
        (m1_macro, 'LWCP+PAS',      'macro_cov',    'guaranteed'),
        (m1_marg,  'LWCP+PAS',      'marginal_cov', 'bonus → red shading'),
        (m2_marg,  'StdCP+softmax', 'marginal_cov', 'guaranteed'),
        (m2_macro, 'StdCP+softmax', 'macro_cov',    'bonus → blue shading'),
    ]

    step    = TARGET_LEVELS[1] - TARGET_LEVELS[0]
    x_edges = np.r_[TARGET_LEVELS - step / 2, TARGET_LEVELS[-1] + step / 2]
    cmap    = plt.cm.RdYlGn
    vmin, vmax = 0.85, 1.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    for ax, (vals, method, metric, subtitle) in zip(axes.flat, panels):
        mesh = ax.pcolormesh(x_edges, x_edges, vals,
                             cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')

        for j in range(n):       # alpha_2 index (y / row)
            for i in range(n):   # alpha_1 index (x / col)
                v          = vals[j, i]
                norm_v     = (v - vmin) / (vmax - vmin)
                text_color = 'white' if norm_v < 0.25 else 'black'
                ax.text(TARGET_LEVELS[i], TARGET_LEVELS[j], f'{v:.3f}',
                        ha='center', va='center', fontsize=6, color=text_color)

        ax.plot([TARGET_LEVELS[0], TARGET_LEVELS[-1]],
                [TARGET_LEVELS[0], TARGET_LEVELS[-1]],
                'k--', lw=0.8, alpha=0.4)
        _set_grid_axes(ax)
        plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'{method} — {metric}\n({subtitle})', fontsize=9)

    fig.suptitle(f'Debug: mean coverage values — {DATASET}  ({N_SEEDS} seeds)', fontsize=11)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    data = load_inputs(DATASET)
    print(f'Loaded {len(data["labels"])} examples, {data["num_classes"]} classes.')

    print('Computing coverage grid...')
    results = compute_coverage_grid(data)

    base = os.path.join(_THIS_DIR, 'results')
    plot_coverage_grid(results,
                       out_path=os.path.join(base, f'coverage_region_{DATASET}.pdf'))
    plot_debug_grids(results,
                     out_path=os.path.join(base, f'coverage_region_{DATASET}_debug.pdf'))


if __name__ == '__main__':
    main()

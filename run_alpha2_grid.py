"""
Alpha-2 grid sweep: fix alpha_1=0.1, vary alpha_2 from 0.01 to 0.10.

For each alpha_2 runs the Simultaneous label-weighted method (max(q_macro, q_marg))
with softmax and PAS scores, and reports MacroCov, MarginalCov, AvgSize, and which
constraint was binding.

Usage:
    python run_alpha2_grid.py [dataset] [--alpha_1 0.1] [--alpha_2_steps 10]
                              [--n_seeds 20] [--cal_frac 0.2]

Examples:
    python run_alpha2_grid.py plantnet-trunc
    python run_alpha2_grid.py plantnet-trunc --alpha_1 0.1 --alpha_2_steps 10
"""
import argparse
import os
import time

import numpy as np

from data import load_inputs
from experiment_setup import experiment_alpha2_grid

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_METRIC_DISPLAY = {
    'marginal_cov': 'MarginalCov',
    'macro_cov':    'MacroCov',
    'avg_set_size': 'AvgSize',
    'binding_macro': 'Binding(macro)',
}

_METRICS_ORDERED = ['marginal_cov', 'macro_cov', 'avg_set_size', 'binding_macro']


def _fmt(mean, se, decimals=3):
    if np.isnan(mean):
        return 'N/A'
    return f'{mean:.{decimals}f} ({se:.{decimals}f})'


def print_table_text(score, alpha_1, alpha_2_grid, grid_results):
    col_w = 10
    val_w = 20
    labels = [_METRIC_DISPLAY[m] for m in _METRICS_ORDERED]
    header = f"{'alpha_2':<{col_w}}" + ''.join(f"{lbl:>{val_w}}" for lbl in labels)
    title = f'\nSimultaneous | {score}  (alpha_1={alpha_1})'
    print(title)
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for alpha_2 in alpha_2_grid:
        res = grid_results[alpha_2][f'Simultaneous | {score}']
        row = f"{alpha_2:<{col_w}.2f}"
        for metric in _METRICS_ORDERED:
            mean, se = res[metric]
            dec = 1 if metric == 'avg_set_size' else 3
            row += f"{_fmt(mean, se, dec):>{val_w}}"
        print(row)
    print()


def save_table_latex(score, alpha_1, alpha_2_grid, grid_results, out_path):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)

    score_latex = {'softmax': r'$s_{\softmax}$', 'PAS': r'$s_{\PAS}$'}.get(score, score)
    metrics_display = ['marginal_cov', 'macro_cov', 'avg_set_size']
    metric_headers = ' & '.join(
        {'marginal_cov': r'$\mathrm{MarginalCov}$',
         'macro_cov':    r'$\mathrm{MacroCov}$',
         'avg_set_size': r'$\mathrm{AvgSize}$'}[m]
        for m in metrics_display
    )

    lines = [
        r'% Requires \usepackage{booktabs}',
        r'\begin{table}[h]',
        r'\centering',
        (f'\\caption{{Simultaneous label-weighted conformal with {score_latex}, '
         f'$\\alpha_1={alpha_1}$ fixed, varying $\\alpha_2$}}'),
        r'\begin{tabular}{rrrr}',
        r'\toprule',
        f'$\\alpha_2$ & {metric_headers} \\\\',
        r'\midrule',
    ]

    for alpha_2 in alpha_2_grid:
        res = grid_results[alpha_2][f'Simultaneous | {score}']
        cells = []
        for metric in metrics_display:
            mean, se = res[metric]
            dec = 1 if metric == 'avg_set_size' else 3
            mean_str = f'{mean:.{dec}f}'
            se_str = f'{se:.{dec}f}'
            bold = (
                (metric == 'macro_cov' and mean >= 1 - alpha_1)
                or (metric == 'marginal_cov' and mean >= 1 - alpha_2)
            )
            cell = f'\\textbf{{{mean_str}}} ({se_str})' if bold else f'{mean_str} ({se_str})'
            cells.append(cell)
        binding_mean = res['binding_macro'][0]
        binding_str = f'macro ({binding_mean:.0%})' if binding_mean >= 0.5 else f'marginal ({1-binding_mean:.0%})'
        lines.append(f'{alpha_2:.2f} & {" & ".join(cells)} \\\\ % binding: {binding_str}')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', default='plantnet-trunc')
    parser.add_argument('--alpha_1', type=float, default=0.1,
                        help='Fixed MacroCov miscoverage level')
    parser.add_argument('--alpha_2_min', type=float, default=0.01,
                        help='Minimum alpha_2 value')
    parser.add_argument('--alpha_2_max', type=float, default=0.10,
                        help='Maximum alpha_2 value')
    parser.add_argument('--alpha_2_steps', type=int, default=10,
                        help='Number of alpha_2 grid points')
    parser.add_argument('--n_seeds', type=int, default=20)
    parser.add_argument('--cal_frac', type=float, default=0.2)
    args = parser.parse_args()

    dataset   = args.dataset
    alpha_1   = args.alpha_1
    alpha_2_grid = np.round(
        np.linspace(args.alpha_2_min, args.alpha_2_max, args.alpha_2_steps), 10
    )

    print(f'Dataset: {dataset}  |  alpha_1={alpha_1}  |  cal_frac={args.cal_frac}  |  n_seeds={args.n_seeds}')
    print(f'alpha_2 grid: {list(np.round(alpha_2_grid, 4))}')

    data = load_inputs(dataset=dataset)
    print(f'Loaded {len(data["labels"])} examples, {data["num_classes"]} classes.\n')

    os.makedirs(os.path.join(_THIS_DIR, 'results'), exist_ok=True)

    grid_results = {}
    for alpha_2 in alpha_2_grid:
        alpha_2 = float(alpha_2)
        print(f'--- alpha_2={alpha_2:.3f} ---')
        t0 = time.time()
        grid_results[alpha_2] = experiment_alpha2_grid(
            data, alpha_1=alpha_1, alpha_2=alpha_2,
            n_splits=args.n_seeds, cal_frac=args.cal_frac,
        )
        print(f'  done in {time.time() - t0:.1f}s')

    for score in ('softmax', 'PAS'):
        print_table_text(score, alpha_1, [float(a) for a in alpha_2_grid], grid_results)

    for score in ('softmax', 'PAS'):
        out_path = os.path.join(
            _THIS_DIR, 'results',
            f'alpha2_grid_{dataset}_alpha1={alpha_1}_{score}.txt'
        )
        save_table_latex(score, alpha_1, [float(a) for a in alpha_2_grid], grid_results, out_path)


if __name__ == '__main__':
    main()

"""
Simultaneous MacroCov + MarginalCov table:
Standard, Classwise, Simultaneous × {softmax, PAS, WPAS} for one (alpha_macro, alpha_marginal) pair.

Usage:
    python run_simultaneous_results.py [dataset] [--alpha_macro 0.1] [--alpha_marginal 0.05]
                                       [--n_seeds 20] [--cal_frac 0.2] [--lambda_steps 21]

Examples:
    python run_simultaneous_results.py plantnet-trunc
    python run_simultaneous_results.py plantnet-trunc --alpha_macro 0.1 --alpha_marginal 0.05
"""
import argparse
import os
import time

import numpy as np

from data import load_inputs
from experiment_setup import experiment_simultaneous_macrocov_marginalcov

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_METRIC_DISPLAY = {
    'marginal_cov': r'$\mathrm{MarginalCov}$',
    'macro_cov':    r'$\mathrm{MacroCov}$',
    'avg_set_size': r'$\mathrm{AvgSize}$',
}

_DATASET_LATEX = {
    'plantnet-trunc':    r'\truncplant',
    'inaturalist-trunc': r'\truncinat',
}


def _fmt(mean, se, decimals=3):
    if np.isnan(mean):
        return 'N/A'
    return f'{mean:.{decimals}f} ({se:.{decimals}f})'


def _metric_label(m):
    return _METRIC_DISPLAY.get(m, m)


def print_table_text(title, results):
    if not results:
        return
    methods = list(results.keys())
    metrics = list(results[methods[0]].keys())
    labels = [_metric_label(m) for m in metrics]

    col_w = max(len(m) for m in methods) + 2
    val_w = 18

    header = f"{'Method':<{col_w}}" + ''.join(f"{lbl:>{val_w}}" for lbl in labels)
    print(f'\n{title}')
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for method in methods:
        row = f"{method:<{col_w}}"
        for metric in metrics:
            mean, se = results[method][metric]
            dec = 1 if metric == 'avg_set_size' else 3
            row += f"{_fmt(mean, se, dec):>{val_w}}"
        print(row)
    print()


def save_table_latex(title, results, out_path, alpha_macro=None, alpha_marginal=None, label=None):
    """
    Bold logic:
      marginal_cov — bold if mean >= 1-alpha_marginal
      macro_cov    — bold if mean >= 1-alpha_macro
      avg_set_size — bold for smallest mean among rows satisfying BOTH guarantees
    """
    if not results:
        return
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)

    keys = list(results.keys())
    metrics = list(results[keys[0]].keys())

    if alpha_macro is not None and alpha_marginal is not None:
        both_keys = [k for k in keys
                     if results[k]['macro_cov'][0] >= 1 - alpha_macro
                     and results[k]['marginal_cov'][0] >= 1 - alpha_marginal]
        best_size_key = (min(both_keys, key=lambda k: results[k]['avg_set_size'][0])
                         if both_keys else None)
    else:
        best_size_key = None

    def _cell(key, metric):
        mean, se = results[key][metric]
        dec = 1 if metric == 'avg_set_size' else 3
        mean_str = f'{mean:.{dec}f}'
        se_str = f'{se:.{dec}f}'
        bold = (
            (metric == 'macro_cov' and alpha_macro is not None and mean >= 1 - alpha_macro)
            or (metric == 'marginal_cov' and alpha_marginal is not None and mean >= 1 - alpha_marginal)
            or (metric == 'avg_set_size' and key == best_size_key)
        )
        if bold:
            return f'\\textbf{{{mean_str}}} ({se_str})'
        return f'{mean_str} ({se_str})'

    _score_latex = {
        'softmax': r'$s_{\softmax}$',
        'PAS':     r'$s_{\PAS}$',
        'WPAS':    r'$\hat{s}_{\lambda}$',
    }
    _method_latex = {
        'Standard':     r'\standard',
        'Classwise':    r'\classwise',
        'Simultaneous': r'\labelw',
    }

    parsed = []
    for k in keys:
        method, score = k.split(' | ', 1) if ' | ' in k else (k, '')
        parsed.append((_method_latex.get(method, method), _score_latex.get(score, score), k))

    from itertools import groupby
    groups = [(m, list(rows)) for m, rows in groupby(parsed, key=lambda x: x[0])]

    col_spec = 'll' + 'r' * len(metrics)
    metric_headers = ' & '.join(_metric_label(m) for m in metrics)

    label_line = f'\\label{{{label}}}' if label else ''
    lines = [
        r'% Requires \usepackage{booktabs,multirow}',
        r'\begin{table}[h]',
        r'\centering',
        f'\\caption{{{title}}}{label_line}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        f'Conformal method & Score & {metric_headers} \\\\',
        r'\midrule',
    ]

    for i, (method, rows) in enumerate(groups):
        span = len(rows)
        for j, (_, score, key) in enumerate(rows):
            vals = ' & '.join(_cell(key, m) for m in metrics)
            method_cell = f'\\multirow{{{span}}}{{*}}{{{method}}}' if j == 0 else ''
            lines.append(f'{method_cell} & {score} & {vals} \\\\')
        if i < len(groups) - 1:
            lines.append(r'\midrule')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', default='plantnet-trunc',
                        help='Dataset name (must have data/<dataset>/ directory)')
    parser.add_argument('--alpha_macro', type=float, default=0.1,
                        help='Miscoverage level for MacroCov guarantee')
    parser.add_argument('--alpha_marginal', type=float, default=0.2,
                        help='Miscoverage level for MarginalCov guarantee')
    parser.add_argument('--n_seeds', type=int, default=20)
    parser.add_argument('--cal_frac', type=float, default=0.2)
    parser.add_argument('--tune_frac', type=float, default=0.5,
                        help='Fraction of cal set used for lambda tuning (rest used for quantile)')
    parser.add_argument('--lambda_steps', type=int, default=11,
                        help='Number of lambda values in [0,1] grid')
    args = parser.parse_args()

    dataset = args.dataset
    alpha_macro = args.alpha_macro
    alpha_marginal = args.alpha_marginal
    lambda_grid = np.linspace(0.0, 1.0, args.lambda_steps)

    print(f'Dataset: {dataset}  |  cal_frac={args.cal_frac}  |  tune_frac={args.tune_frac}  |  n_seeds={args.n_seeds}')
    print(f'alpha_macro={alpha_macro}  |  alpha_marginal={alpha_marginal}')
    print(f'lambda_grid: {args.lambda_steps} values in [0, 1]')

    data = load_inputs(dataset=dataset)
    print(f'Loaded {len(data["labels"])} examples, {data["num_classes"]} classes.')

    os.makedirs(os.path.join(_THIS_DIR, 'results'), exist_ok=True)

    t0 = time.time()
    results = experiment_simultaneous_macrocov_marginalcov(
        data,
        alpha_macro=alpha_macro,
        alpha_marginal=alpha_marginal,
        n_splits=args.n_seeds,
        cal_frac=args.cal_frac,
        tune_frac=args.tune_frac,
        lambda_grid=lambda_grid,
    )
    elapsed = time.time() - t0

    dataset_latex = _DATASET_LATEX.get(dataset, dataset)
    title = (f'Targeting $1-\\alpha_1={1 - alpha_macro}$ $\\mathrm{{MacroCov}}$ and '
             f'$1-\\alpha_2={1 - alpha_marginal}$ $\\mathrm{{MarginalCov}}$ on {dataset_latex}')
    print_table_text(title, results)

    out_path = os.path.join(
        _THIS_DIR, 'results',
        f'simultaneous_{dataset}_alpha_macro={alpha_macro}_alpha_marginal={alpha_marginal}.txt')
    label = f'tab:{dataset}_simultaneous_alpha_macro={alpha_macro}_alpha_marginal={alpha_marginal}'
    save_table_latex(title, results, out_path,
                     alpha_macro=alpha_macro, alpha_marginal=alpha_marginal, label=label)
    print(f'Time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()

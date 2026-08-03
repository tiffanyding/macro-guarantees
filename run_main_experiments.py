"""
Main results table: Standard, Classwise, Label-weighted × {softmax, PAS}
for multiple alpha values. One LaTeX table per alpha.

Usage:
    python run_main_results.py [dataset] [--alphas 0.1,0.05] [--n_seeds 20] [--cal_frac 0.2]

Examples:
    python run_main_results.py plantnet-trunc
    python run_main_results.py plantnet-trunc --alphas 0.1 --n_seeds 50
"""
import argparse
import os
import time

import numpy as np

from data import load_inputs
from diagnostics import run_diagnostics
from experiment_setup import experiment_main_results

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_METRIC_DISPLAY = {
    'marginal_cov': r'$\mathrm{MarginalCov}$',
    'macro_cov':    r'$\mathrm{MacroCov}$',
    'avg_set_size': r'$\mathrm{AvgSize}$',
}

_DATASET_LATEX = {
    'plantnet-trunc':     r'\truncplant',
    'inaturalist-trunc':  r'\truncinat',
}


def _fmt(mean, se, decimals=3):
    if np.isnan(mean):
        return 'N/A'
    return f'{mean:.{decimals}f} ({se:.{decimals}f})'


def _metric_label(m):
    return _METRIC_DISPLAY.get(m, m)


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

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


def save_table_latex(title, results, out_path, alpha=None, label=None):
    """Two-column (Method, Score) LaTeX table with multirow grouping.

    Formatting:
      macro_cov    — bold if mean >= 1-alpha (guarantee met)
      avg_set_size — bold for the row with the smallest mean among
                     guarantee-meeting rows
    """
    if not results:
        return
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)

    keys = list(results.keys())
    metrics = list(results[keys[0]].keys())

    # Key with smallest AvgSize among rows that meet the MacroCov guarantee
    if alpha is not None:
        guarantee_keys = [k for k in keys
                          if results[k]['macro_cov'][0] >= 1 - alpha]
        best_size_key = (min(guarantee_keys,
                             key=lambda k: results[k]['avg_set_size'][0])
                         if guarantee_keys else None)
    else:
        best_size_key = None

    def _cell(key, metric):
        mean, se = results[key][metric]
        dec = 1 if metric == 'avg_set_size' else 3
        mean_str = f'{mean:.{dec}f}'
        se_str = f'{se:.{dec}f}'
        bold = (
            (metric == 'macro_cov' and alpha is not None and mean >= 1 - alpha)
            or (metric == 'avg_set_size' and key == best_size_key)
        )
        if bold:
            return f'\\textbf{{{mean_str}}} ({se_str})'
        return f'{mean_str} ({se_str})'

    _score_latex  = {'softmax': r'$s_{\softmax}$', 'PAS': r'$s_{\PAS}$',
                     'APS': r'$s_{\mathrm{APS}}$', 'RAPS': r'$s_{\mathrm{RAPS}}$'}
    _method_latex = {'Standard': r'\standard', 'Classwise': r'\classwise',
                     'Label-weighted': r'\labelw',
                     'Clustered': r'\clustered', 'RC3P': r'\rcp'}

    # Parse 'Method | Score' keys
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', default='plantnet-trunc',
                        help='Dataset name (must have data/<dataset>/ directory)')
    parser.add_argument('--alphas', type=str, default='0.1,0.05',
                        help='Comma-separated miscoverage levels')
    parser.add_argument('--n_seeds', type=int, default=20)
    parser.add_argument('--cal_frac', type=float, default=0.1)
    parser.add_argument('--methods', type=str, default='standard,classwise,label_weighted',
                        help="Comma-separated list from: standard, classwise, label_weighted, "
                             "clustered, rc3p, tacp (tacp is not yet implemented)")
    parser.add_argument('--raps_lambda', type=float, default=0.01,
                        help='RAPS regularization strength (only used by clustered/rc3p)')
    parser.add_argument('--raps_kreg', type=int, default=5,
                        help='RAPS rank threshold (only used by clustered/rc3p)')
    parser.add_argument('--no-diagnostics', dest='diagnostics',
                        action='store_false', default=True,
                        help='Skip cal-count plot and metrics file')
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(',')]
    methods = [m.strip() for m in args.methods.split(',')]
    dataset = args.dataset

    print(f'Dataset: {dataset}  |  cal_frac={args.cal_frac}  |  n_seeds={args.n_seeds}')
    print(f'Alphas: {alphas}')

    data = load_inputs(dataset=dataset)
    print(f'Loaded {len(data["labels"])} examples, {data["num_classes"]} classes.')

    # Diagnostics: generates cal-count plot + metrics file, prints zero-class stats
    if args.diagnostics:
        run_diagnostics(dataset, args.n_seeds, args.cal_frac, show=False)

    os.makedirs(os.path.join(_THIS_DIR, 'results'), exist_ok=True)

    for alpha in alphas:
        print(f'\n=== alpha={alpha} ===')
        t0 = time.time()
        results = experiment_main_results(
            data, alpha=alpha, n_splits=args.n_seeds, cal_frac=args.cal_frac,
            methods=methods, raps_lambda=args.raps_lambda, raps_kreg=args.raps_kreg)
        elapsed = time.time() - t0

        dataset_latex = _DATASET_LATEX.get(dataset, dataset)
        title = f'Targeting $1-\\alpha={1 - alpha}$ $\\mathrm{{MacroCov}}$ on {dataset_latex}'
        print_table_text(title, results)

        out_path = os.path.join(
            _THIS_DIR, 'results', f'main_{dataset}_alpha={alpha}.txt')
        label = f'tab:{dataset}_main_alpha={alpha}'
        save_table_latex(title, results, out_path, alpha=alpha, label=label)
        print(f'Time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()

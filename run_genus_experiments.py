"""
Genus macro-coverage table:
Standard, Classwise, Label-weighted × {softmax, GroupPAS} for multiple alpha values.
One LaTeX table per alpha.

Usage:
    python run_genus_results.py [dataset] [--alphas 0.1,0.05] [--n_seeds 20] [--cal_frac 0.2]

Examples:
    python run_genus_results.py plantnet-trunc
    python run_genus_results.py plantnet-trunc --alphas 0.1
"""
import argparse
import os
import time

import numpy as np

from data import load_inputs, load_genus_assignments
from experiment_setup import experiment_genus_macrocov

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_METRIC_DISPLAY = {
    'marginal_cov':    r'$\mathrm{MarginalCov}$',
    'group_macro_cov': r'$\mathrm{GenusMacroCov}$',
    'avg_set_size':    r'$\mathrm{AvgSize}$',
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


def save_table_latex(title, results, out_path, alpha=None, label=None):
    """Two-column (Method, Score) LaTeX table with multirow grouping.

    Formatting:
      group_macro_cov — bold if mean >= 1-alpha (guarantee met)
      avg_set_size    — bold for the row with the smallest mean among
                        guarantee-meeting rows
    """
    if not results:
        return
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)

    keys = list(results.keys())
    metrics = list(results[keys[0]].keys())

    if alpha is not None:
        guarantee_keys = [k for k in keys
                          if results[k]['group_macro_cov'][0] >= 1 - alpha]
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
            (metric == 'group_macro_cov' and alpha is not None and mean >= 1 - alpha)
            or (metric == 'avg_set_size' and key == best_size_key)
        )
        if bold:
            return f'\\textbf{{{mean_str}}} ({se_str})'
        return f'{mean_str} ({se_str})'

    _score_latex  = {'softmax': r'$s_{\softmax}$', 'GroupPAS': r'$\hat{s}_{g,w}$'}
    _method_latex = {'Standard': r'\standard', 'Classwise': r'\classwise',
                     'Label-weighted': r'\labelw'}

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
    parser.add_argument('--alphas', type=str, default='0.1,0.05',
                        help='Comma-separated miscoverage levels')
    parser.add_argument('--n_seeds', type=int, default=20)
    parser.add_argument('--cal_frac', type=float, default=0.2)
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(',')]
    dataset = args.dataset

    print(f'Dataset: {dataset}  |  cal_frac={args.cal_frac}  |  n_seeds={args.n_seeds}')
    print(f'Alphas: {alphas}')

    data = load_inputs(dataset=dataset)
    print(f'Loaded {len(data["labels"])} examples, {data["num_classes"]} classes.')

    data_dir = os.path.join(_THIS_DIR, 'data', dataset)
    try:
        genus_assignments, genus_names = load_genus_assignments(data_dir)
        print(f'Loaded genus assignments: {len(genus_names)} genera for {data["num_classes"]} classes.')
    except FileNotFoundError as e:
        print(f'Error: could not load genus assignments ({e}).')
        return

    os.makedirs(os.path.join(_THIS_DIR, 'results'), exist_ok=True)

    for alpha in alphas:
        print(f'\n=== alpha={alpha} ===')
        t0 = time.time()
        results = experiment_genus_macrocov(
            data, alpha=alpha, n_splits=args.n_seeds, cal_frac=args.cal_frac,
            genus_assignments=genus_assignments)
        elapsed = time.time() - t0

        dataset_latex = _DATASET_LATEX.get(dataset, dataset)
        title = f'Targeting $1-\\alpha={1 - alpha}$ $\\mathrm{{GenusMacroCov}}$ on {dataset_latex}'
        print_table_text(title, results)

        out_path = os.path.join(
            _THIS_DIR, 'results', f'genus_{dataset}_alpha={alpha}.txt')
        label = f'tab:{dataset}_genus_alpha={alpha}'
        save_table_latex(title, results, out_path, alpha=alpha, label=label)
        print(f'Time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()

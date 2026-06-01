import numpy as np

from conformal import (
    get_conformal_scores,
    macro_weights,
    weighted_macro_weights, group_weights, group_macro_weights,
    compute_weighted_qhat, standard_qhat, classwise_qhats,
    create_prediction_sets, compute_metrics,
)
from data import random_cal_test_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_train_class_distr(data, cal_labels):
    """Use train_labels if available, else fall back to cal_labels."""
    num_classes = data['num_classes']
    # src = data['train_labels'] if 'train_labels' in data else cal_labels
    src = data['train_labels']
    counts = np.bincount(src, minlength=num_classes).astype(float)
    return np.maximum(counts, 1e-10)

def compute_sim_pas_weights(num_classes, p_hat, lam):
    """SimPAS class weights: lam/K + (1-lam)*p_hat, interpolates PAS (lam=1) and softmax (lam=0)."""
    return lam / num_classes + (1.0 - lam) * p_hat



# ---------------------------------------------------------------------------
# Shared run loop
# ---------------------------------------------------------------------------

def _run_splits(methods_fn, data, alpha, n_splits, cal_frac):
    """
    methods_fn(cal_scores_dict, cal_labels, test_scores_dict, test_labels, alpha, num_classes, seed)
        -> dict: method_name -> metrics_dict

    Runs n_splits times with seeds 0..n_splits-1.
    Returns: dict method_name -> {metric_name -> (mean, se)}
    """
    softmax = data['softmax']
    labels = data['labels']
    num_classes = data['num_classes']

    all_results = {}

    for seed in range(n_splits):
        cal_sm, cal_labels, test_sm, test_labels = random_cal_test_split(
            softmax, labels, cal_frac=cal_frac, seed=seed
        )

        train_class_distr = _get_train_class_distr(data, cal_labels)

        # WPAS* weights: c_k = 1/K_active for active classes, 0 otherwise
        active = np.bincount(cal_labels, minlength=num_classes) > 0
        K_active = int(active.sum())
        wpas_star_weights = np.where(active, 1.0 / K_active if K_active > 0 else 0.0, 0.0)

        cal_scores_dict = {
            'softmax': get_conformal_scores(cal_sm, 'softmax'),
            'PAS':     get_conformal_scores(cal_sm, 'PAS', train_class_distr),
            'WPAS*':   get_conformal_scores(cal_sm, 'WPAS', train_class_distr, wpas_star_weights),
        }
        test_scores_dict = {
            'softmax': get_conformal_scores(test_sm, 'softmax'),
            'PAS':     get_conformal_scores(test_sm, 'PAS', train_class_distr),
            'WPAS*':   get_conformal_scores(test_sm, 'WPAS', train_class_distr, wpas_star_weights),
        }

        split_results = methods_fn(
            cal_scores_dict, cal_labels, test_scores_dict, test_labels,
            alpha, num_classes, seed,
        )

        for method, metrics in split_results.items():
            all_results.setdefault(method, []).append(metrics)

    # Aggregate
    aggregated = {}
    for method, metrics_list in all_results.items():
        metric_names = metrics_list[0].keys()
        agg = {}
        for metric in metric_names:
            vals = np.array([d[metric] for d in metrics_list if not np.isnan(d[metric])])
            mean = vals.mean() if len(vals) > 0 else np.nan
            se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
            agg[metric] = (mean, se)
        aggregated[method] = agg

    return aggregated


def _aggregate(all_results):
    aggregated = {}
    for method, metrics_list in all_results.items():
        metric_names = metrics_list[0].keys()
        agg = {}
        for metric in metric_names:
            vals = np.array([d[metric] for d in metrics_list if not np.isnan(d.get(metric, np.nan))])
            mean = vals.mean() if len(vals) > 0 else np.nan
            se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
            agg[metric] = (mean, se)
        aggregated[method] = agg
    return aggregated


def _true_scores(scores_all, labels):
    """Extract the true-class score for each calibration example."""
    return scores_all[np.arange(len(labels)), labels]


# ---------------------------------------------------------------------------
# Main results
# ---------------------------------------------------------------------------

def experiment_main_results(data, alpha=0.1, n_splits=20, cal_frac=0.2):
    """
    Main results table: Standard, Classwise, Label-weighted × {softmax, PAS}.
    Reports MarginalCov, MacroCov (over all test classes), AvgSize.
    """
    def methods_fn(cal_score_dict, cal_labels, test_sd, test_labels, alpha, num_classes, seed):
        m = lambda psets: compute_metrics(psets, test_labels, num_classes)
        results = {}

        q = standard_qhat(_true_scores(cal_score_dict['softmax'], cal_labels), alpha)
        results['Standard | softmax'] = m(create_prediction_sets(test_sd['softmax'], q))

        q = standard_qhat(_true_scores(cal_score_dict['PAS'], cal_labels), alpha)
        results['Standard | PAS'] = m(create_prediction_sets(test_sd['PAS'], q))

        qhats = classwise_qhats(_true_scores(cal_score_dict['softmax'], cal_labels), cal_labels, num_classes, alpha)
        results['Classwise | softmax'] = m(create_prediction_sets(test_sd['softmax'], qhats))

        w = macro_weights(cal_labels, num_classes)
        q = compute_weighted_qhat(_true_scores(cal_score_dict['softmax'], cal_labels), alpha, w)
        results['Label-weighted | softmax'] = m(create_prediction_sets(test_sd['softmax'], q))

        q = compute_weighted_qhat(_true_scores(cal_score_dict['PAS'], cal_labels), alpha, w)
        results['Label-weighted | PAS'] = m(create_prediction_sets(test_sd['PAS'], q))

        return results

    return _run_splits(methods_fn, data, alpha, n_splits, cal_frac)


def _build_group_pas_weights(cal_labels, num_classes, genus_assignments):
    """
    Normalized WPAS class weights for GroupPAS score:
        c_k = 1 / (G * K_{g(k)})
    where G = active groups, K_g = active classes in group g.
    Active = has ≥1 calibration example.
    """
    counts = np.bincount(cal_labels, minlength=num_classes)
    active = counts > 0
    num_groups = int(genus_assignments.max()) + 1
    K_g = np.array([(active & (genus_assignments == g)).sum()
                    for g in range(num_groups)], dtype=float)
    G = float((K_g > 0).sum())
    if G == 0:
        return np.zeros(num_classes)
    safe_denom = np.where(K_g[genus_assignments] > 0, G * K_g[genus_assignments], 1.0)
    class_weights = np.where(active & (K_g[genus_assignments] > 0), 1.0 / safe_denom, 0.0)
    return class_weights


# ---------------------------------------------------------------------------
# At-risk upweighted macro-coverage experiment
# ---------------------------------------------------------------------------

def experiment_rare_upweighted_macrocov(data, alpha, n_splits, cal_frac,
                                         rare_mask, lambda_val=100):
    """
    Weighted macro-coverage experiment targeting MacroCov_rare↑ ≥ 1-alpha.

    Rare species (bottom rare_frac by frequency) receive weight lambda_val; others 1.
    Methods: Standard | softmax, Standard | WPAS, Classwise | softmax,
             Label-weighted | softmax, Label-weighted | WPAS.
    Metrics: weighted_macro_cov (= MacroCov_rare↑), marginal_cov, avg_set_size.
    """
    num_classes = data['num_classes']
    class_importance = np.where(rare_mask, float(lambda_val), 1.0)
    wpas_weights = class_importance.copy()
    wpas_weights = class_importance / class_importance.sum()

    softmax = data['softmax']
    labels = data['labels']
    all_results = {}
    keep = {'marginal_cov', 'weighted_macro_cov', 'avg_set_size'}

    for seed in range(n_splits):
        cal_sm, cal_labels, test_sm, test_labels = random_cal_test_split(
            softmax, labels, cal_frac=cal_frac, seed=seed
        )
        train_class_distr = _get_train_class_distr(data, cal_labels)

        cal_sd = {
            'softmax': get_conformal_scores(cal_sm, 'softmax'),
            'WPAS':    get_conformal_scores(cal_sm, 'WPAS', train_class_distr, wpas_weights),
        }
        test_sd = {
            'softmax': get_conformal_scores(test_sm, 'softmax'),
            'WPAS':    get_conformal_scores(test_sm, 'WPAS', train_class_distr, wpas_weights),
        }

        def m(psets):
            full = compute_metrics(psets, test_labels, num_classes, class_weights=wpas_weights)
            return {k: full[k] for k in full if k in keep}

        results = {}

        q = standard_qhat(_true_scores(cal_sd['softmax'], cal_labels), alpha)
        results['Standard | softmax'] = m(create_prediction_sets(test_sd['softmax'], q))

        q = standard_qhat(_true_scores(cal_sd['WPAS'], cal_labels), alpha)
        results['Standard | WPAS'] = m(create_prediction_sets(test_sd['WPAS'], q))

        qhats = classwise_qhats(_true_scores(cal_sd['softmax'], cal_labels),
                                cal_labels, num_classes, alpha)
        results['Classwise | softmax'] = m(create_prediction_sets(test_sd['softmax'], qhats))

        w = weighted_macro_weights(cal_labels, num_classes, class_importance)

        q = compute_weighted_qhat(_true_scores(cal_sd['softmax'], cal_labels), alpha, w)
        results['Label-weighted | softmax'] = m(create_prediction_sets(test_sd['softmax'], q))

        q = compute_weighted_qhat(_true_scores(cal_sd['WPAS'], cal_labels), alpha, w)
        results['Label-weighted | WPAS'] = m(create_prediction_sets(test_sd['WPAS'], q))

        for method, metrics in results.items():
            all_results.setdefault(method, []).append(metrics)

    return _aggregate(all_results)


# ---------------------------------------------------------------------------
# Genus macro-coverage experiment
# ---------------------------------------------------------------------------

def experiment_genus_macrocov(data, alpha, n_splits, cal_frac, genus_assignments):
    """
    Genus-level macro-coverage experiment targeting GenusMacroCov ≥ 1-alpha.

    Methods: Standard | softmax, Standard | GroupPAS,
             Classwise | softmax,
             Label-weighted | softmax, Label-weighted | GroupPAS.
    Metrics: marginal_cov, group_macro_cov (= GenusMacroCov), avg_set_size.
    """
    num_classes = data['num_classes']
    softmax = data['softmax']
    labels = data['labels']
    all_results = {}
    keep = {'marginal_cov', 'group_macro_cov', 'avg_set_size'}

    for seed in range(n_splits):
        cal_sm, cal_labels, test_sm, test_labels = random_cal_test_split(
            softmax, labels, cal_frac=cal_frac, seed=seed
        )
        train_class_distr = _get_train_class_distr(data, cal_labels)

        group_pas_weights = _build_group_pas_weights(cal_labels, num_classes, genus_assignments)

        cal_sd = {
            'softmax':  get_conformal_scores(cal_sm, 'softmax'),
            'GroupPAS': get_conformal_scores(cal_sm, 'WPAS', train_class_distr, group_pas_weights),
        }
        test_sd = {
            'softmax':  get_conformal_scores(test_sm, 'softmax'),
            'GroupPAS': get_conformal_scores(test_sm, 'WPAS', train_class_distr, group_pas_weights),
        }

        def m(psets):
            full = compute_metrics(psets, test_labels, num_classes,
                                   group_assignments=genus_assignments)
            return {k: full[k] for k in ['marginal_cov', 'group_macro_cov', 'avg_set_size']}

        results = {}

        q = standard_qhat(_true_scores(cal_sd['softmax'], cal_labels), alpha)
        results['Standard | softmax'] = m(create_prediction_sets(test_sd['softmax'], q))

        q = standard_qhat(_true_scores(cal_sd['GroupPAS'], cal_labels), alpha)
        results['Standard | GroupPAS'] = m(create_prediction_sets(test_sd['GroupPAS'], q))

        qhats = classwise_qhats(_true_scores(cal_sd['softmax'], cal_labels),
                                cal_labels, num_classes, alpha)
        results['Classwise | softmax'] = m(create_prediction_sets(test_sd['softmax'], qhats))

        w = group_macro_weights(cal_labels, num_classes, genus_assignments)

        q = compute_weighted_qhat(_true_scores(cal_sd['softmax'], cal_labels), alpha, w)
        results['Label-weighted | softmax'] = m(create_prediction_sets(test_sd['softmax'], q))

        q = compute_weighted_qhat(_true_scores(cal_sd['GroupPAS'], cal_labels), alpha, w)
        results['Label-weighted | GroupPAS'] = m(create_prediction_sets(test_sd['GroupPAS'], q))

        for method, metrics in results.items():
            all_results.setdefault(method, []).append(metrics)

    return _aggregate(all_results)


# ---------------------------------------------------------------------------
# Simultaneous MacroCov + MarginalCov experiment
# ---------------------------------------------------------------------------

def _tune_lambda_loo(tune_softmax, tune_labels, num_classes,
                     alpha_macro, alpha_marginal, lambda_grid, train_class_distr):
    """
    Leave-one-out lambda tuning: pick lambda minimizing expected set size.
    train_class_distr is used as both the WPAS score denominator and the
    class-weight input to compute_sim_pas_weights.
    """
    n = len(tune_labels)
    w_macro_full = macro_weights(tune_labels, num_classes)

    best_lam = lambda_grid[0]
    best_mean_size = np.inf

    for lam in lambda_grid:
        sim_weights = compute_sim_pas_weights(num_classes, train_class_distr, lam)
        scores_all = get_conformal_scores(tune_softmax, 'WPAS', train_class_distr, sim_weights)
        true_sc = scores_all[np.arange(n), tune_labels]

        total_size = 0.0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            sc_loo = true_sc[mask]
            w_loo = w_macro_full[mask]
            w_sum = w_loo.sum()
            w_loo_norm = w_loo / w_sum if w_sum > 0 else w_loo

            q_macro_loo = compute_weighted_qhat(sc_loo, alpha_macro, w_loo_norm)
            q_marg_loo = standard_qhat(sc_loo, alpha_marginal)
            q_loo = max(q_macro_loo, q_marg_loo)

            total_size += float((scores_all[i] <= q_loo).sum())

        mean_size = total_size / n
        print(f'  lambda={lam:.3f}  avg_size={mean_size:.4f}')
        if mean_size < best_mean_size:
            best_mean_size = mean_size
            best_lam = lam

    print(f'  => selected lambda={best_lam:.3f} (avg_size={best_mean_size:.4f})')
    return best_lam


def experiment_simultaneous_macrocov_marginalcov(
        data, alpha_macro=0.1, alpha_marginal=0.05,
        n_splits=20, cal_frac=0.2, tune_frac=0.5, lambda_grid=None):
    """
    Simultaneous MacroCov >= 1-alpha_macro and MarginalCov >= 1-alpha_marginal.

    For softmax and PAS: quantiles are computed using all cal examples.
    For WPAS: tune_frac of cal is used to tune lambda; the rest (proper_cal)
    is used to compute quantiles.

    train_class_distr is the normalized class distribution from train_labels
    (if available, else from all cal_labels). It serves as both the WPAS score
    denominator and the input to compute_sim_pas_weights.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.0, 1.0, 11)

    alpha_base = min(alpha_macro, alpha_marginal)
    num_classes = data['num_classes']
    softmax_all = data['softmax']
    labels = data['labels']
    all_results = {}
    keep = ('marginal_cov', 'macro_cov', 'avg_set_size')

    for seed in range(n_splits):
        cal_sm, cal_labels, test_sm, test_labels = random_cal_test_split(
            softmax_all, labels, cal_frac=cal_frac, seed=seed
        )

        # Unified normalized class distribution for score denominator and sim_weights
        raw = _get_train_class_distr(data, cal_labels)
        train_class_distr = raw / raw.sum()

        # Split cal into tune (lambda selection) + proper_cal (WPAS quantile)
        n_tune = int(len(cal_labels) * tune_frac)
        tune_sm, tune_labels = cal_sm[:n_tune], cal_labels[:n_tune]
        prop_sm, prop_labels = cal_sm[n_tune:], cal_labels[n_tune:]

        # *** TEMP: set lambda manually
        # best_lam = 0.6
        # print(f"FOR DEBUGGING: USING FIXED LAMBDA={best_lam}")

        # Identify best lambda
        best_lam = _tune_lambda_loo(tune_sm, tune_labels, num_classes,
                                    alpha_macro, alpha_marginal, lambda_grid,
                                    train_class_distr)

        sim_weights = compute_sim_pas_weights(num_classes, train_class_distr, best_lam)

        # # *** TEMP: Sanity check if issue is due to data splitting
        # cal_sm = prop_sm
        # cal_labels = prop_labels

        # *** TEMP: Try double dipping in the data
        print("FOR DEBUGGING: DOUBLE DIPPING ")
        prop_sm = cal_sm
        prop_labels = cal_labels


        # Softmax and PAS use ALL cal; WPAS uses proper_cal only
        cal_sd_all = {
            'softmax': get_conformal_scores(cal_sm, 'softmax'),
            'PAS':     get_conformal_scores(cal_sm, 'PAS', train_class_distr),
        }
        cal_sd_prop = {
            'WPAS': get_conformal_scores(prop_sm, 'WPAS', train_class_distr, sim_weights),
        }
        test_sd = {
            'softmax': get_conformal_scores(test_sm, 'softmax'),
            'PAS':     get_conformal_scores(test_sm, 'PAS', train_class_distr),
            'WPAS':    get_conformal_scores(test_sm, 'WPAS', train_class_distr, sim_weights),
        }

        def m(psets):
            full = compute_metrics(psets, test_labels, num_classes)
            return {k: full[k] for k in keep}

        w_macro_all  = macro_weights(cal_labels, num_classes)
        w_macro_prop = macro_weights(prop_labels, num_classes)
        results = {}

        # Standard baselines
        for score in ('softmax', 'PAS'):
            q = standard_qhat(_true_scores(cal_sd_all[score], cal_labels), alpha_base)
            results[f'Standard | {score}'] = m(create_prediction_sets(test_sd[score], q))
        q = standard_qhat(_true_scores(cal_sd_prop['WPAS'], prop_labels), alpha_base)
        results['Standard | WPAS'] = m(create_prediction_sets(test_sd['WPAS'], q))

        # Classwise baseline (all cal)
        qhats = classwise_qhats(_true_scores(cal_sd_all['softmax'], cal_labels),
                                cal_labels, num_classes, alpha_base)
        results['Classwise | softmax'] = m(create_prediction_sets(test_sd['softmax'], qhats))

        # Simultaneous: softmax and PAS use all cal
        for score in ('softmax', 'PAS'):
            ts = _true_scores(cal_sd_all[score], cal_labels)
            q_macro = compute_weighted_qhat(ts, alpha_macro, w_macro_all)
            q_marg  = standard_qhat(ts, alpha_marginal)
            print(f'  Simultaneous | {score}: q_macro={q_macro:.4f}, q_marg={q_marg:.4f}, binding={"macro" if q_macro >= q_marg else "marginal"}')
            results[f'Simultaneous | {score}'] = m(
                create_prediction_sets(test_sd[score], max(q_macro, q_marg)))

        # Simultaneous: WPAS uses proper_cal
        ts = _true_scores(cal_sd_prop['WPAS'], prop_labels)
        q_macro = compute_weighted_qhat(ts, alpha_macro, w_macro_prop)
        q_marg  = standard_qhat(ts, alpha_marginal)
        print(f'  Simultaneous | WPAS: q_macro={q_macro:.4f}, q_marg={q_marg:.4f}, binding={"macro" if q_macro >= q_marg else "marginal"}')
        results['Simultaneous | WPAS'] = m(
            create_prediction_sets(test_sd['WPAS'], max(q_macro, q_marg)))

        for method, met in results.items():
            all_results.setdefault(method, []).append(met)

    return _aggregate(all_results)

import numpy as np

from conformal import (
    get_conformal_scores,
    macro_weights,
    weighted_macro_weights, group_weights, group_macro_weights,
    compute_weighted_qhat, standard_qhat, classwise_qhats, mondrian_qhats,
    create_prediction_sets, compute_metrics,
)
from baselines import (
    get_aps_scores_all, get_raps_scores_all,
    clustered_conformal, compute_rc3p_params, create_rc3p_prediction_sets,
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

_DEFAULT_MAIN_METHODS = ('standard', 'classwise', 'label_weighted')


def experiment_main_results(data, alpha=0.1, n_splits=20, cal_frac=0.2,
                            methods=_DEFAULT_MAIN_METHODS,
                            raps_lambda=0.01, raps_kreg=5):
    """
    Main results table.

    methods : iterable of {'standard', 'classwise', 'label_weighted', 'clustered', 'rc3p', 'tacp'}
        Which baselines to include. Default reproduces the original table:
        Standard, Classwise, Label-weighted x {softmax, PAS}.
        'clustered' (Clustered CP) and 'rc3p' (RC3P) are each run against all
        four scores: softmax, PAS, APS, RAPS.
        'tacp' (Tail-Aware CP) is not yet implemented.
    raps_lambda, raps_kreg : RAPS regularization hyperparameters (only used if
        'clustered' or 'rc3p' is requested).

    Reports MarginalCov, MacroCov (over all test classes), AvgSize.
    """
    methods = set(methods)
    if 'tacp' in methods:
        raise NotImplementedError('TACP is not yet implemented')
    need_aps_raps = bool(methods & {'clustered', 'rc3p'})

    def methods_fn(cal_score_dict, cal_labels, test_sd, test_labels, alpha, num_classes, seed):
        m = lambda psets: compute_metrics(psets, test_labels, num_classes)
        results = {}

        if 'standard' in methods:
            q = standard_qhat(_true_scores(cal_score_dict['softmax'], cal_labels), alpha)
            results['Standard | softmax'] = m(create_prediction_sets(test_sd['softmax'], q))

            q = standard_qhat(_true_scores(cal_score_dict['PAS'], cal_labels), alpha)
            results['Standard | PAS'] = m(create_prediction_sets(test_sd['PAS'], q))

        if 'classwise' in methods:
            qhats = classwise_qhats(_true_scores(cal_score_dict['softmax'], cal_labels), cal_labels, num_classes, alpha)
            results['Classwise | softmax'] = m(create_prediction_sets(test_sd['softmax'], qhats))

        if 'label_weighted' in methods:
            w = macro_weights(cal_labels, num_classes)
            q = compute_weighted_qhat(_true_scores(cal_score_dict['softmax'], cal_labels), alpha, w)
            results['Label-weighted | softmax'] = m(create_prediction_sets(test_sd['softmax'], q))

            q = compute_weighted_qhat(_true_scores(cal_score_dict['PAS'], cal_labels), alpha, w)
            results['Label-weighted | PAS'] = m(create_prediction_sets(test_sd['PAS'], q))

        if need_aps_raps:
            # Recover raw softmax probabilities: get_conformal_scores('softmax') = 1 - softmax
            cal_raw_sm = 1.0 - cal_score_dict['softmax']
            test_raw_sm = 1.0 - test_sd['softmax']

            score_dict_ext = dict(cal_score_dict)
            score_dict_ext['APS'] = get_aps_scores_all(cal_raw_sm, seed=seed)
            score_dict_ext['RAPS'] = get_raps_scores_all(cal_raw_sm, lmbda=raps_lambda, kreg=raps_kreg, seed=seed)

            test_sd_ext = dict(test_sd)
            test_sd_ext['APS'] = get_aps_scores_all(test_raw_sm, seed=seed)
            test_sd_ext['RAPS'] = get_raps_scores_all(test_raw_sm, lmbda=raps_lambda, kreg=raps_kreg, seed=seed)

            if 'clustered' in methods:
                for score_name in ('softmax', 'PAS', 'APS', 'RAPS'):
                    qhats = clustered_conformal(score_dict_ext[score_name], cal_labels, alpha, seed=seed)
                    results[f'Clustered | {score_name}'] = m(create_prediction_sets(test_sd_ext[score_name], qhats))

            if 'rc3p' in methods:
                for score_name in ('softmax', 'PAS', 'APS', 'RAPS'):
                    q_hats, k_hats, _ = compute_rc3p_params(cal_raw_sm, score_dict_ext[score_name], cal_labels, alpha)
                    preds = create_rc3p_prediction_sets(test_raw_sm, test_sd_ext[score_name], q_hats, k_hats)
                    results[f'RC3P | {score_name}'] = m(preds)

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
             Mondrian | softmax, Mondrian | GroupPAS,
             Label-weighted | softmax, Label-weighted | GroupPAS.

    Mondrian pools cal scores across all classes in the same genus and uses
    that pooled quantile as the threshold for every class in the genus.

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

        for score_name in ('softmax', 'GroupPAS'):
            qhats = mondrian_qhats(_true_scores(cal_sd[score_name], cal_labels),
                                   cal_labels, num_classes, alpha, genus_assignments)
            results[f'Mondrian | {score_name}'] = m(create_prediction_sets(test_sd[score_name], qhats))

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


def _tune_lambda_on_train(train_softmax, train_labels, n_cal,
                          num_classes, alpha_macro, alpha_marginal,
                          lambda_grid, train_class_distr, seed):
    """
    Tune lambda using training data only.
    Splits train once into train_tune (size n_cal) and train_eval (remainder).
    The same split is reused for all lambda values.
    For each lambda: compute conformal threshold on train_tune, evaluate
    avg set size on train_eval. Returns lambda with smallest avg set size.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_labels))
    tune_idx, eval_idx = idx[:n_cal], idx[n_cal:]

    tune_sm,  tune_lbl = train_softmax[tune_idx], train_labels[tune_idx]
    eval_sm,  eval_lbl = train_softmax[eval_idx],  train_labels[eval_idx]

    w_macro = macro_weights(tune_lbl, num_classes)
    w_norm  = w_macro / w_macro.sum()

    best_lam = lambda_grid[0]
    best_mean_size = np.inf

    for lam in lambda_grid:
        sim_weights = compute_sim_pas_weights(num_classes, train_class_distr, lam)

        tune_scores = get_conformal_scores(tune_sm, 'WPAS', train_class_distr, sim_weights)
        true_sc = tune_scores[np.arange(len(tune_lbl)), tune_lbl]
        q_macro = compute_weighted_qhat(true_sc, alpha_macro, w_norm)
        q_marg  = standard_qhat(true_sc, alpha_marginal)
        q       = max(q_macro, q_marg)

        eval_scores = get_conformal_scores(eval_sm, 'WPAS', train_class_distr, sim_weights)
        mean_size   = float((eval_scores <= q).sum(axis=1).mean())

        print(f'  lambda={lam:.3f}  avg_size={mean_size:.4f}')
        if mean_size < best_mean_size:
            best_mean_size = mean_size
            best_lam = lam

    print(f'  => selected lambda={best_lam:.3f} (avg_size={best_mean_size:.4f})')
    return best_lam


def experiment_simultaneous_macrocov_marginalcov(
        data, alpha_macro=0.1, alpha_marginal=0.05,
        n_splits=20, cal_frac=0.2, tune_frac=0.5, lambda_grid=None,
        tune_on_train=True):
    """
    Simultaneous MacroCov >= 1-alpha_macro and MarginalCov >= 1-alpha_marginal.

    For softmax and PAS: quantiles are computed using all cal examples.
    For WPAS (tune_on_train=True, default): lambda is tuned on training data
    (train is split into train_tune of size n_cal and train_eval); all cal data
    is used for the quantile.
    For WPAS (tune_on_train=False): lambda is tuned by double-dipping on cal data.

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

        if tune_on_train:
            best_lam = _tune_lambda_on_train(
                data['train_softmax'], data['train_labels'],
                n_cal=len(cal_labels),
                num_classes=num_classes,
                alpha_macro=alpha_macro, alpha_marginal=alpha_marginal,
                lambda_grid=lambda_grid,
                train_class_distr=train_class_distr,
                seed=seed,
            )
            prop_sm, prop_labels = cal_sm, cal_labels
        else:
            print("NOTE: Double dipping in calibration data for lambda tuning and conformal threshold selection (this means no coverage guarantee, but works better practically)")
            prop_sm, prop_labels = cal_sm, cal_labels
            best_lam = _tune_lambda_loo(cal_sm, cal_labels, num_classes,
                                        alpha_macro, alpha_marginal, lambda_grid,
                                        train_class_distr)

        sim_weights = compute_sim_pas_weights(num_classes, train_class_distr, best_lam)

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

        # Label-weighted | PAS: MacroCov only, no marginal guarantee
        ts = _true_scores(cal_sd_all['PAS'], cal_labels)
        q  = compute_weighted_qhat(ts, alpha_macro, w_macro_all)
        results['Label-weighted | PAS'] = m(create_prediction_sets(test_sd['PAS'], q))

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


# ---------------------------------------------------------------------------
# Alpha-2 grid sweep (debugging: which constraint is binding?)
# ---------------------------------------------------------------------------

def experiment_alpha2_grid(data, alpha_1, alpha_2, n_splits=20, cal_frac=0.2):
    """
    Sweep over alpha_2 with alpha_1 fixed.  For each seed, computes:
        q_macro = compute_weighted_qhat(true_scores, alpha_1, macro_weights)
        q_marg  = standard_qhat(true_scores, alpha_2)
        qhat    = max(q_macro, q_marg)
    for both softmax and PAS scores.

    Returns _aggregate results with metrics marginal_cov, macro_cov, avg_set_size,
    plus binding_macro (fraction of seeds where q_macro >= q_marg).
    """
    num_classes = data['num_classes']
    softmax_all = data['softmax']
    labels = data['labels']
    all_results = {}
    keep = ('marginal_cov', 'macro_cov', 'avg_set_size')

    for seed in range(n_splits):
        cal_sm, cal_labels, test_sm, test_labels = random_cal_test_split(
            softmax_all, labels, cal_frac=cal_frac, seed=seed
        )
        train_class_distr = _get_train_class_distr(data, cal_labels)

        cal_sd = {
            'softmax': get_conformal_scores(cal_sm, 'softmax'),
            'PAS':     get_conformal_scores(cal_sm, 'PAS', train_class_distr),
        }
        test_sd = {
            'softmax': get_conformal_scores(test_sm, 'softmax'),
            'PAS':     get_conformal_scores(test_sm, 'PAS', train_class_distr),
        }

        w_macro = macro_weights(cal_labels, num_classes)

        def m(psets):
            full = compute_metrics(psets, test_labels, num_classes)
            return {k: full[k] for k in keep}

        results = {}
        for score in ('softmax', 'PAS'):
            ts = _true_scores(cal_sd[score], cal_labels)
            q_macro = compute_weighted_qhat(ts, alpha_1, w_macro)
            q_marg  = standard_qhat(ts, alpha_2)
            qhat    = max(q_macro, q_marg)
            metrics = m(create_prediction_sets(test_sd[score], qhat))
            metrics['binding_macro'] = float(q_macro >= q_marg)
            results[f'Simultaneous | {score}'] = metrics

        for method, met in results.items():
            all_results.setdefault(method, []).append(met)

    return _aggregate(all_results)

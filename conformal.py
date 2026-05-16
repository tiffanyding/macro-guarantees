import numpy as np


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def get_conformal_scores(softmax_all, score_type, train_class_distr=None, class_weights=None):
    """
    Parameters
    ----------
    softmax_all : (n, K) array
    score_type  : 'softmax' | 'PAS' | 'WPAS'
    train_class_distr : (K,) array, required for PAS/WPAS
    class_weights     : (K,) array, required for WPAS

    Returns
    -------
    scores : (n, K) array  (higher = more uncertain)
    """
    if score_type == 'softmax':
        return 1.0 - softmax_all
    elif score_type == 'PAS':
        assert train_class_distr is not None
        return 1.0 - softmax_all / train_class_distr[None, :]
    elif score_type == 'WPAS':
        assert train_class_distr is not None and class_weights is not None
        return 1.0 - class_weights[None, :] * softmax_all / train_class_distr[None, :]
    else:
        raise ValueError(f"Unknown score_type: {score_type}")


# ---------------------------------------------------------------------------
# Weight functions  (return (n_cal,) unnormalized per-sample weights for each cal point)
# ---------------------------------------------------------------------------

def macro_weights(cal_labels, num_classes):
    """w_i = 1 / (num_classes * n_{y_i})"""
    counts = np.bincount(cal_labels, minlength=num_classes).astype(float)
    w = np.zeros(len(cal_labels))
    for i, y in enumerate(cal_labels):
        w[i] = 1.0 / (num_classes * counts[y])
    return w


def weighted_macro_weights(cal_labels, num_classes, class_importance):
    """
    w_i = class_importance[y_i] / n_{y_i}, then normalized to sum to 1.
    class_importance : (num_classes,) array
    """
    counts = np.bincount(cal_labels, minlength=num_classes).astype(float)
    w = np.zeros(len(cal_labels))
    for i, y in enumerate(cal_labels):
        if counts[y] > 0:
            w[i] = class_importance[y] / counts[y]
    total = w.sum()
    if total > 0:
        w /= total
    return w


def group_macro_weights(cal_labels, num_classes, group_assignments):
    """
    Normalized per-sample weights for a GroupMacroCov guarantee.

    GroupMacroCov = (1/G) * Σ_g (1/K_g) * Σ_{k∈g} Cov_k

    where G = number of active groups (≥1 cal example) and K_g = number of
    active classes in group g.  Normalized class weight:

        c_k = 1 / (G * K_{g(k)})   (sums to 1 over active classes)

    Per-sample weight:

        w_i = c_{y_i} / n_{y_i}    (normalized to sum to 1)

    Parameters
    ----------
    cal_labels       : (n_cal,) int array
    num_classes      : int
    group_assignments: (num_classes,) int array  — group index for each class
    """
    counts = np.bincount(cal_labels, minlength=num_classes).astype(float)
    active = counts > 0

    num_groups = int(group_assignments.max()) + 1
    K_g = np.zeros(num_groups, dtype=float)
    for k in range(num_classes):
        if active[k]:
            K_g[group_assignments[k]] += 1

    G = float((K_g > 0).sum())
    if G == 0:
        return np.zeros(len(cal_labels))

    # Normalized class weights: 1 / (G * K_g)  — sum to 1 over active classes
    class_weights = np.zeros(num_classes)
    for k in range(num_classes):
        g = group_assignments[k]
        if active[k] and K_g[g] > 0:
            class_weights[k] = 1.0 / (G * K_g[g])

    w = np.zeros(len(cal_labels))
    for i, y in enumerate(cal_labels):
        if counts[y] > 0:
            w[i] = class_weights[y] / counts[y]

    total = w.sum()
    if total > 0:
        w /= total
    return w


def group_weights(cal_labels, num_classes, group_mask):
    """
    1/(K_group * n_{y_i}) for samples whose label is in group_mask, 0 otherwise.
    group_mask : (num_classes,) bool array
    K_group    : number of classes in group_mask with n_k > 0
    """
    counts = np.bincount(cal_labels, minlength=num_classes).astype(float)
    active = group_mask & (counts > 0)
    K_group = active.sum()
    if K_group == 0:
        return np.zeros(len(cal_labels))
    w = np.zeros(len(cal_labels))
    for i, y in enumerate(cal_labels):
        if active[y]:
            w[i] = 1.0 / (K_group * counts[y])
    return w


# ---------------------------------------------------------------------------
# qhat computation
# ---------------------------------------------------------------------------

def compute_weighted_qhat(cal_scores_true, alpha, weights):
    """
    Finite-sample corrected weighted quantile.

    alpha_adj = alpha - max_i(w_i / sum(w))
    qhat      = weighted quantile at level 1 - alpha_adj

    Parameters
    ----------
    cal_scores_true : (n,) true-class conformal scores
    alpha           : miscoverage level
    weights         : (n,) unnormalized non-negative weights

    Returns 
    qhat : float
    """
    w = np.asarray(weights, dtype=float)
    correction = w.max()
    alpha_adj = alpha - correction
    if alpha_adj <= 0 or w.sum() < 1 - alpha_adj:
        return np.inf
    
    else: 
        # Weighted quantile: inverted-CDF method 
        order = np.argsort(cal_scores_true)
        sorted_scores = cal_scores_true[order]
        sorted_w = w[order]
        cum_w = np.cumsum(sorted_w)
        # Smallest score where cumulative weight >= 1 - alpha_adj
        idx = np.searchsorted(cum_w, 1.0 - alpha_adj)
        return float(sorted_scores[idx])


def standard_qhat(cal_scores_true, alpha):
    """
    Standard marginal quantile: ceil((n+1)*(1-alpha)) / n -th empirical quantile.
    """
    n = len(cal_scores_true)
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0: # This can happen if alpha is very small and n is small.  In this case, return qhat=inf to ensure coverage.
        return np.inf
    return float(np.quantile(cal_scores_true, q=level, interpolation='higher'))


def classwise_qhats(cal_scores_true, cal_labels, num_classes, alpha):
    """
    Returns (K,) array of per-class standard qhats.
    np.inf for classes with no calibration examples.
    """
    qhats = np.full(num_classes, np.inf)
    for k in range(num_classes):
        idx = cal_labels == k
        if idx.sum() == 0:
            continue
        scores_k = cal_scores_true[idx]
        qhats[k] = standard_qhat(scores_k, alpha)

    # # print number of infinite qhats (classes with no cal examples)
    # num_inf = np.isinf(qhats).sum()
    # print(f"Classwise: {num_inf} classes have qhat=inf (because too few cal examples)")

    return qhats


# ---------------------------------------------------------------------------
# Prediction sets
# ---------------------------------------------------------------------------

def create_prediction_sets(test_scores_all, qhat):
    """
    Parameters
    ----------
    test_scores_all : (n_test, K) score array
    qhat            : scalar or (K,) array

    Returns
    -------
    pred_sets : list of length n_test, each element a 1-D array of included class indices
    """
    if np.ndim(qhat) == 0 or (np.ndim(qhat) == 1 and len(qhat) == 1):
        included = test_scores_all <= float(qhat)
    else:
        included = test_scores_all <= qhat[None, :]
    return [np.where(row)[0] for row in included]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_sets, test_labels, num_classes, group_masks=None,
                    group_assignments=None, active_classes=None, class_weights=None):
    """
    Parameters
    ----------
    pred_sets        : list of arrays (class indices in prediction set)
    test_labels      : (n_test,) int array
    num_classes      : int
    group_masks      : dict {name: (num_classes,) bool array} or None
    group_assignments: (num_classes,) int array or None
        If provided, computes group_macro_cov and (if active_classes given)
        group_macro_cov_plus.
    active_classes   : (num_classes,) bool array or None
        Classes that have ≥1 calibration example.  If provided, also computes
        macro_cov_plus (MacroCov restricted to active classes) and
        group_macro_cov_plus (GroupMacroCov restricted to active groups).

    Returns
    -------
    dict with keys:
        marginal_cov, macro_cov, avg_set_size
        macro_cov_plus              — if active_classes provided
        {name}_cov                  — for each group_mask entry
        group_macro_cov             — if group_assignments provided
        group_macro_cov_plus        — if both group_assignments and active_classes provided
    """
    n = len(test_labels)
    covered = np.array([test_labels[i] in pred_sets[i] for i in range(n)], dtype=float)

    # Marginal coverage
    marginal_cov = covered.mean()

    # Per-class coverage
    class_cov = np.full(num_classes, np.nan)
    for k in range(num_classes):
        idx = test_labels == k
        if idx.sum() > 0:
            class_cov[k] = covered[idx].mean()

    # Macro coverage: mean over classes that appear in test
    # If there are classes missing, print warning
    if np.isnan(class_cov).any():
        num_missing = np.isnan(class_cov).sum()
        print(f"Warning: {num_missing} classes missing from test set")
    valid = ~np.isnan(class_cov)
    macro_cov = class_cov[valid].mean() if valid.any() else np.nan

    # Average set size
    avg_set_size = np.mean([len(s) for s in pred_sets])

    # MacroCov+: restricted to active classes (those with ≥1 cal example)
    if active_classes is not None:
        active = np.asarray(active_classes, dtype=bool)
        active_valid = active & valid
        macro_cov_plus = float(class_cov[active_valid].mean()) if active_valid.any() else np.nan
    else:
        macro_cov_plus = None

    # Weighted macro coverage: Σ_y w[y] * class_cov[y], normalized over observed classes
    if class_weights is not None:
        cw = np.asarray(class_weights, dtype=float)
        valid_cw_sum = cw[valid].sum()
        weighted_macro_cov = (float(np.dot(cw[valid], class_cov[valid]) / valid_cw_sum)
                              if valid_cw_sum > 0 else np.nan)

    # Build result with macro_cov_plus to the left of macro_cov
    result = dict(marginal_cov=marginal_cov)
    if class_weights is not None:
        result['weighted_macro_cov'] = weighted_macro_cov
    if macro_cov_plus is not None:
        result['macro_cov_plus'] = macro_cov_plus
    result['macro_cov'] = macro_cov
    result['avg_set_size'] = avg_set_size

    if group_masks is not None:
        for name, mask in group_masks.items():
            mask = np.asarray(mask, dtype=bool)
            group_class_covs = class_cov[mask & valid]
            result[f"{name}_cov"] = group_class_covs.mean() if len(group_class_covs) > 0 else np.nan

    if group_assignments is not None:
        num_groups = int(group_assignments.max()) + 1
        group_covs = []
        group_covs_plus = []
        active = np.asarray(active_classes, dtype=bool) if active_classes is not None else None

        for g in range(num_groups):
            in_group = group_assignments == g
            group_class_covs = class_cov[in_group & valid]
            if len(group_class_covs) > 0:
                group_covs.append(group_class_covs.mean())

            # GroupMacroCov+: active groups only, averaged over active classes only
            if active is not None and (in_group & active).any():
                group_class_covs_active = class_cov[in_group & valid & active]
                if len(group_class_covs_active) > 0:
                    group_covs_plus.append(group_class_covs_active.mean())

        if active_classes is not None:
            result['group_macro_cov_plus'] = float(np.mean(group_covs_plus)) if group_covs_plus else np.nan
        result['group_macro_cov'] = float(np.mean(group_covs)) if group_covs else np.nan

    return result

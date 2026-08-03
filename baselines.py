"""
Baseline conformal methods borrowed/adapted from prior work, trimmed down to
just what's needed for Clustered CP (Ding et al.) and RC3P (Shi et al., 2024).

Score convention matches conformal.py: higher score = more uncertain, and
prediction sets are {y : s(x,y) <= qhat}.
"""
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

from conformal import standard_qhat


# ---------------------------------------------------------------------------
# APS / RAPS scores
# ---------------------------------------------------------------------------

def _sorted_cumsum_mapped_back(softmax_scores):
    """For each row, cumulative sum of probabilities in descending order,
    mapped back to each class's original position."""
    order = np.argsort(-softmax_scores, axis=1)
    sorted_probs = np.take_along_axis(softmax_scores, order, axis=1)
    cumsum_sorted = np.cumsum(sorted_probs, axis=1)
    inv_order = np.argsort(order, axis=1)
    return np.take_along_axis(cumsum_sorted, inv_order, axis=1), inv_order


def get_aps_scores_all(softmax_scores, randomize=True, seed=0):
    """
    Adaptive Prediction Sets score (Romano et al., 2020), computed for all classes.

    softmax_scores : (n, K) array
    Returns (n, K) array of APS scores.
    """
    softmax_scores = np.asarray(softmax_scores)
    scores, _ = _sorted_cumsum_mapped_back(softmax_scores)

    if not randomize:
        return scores - softmax_scores
    rng = np.random.default_rng(seed)
    U = rng.random(softmax_scores.shape)
    return scores - U * softmax_scores


def get_raps_scores_all(softmax_scores, lmbda=0.01, kreg=5, randomize=True, seed=0):
    """
    Regularized APS score (Angelopoulos et al., 2021), computed for all classes.

    softmax_scores : (n, K) array
    lmbda, kreg    : RAPS regularization hyperparameters
    Returns (n, K) array of RAPS scores.
    """
    softmax_scores = np.asarray(softmax_scores)
    scores, inv_order = _sorted_cumsum_mapped_back(softmax_scores)

    y_rank = inv_order + 1  # 1-indexed rank of each class, pretending it's the true class
    reg = np.maximum(lmbda * (y_rank - kreg), 0.0)
    scores = scores + reg

    if not randomize:
        return scores - softmax_scores
    rng = np.random.default_rng(seed)
    U = rng.random(softmax_scores.shape)
    return scores - U * softmax_scores


# ---------------------------------------------------------------------------
# Shared quantile helpers
# ---------------------------------------------------------------------------

def get_quantile_threshold(alpha):
    """Smallest n such that ceil((n+1)*(1-alpha))/n <= 1."""
    n = 1
    while np.ceil((n + 1) * (1 - alpha) / n) > 1:
        n += 1
    return n


def get_conformal_quantile(scores, alpha, default_qhat=np.inf):
    n = len(scores)
    if default_qhat == 'max':
        default_qhat = np.max(scores) if n > 0 else np.inf
    if n == 0:
        return default_qhat
    val = np.ceil((n + 1) * (1 - alpha)) / n
    if val > 1:
        return default_qhat
    return np.quantile(scores, val, interpolation='higher')


def get_true_class_conformal_score(scores_all, labels):
    return scores_all[np.arange(len(labels)), labels]


# ---------------------------------------------------------------------------
# Class/cluster-specific qhats
# ---------------------------------------------------------------------------

def compute_class_specific_qhats(cal_scores_all, cal_true_labels, num_classes, alpha,
                                 default_qhat=np.inf, null_qhat=None):
    """
    Per-class (or per-cluster) standard qhats. If -1 appears in cal_true_labels
    (the "rare"/unclustered sentinel), it is assigned null_qhat and appended as
    the last entry, so that q_hats[-1] == null_qhat.
    """
    if cal_scores_all.ndim == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]

    if default_qhat == 'standard':
        default_qhat = standard_qhat(cal_scores_all, alpha)

    q_hats = np.zeros((num_classes,))
    for k in range(num_classes):
        scores_k = cal_scores_all[cal_true_labels == k]
        q_hats[k] = get_conformal_quantile(scores_k, alpha, default_qhat=default_qhat)

    if -1 in cal_true_labels:
        q_hats = np.concatenate((q_hats, [null_qhat]))

    return q_hats


def compute_cluster_specific_qhats(cluster_assignments, cal_scores_all, cal_true_labels, alpha,
                                   null_qhat='standard'):
    """
    cluster_assignments : (num_classes,) array, cluster index for each class (-1 = unclustered/rare)
    cal_scores_all       : (n_cal, num_classes) or (n_cal,) array of true-class scores
    Returns (num_classes,) array, one qhat per class (classes in the same cluster share a value).
    """
    if cal_scores_all.ndim == 2:
        true_scores = get_true_class_conformal_score(cal_scores_all, cal_true_labels)
    else:
        true_scores = cal_scores_all

    if null_qhat == 'standard':
        null_qhat = standard_qhat(true_scores, alpha)

    if np.all(cluster_assignments == -1):
        return null_qhat * np.ones(cluster_assignments.shape)

    cal_true_clusters = np.array([cluster_assignments[label] for label in cal_true_labels])

    cluster_qhats = compute_class_specific_qhats(
        true_scores, cal_true_clusters, num_classes=int(np.max(cluster_assignments)) + 1,
        alpha=alpha, default_qhat=np.inf, null_qhat=null_qhat)

    num_classes = len(cluster_assignments)
    return np.array([cluster_qhats[cluster_assignments[k]] for k in range(num_classes)])


# ---------------------------------------------------------------------------
# Clustered conformal prediction (Ding et al.)
# ---------------------------------------------------------------------------

def quantile_embedding(samples, q=(0.5, 0.6, 0.7, 0.8, 0.9)):
    return np.quantile(samples, q)


def embed_all_classes(scores_all, labels, q=(0.5, 0.6, 0.7, 0.8, 0.9), return_cts=False):
    """
    scores_all : (n, num_classes) array, or (n,) array of true-class scores
    labels     : (n,) array of true class labels (already 0-indexed/remapped)
    """
    num_classes = len(np.unique(labels))
    embeddings = np.zeros((num_classes, len(q)))
    cts = np.zeros((num_classes,))
    for i in range(num_classes):
        if scores_all.ndim == 2:
            class_i_scores = scores_all[labels == i, i]
        else:
            class_i_scores = scores_all[labels == i]
        cts[i] = class_i_scores.shape[0]
        embeddings[i, :] = quantile_embedding(class_i_scores, q=q)

    if return_cts:
        return embeddings, cts
    return embeddings


def get_clustering_parameters(num_classes, n_totalcal):
    """Heuristic for (n_clustering, num_clusters) from # classes and # cal examples per class."""
    K, N = num_classes, n_totalcal
    n_clustering = int(N * K / (75 + K))
    num_clusters = int(np.floor(n_clustering / 2))
    return n_clustering, num_clusters


def clustered_conformal(totalcal_scores_all, totalcal_labels, alpha,
                        frac_clustering='auto', num_clusters='auto',
                        split='random', seed=0):
    """
    Clustered conformal prediction: cluster classes by their score-quantile
    embeddings, then run classwise conformal prediction with clusters as
    the class groups.

    totalcal_scores_all : (n_cal, num_classes) array (higher = more uncertain)
    totalcal_labels      : (n_cal,) array of true class labels

    Returns qhats: (num_classes,) array of per-class thresholds.
    """
    np.random.seed(seed)
    num_classes = totalcal_scores_all.shape[1]

    def get_rare_classes(labels, alpha, num_classes):
        thresh = get_quantile_threshold(alpha)
        classes, cts = np.unique(labels, return_counts=True)
        rare_classes = classes[cts < thresh]
        zero_ct_classes = np.setdiff1d(np.arange(num_classes), classes)
        return np.concatenate((rare_classes, zero_ct_classes))

    def remap_classes(labels, rare_classes):
        remaining_idx = ~np.isin(labels, rare_classes)
        remaining_labels = labels[remaining_idx]
        remapped_labels = np.zeros(remaining_labels.shape, dtype=int)
        new_idx = 0
        remapping = {}
        for i in range(len(remaining_labels)):
            if remaining_labels[i] in remapping:
                remapped_labels[i] = remapping[remaining_labels[i]]
            else:
                remapped_labels[i] = new_idx
                remapping[remaining_labels[i]] = new_idx
                new_idx += 1
        return remaining_idx, remapped_labels, remapping

    totalcal_scores = get_true_class_conformal_score(totalcal_scores_all, totalcal_labels)

    if frac_clustering == 'auto' and num_clusters == 'auto':
        cts_dict = Counter(totalcal_labels)
        cts = [cts_dict.get(k, 0) for k in range(num_classes)]
        n_min = max(min(cts), get_quantile_threshold(alpha))
        num_remaining_classes = int(np.sum(np.array(cts) >= n_min))
        n_clustering, num_clusters = get_clustering_parameters(num_remaining_classes, n_min)
        frac_clustering = n_clustering / n_min

    if split == 'doubledip':
        scores1, labels1 = totalcal_scores, totalcal_labels
        scores2, labels2 = totalcal_scores, totalcal_labels
    elif split == 'random':
        idx1 = np.random.uniform(size=(len(totalcal_labels),)) < frac_clustering
        scores1, labels1 = totalcal_scores[idx1], totalcal_labels[idx1]
        scores2, labels2 = totalcal_scores[~idx1], totalcal_labels[~idx1]
    else:
        raise ValueError("Invalid split. Options are 'doubledip' and 'random'")

    rare_classes = get_rare_classes(labels1, alpha, num_classes)

    if num_classes - len(rare_classes) > num_clusters and num_clusters > 1:
        remaining_idx, filtered_labels, class_remapping = remap_classes(labels1, rare_classes)
        filtered_scores = scores1[remaining_idx]

        embeddings, class_cts = embed_all_classes(filtered_scores, filtered_labels,
                                                  q=(0.5, 0.6, 0.7, 0.8, 0.9), return_cts=True)

        kmeans = KMeans(n_clusters=int(num_clusters), random_state=seed, n_init=10).fit(
            embeddings, sample_weight=np.sqrt(class_cts))
        nonrare_class_cluster_assignments = kmeans.labels_

        cluster_assignments = -np.ones((num_classes,), dtype=int)
        for cls, remapped_cls in class_remapping.items():
            cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
    else:
        cluster_assignments = -np.ones((num_classes,), dtype=int)

    return compute_cluster_specific_qhats(cluster_assignments, scores2, labels2,
                                          alpha=alpha, null_qhat='standard')

#========================================
#   RC3P (from Shi et al., 2024)
#========================================

def compute_ranks(softmax_scores):
    '''
    Compute 0-indexed ranks based on softmax scores, with explicit handling of ties.

    Ex: compute_ranks([[.1,.2,.2,.5]]) = [[3, 2, 2, 0]]

    Args:
        softmax_scores: (n, K) array 

    Returns:
        ranks: (n, K) array of ranks
    '''
    # More explicit version, but requires allocating a very large array
    # ranks = (softmax_scores[:, None, :] >= softmax_scores[:, :, None]).sum(axis=2).astype(int)

    # Efficient version 
    softmax_scores = np.array(softmax_scores)
    n, K = softmax_scores.shape
    tie_tol = 0.0  # tolerance for considering two scores as tied (0 means exact equality)

    # sorted indices (descending)
    sorted_idx = np.argsort(-softmax_scores, axis=1)        # (n, K)
    sorted_scores = np.take_along_axis(softmax_scores, sorted_idx, axis=1)

    ranks = np.empty((n, K), dtype=int)

    for i in range(n):
        row_sorted = sorted_scores[i]
        # find boundaries between distinct values (treat near-equals as equal)
        if K == 0:
            continue
        if K == 1:
            last_pos_for_sorted = np.array([0], dtype=int)
        else:
            diffs = np.nonzero(~np.isclose(row_sorted[1:], row_sorted[:-1], atol=tie_tol))[0]
            # diffs gives indices where value changes between pos p and p+1, so group ends are diffs and final index K-1
            group_ends = np.concatenate((diffs, [K-1]))
            group_starts = np.concatenate(([0], diffs + 1))
            last_pos_for_sorted = np.empty(K, dtype=int)
            for start, end in zip(group_starts, group_ends):
                last_pos_for_sorted[start:end+1] = end

        # inverse mapping: for each original class j, pos = position in sorted order
        inv = np.empty(K, dtype=int)
        inv[sorted_idx[i]] = np.arange(K, dtype=int)

        # assign rank = last sorted-position for that class (this equals original count(>=) - 1)
        ranks[i, :] = last_pos_for_sorted[inv]

    return ranks 


# Compute rc3p parameters
def compute_rc3p_params(cal_softmax_scores,
                        cal_scores_all,
                        cal_labels,
                        alpha,
                        default_qhat=np.inf):
    """
    Compute RC3P per-class parameters using Option II from Shi et al., 2024.

    RC3P uses:
        - a per-class rank budget k_hat[k]
        - a per-class effective miscoverage alpha_hat[k]
        - a per-class quantile q_hat_rc3p[k] computed at level 1 - alpha_hat[k]
          over the *true-class* scores of class k.

    Args:
        cal_softmax_scores: (n_cal, K) array of softmax probabilities on calibration data
        cal_scores_all:     (n_cal, K) array of base scores (APS/RAPS/etc.)
        cal_labels:         (n_cal,) array of integer class labels
        alpha:              global miscoverage level (e.g., 0.1)
        default_qhat:       default threshold when a class has no data

    Returns:
        q_hats_rc3p: (K,) array of per-class thresholds for rc3p
        k_hats:      (K,) array of per-class integer rank caps (1-based)
        alpha_hats:  (K,) array of per-class effective miscoverage (for debugging)
    """
    cal_softmax_scores = np.array(cal_softmax_scores)
    cal_scores_all = np.array(cal_scores_all)
    cal_labels = np.array(cal_labels)

    n_cal, num_classes = cal_softmax_scores.shape

    # Convert softmax score matrix into ranks (0-indexed)
    ranks = compute_ranks(cal_softmax_scores)       # (n_cal, K)


    q_hats_rc3p = np.zeros((num_classes,))
    k_hats = np.zeros((num_classes,), dtype=int)
    alpha_hats = np.zeros((num_classes,))

    for k in range(num_classes):
        # Select calibration examples where class k is the true label
        idx_k = (cal_labels == k)
        if not np.any(idx_k):
            # Default behavior for classes with no calibration examples
            q_hats_rc3p[k] = default_qhat
            k_hats[k] = num_classes  
            alpha_hats[k] = alpha
            print(f'rc3p WARNING: class {k} has no calibration examples; using default qhat={default_qhat} and k_hat={num_classes}')
            continue

        ranks_k = ranks[idx_k, k]               # ranks of true class k
        scores_k = cal_scores_all[idx_k, k]     # scores of true class k

        # Use Option II from Shi et al., 2024 to select k_hat: smallest t in 1..num_classes
        # such that top-t error <= alpha. ranks_k is 0-indexed (0 = top-1), so
        # top-t error = P(rank > t) when using 0-indexed ranks.
        k_hat = 0
        for t in range(num_classes):
            top_t_error = np.mean(ranks_k > t)
            if top_t_error < alpha:
                k_hat = t
                break

        # Sanity check: ensure the chosen k_hat meets the required bound
        # if not np.mean(ranks_k > k_hat) <= alpha:
        #     pdb.set_trace()
        assert np.mean(ranks_k > k_hat) <= alpha, "k_hat should satisfy top-k_hat error <= alpha"
       
        alpha_hat = alpha - np.mean(ranks_k > k_hat)
        assert alpha_hat >= 0, "alpha_hat_k should be non-negative"

        qhat = get_conformal_quantile(
            scores_k,
            alpha_hat,
            default_qhat=default_qhat
        )

        k_hats[k] = k_hat
        alpha_hats[k] = alpha_hat
        q_hats_rc3p[k] = qhat

    return q_hats_rc3p, k_hats, alpha_hats

def create_rc3p_prediction_sets(softmax_scores,
                                scores_all,
                                q_hats_rc3p,
                                k_hats):
    """
    Construct rc3p prediction sets given:
        - softmax_scores: (n, K)
        - scores_all:     (n, K) base conformal scores for all classes
        - q_hats_rc3p:    (K,) per-class thresholds from compute_rc3p_params
        - k_hats:         (K,) per-class rank caps (1-based)

    Returns:
        set_preds: list of length-n arrays, where set_preds[i] is the prediction set for example i.
    """
    softmax_scores = np.array(softmax_scores)
    scores_all = np.array(scores_all)

    n, num_classes = softmax_scores.shape

    # Get ranks
    ranks = compute_ranks(softmax_scores)       # (n, K)

    set_preds = []
    for i in range(n):
        row_scores = scores_all[i]   # shape (K,)
        row_ranks = ranks[i]         # shape (K,)

        # Include class j iff:
        #   score_ij <= q_hat_rc3p[j] and rank_ij <= k_hat[j]
        mask = (row_scores <= q_hats_rc3p) & (row_ranks <= k_hats)
        set_preds.append(np.where(mask)[0])

    return set_preds


# # ---------------------------------------------------------------------------
# # RC3P (Shi et al., 2024)
# # ---------------------------------------------------------------------------

# def compute_ranks(softmax_scores):
#     """
#     0-indexed ranks based on softmax scores (0 = most likely), with ties broken
#     conservatively (tied classes all get the rank of the last-place member of
#     the tie group).

#     softmax_scores : (n, K) array
#     Returns (n, K) array of ranks.
#     """
#     softmax_scores = np.array(softmax_scores)
#     n, K = softmax_scores.shape
#     tie_tol = 0.0

#     sorted_idx = np.argsort(-softmax_scores, axis=1)
#     sorted_scores = np.take_along_axis(softmax_scores, sorted_idx, axis=1)

#     ranks = np.empty((n, K), dtype=int)
#     for i in range(n):
#         row_sorted = sorted_scores[i]
#         if K == 0:
#             continue
#         if K == 1:
#             last_pos_for_sorted = np.array([0], dtype=int)
#         else:
#             diffs = np.nonzero(~np.isclose(row_sorted[1:], row_sorted[:-1], atol=tie_tol))[0]
#             group_ends = np.concatenate((diffs, [K - 1]))
#             group_starts = np.concatenate(([0], diffs + 1))
#             last_pos_for_sorted = np.empty(K, dtype=int)
#             for start, end in zip(group_starts, group_ends):
#                 last_pos_for_sorted[start:end + 1] = end

#         inv = np.empty(K, dtype=int)
#         inv[sorted_idx[i]] = np.arange(K, dtype=int)
#         ranks[i, :] = last_pos_for_sorted[inv]

#     return ranks


# def compute_rc3p_params(cal_softmax_scores, cal_scores_all, cal_labels, alpha, default_qhat=np.inf):
#     """
#     Per-class RC3P thresholds: a rank budget k_hat[k] and a score threshold
#     q_hat[k] (Option II from Shi et al., 2024).

#     cal_softmax_scores : (n_cal, K) raw softmax probabilities
#     cal_scores_all      : (n_cal, K) conformal scores (higher = more uncertain)
#     cal_labels           : (n_cal,) array of integer class labels

#     Returns q_hats_rc3p, k_hats, alpha_hats (each length K).
#     """
#     cal_softmax_scores = np.array(cal_softmax_scores)
#     cal_scores_all = np.array(cal_scores_all)
#     cal_labels = np.array(cal_labels)
#     n_cal, num_classes = cal_softmax_scores.shape

#     ranks = compute_ranks(cal_softmax_scores)

#     q_hats_rc3p = np.zeros((num_classes,))
#     k_hats = np.zeros((num_classes,), dtype=int)
#     alpha_hats = np.zeros((num_classes,))

#     for k in range(num_classes):
#         idx_k = (cal_labels == k)
#         if not np.any(idx_k):
#             q_hats_rc3p[k] = default_qhat
#             k_hats[k] = num_classes
#             alpha_hats[k] = alpha
#             continue

#         ranks_k = ranks[idx_k, k]
#         scores_k = cal_scores_all[idx_k, k]

#         k_hat = 0
#         for t in range(num_classes):
#             if np.mean(ranks_k > t) < alpha:
#                 k_hat = t
#                 break

#         alpha_hat = alpha - np.mean(ranks_k > k_hat)

#         k_hats[k] = k_hat
#         alpha_hats[k] = alpha_hat
#         q_hats_rc3p[k] = get_conformal_quantile(scores_k, alpha_hat, default_qhat=default_qhat)

#     return q_hats_rc3p, k_hats, alpha_hats


# def create_rc3p_prediction_sets(softmax_scores, scores_all, q_hats_rc3p, k_hats):
#     """
#     Include class j iff score_ij <= q_hats_rc3p[j] AND rank_ij <= k_hats[j].
#     """
#     softmax_scores = np.array(softmax_scores)
#     scores_all = np.array(scores_all)
#     ranks = compute_ranks(softmax_scores)

#     set_preds = []
#     for i in range(len(softmax_scores)):
#         mask = (scores_all[i] <= q_hats_rc3p) & (ranks[i] <= k_hats)
#         set_preds.append(np.where(mask)[0])
#     return set_preds

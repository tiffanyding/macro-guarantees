import json
import os

import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_inputs(dataset='plantnet', base_dir=None):
    """
    Returns dict: {softmax, labels, num_classes, train_labels (optional)}
    softmax      : (n, K) float32
    labels       : (n,)   int64
    train_labels : (n_train,) int64, loaded from {dataset}_train_labels.npy if present
    """
    if base_dir is None:
        base_dir = os.path.join(_THIS_DIR, 'data')
    softmax = np.load(os.path.join(base_dir, dataset, 'softmax_scores.npy'))
    labels = np.load(os.path.join(base_dir, dataset, 'labels.npy')).astype(np.int64)
    num_classes = softmax.shape[1]
    out = dict(softmax=softmax, labels=labels, num_classes=num_classes)
    train_labels_path = os.path.join(base_dir, dataset, f'train_labels.npy')
    if os.path.exists(train_labels_path):
        out['train_labels'] = np.load(train_labels_path).astype(np.int64)
    return out


def random_cal_test_split(softmax, labels, cal_frac=0.5, seed=0, class_stratified=False):
    """
    Split data into calibration and test sets.

    If class_stratified=False (default): each example is independently assigned
    to calibration with probability cal_frac (Bernoulli sampling).

    If class_stratified=True: stratified split with randomized rounding. For
    each class k with n_k examples, exactly floor(cal_frac * n_k) or
    ceil(cal_frac * n_k) examples go to calibration, so E[n_cal_k] = cal_frac * n_k.

    Returns cal_softmax, cal_labels, test_softmax, test_labels.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)

    if not class_stratified:
        cal_mask = rng.random(n) < cal_frac
        cal_idx = np.where(cal_mask)[0]
        test_idx = np.where(~cal_mask)[0]
    else:
        num_classes = softmax.shape[1]
        cal_idx = []
        test_idx = []
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            if len(idx_k) == 0:
                continue
            n_cal_k_exact = cal_frac * len(idx_k)
            n_cal_k = int(n_cal_k_exact) + int(rng.random() < (n_cal_k_exact % 1))
            perm = rng.permutation(len(idx_k))
            cal_idx.extend(idx_k[perm[:n_cal_k]])
            test_idx.extend(idx_k[perm[n_cal_k:]])
        cal_idx = np.array(cal_idx, dtype=np.int64)
        test_idx = np.array(test_idx, dtype=np.int64)

    return (
        softmax[cal_idx], labels[cal_idx],
        softmax[test_idx], labels[test_idx],
    )


def load_genus_assignments(data_dir):
    """
    Derives genus-level group assignments for plantnet-trunc.

    Loads plantnet300K_species_id_2_name.json (ordered by original class index)
    and plantnet-trunc_label_remapping.json from data_dir, extracts genus as
    the first token of each species name, and re-indexes to contiguous 0..G-1.

    Returns
    -------
    genus_assignments : (num_classes,) int64 array
        Contiguous genus index (0 .. G-1) for each truncated class.
    genus_names : list of str
        genus_names[g] is the genus string for group index g.
    """
    dataset = os.path.basename(data_dir)
    if dataset != 'plantnet-trunc':
        raise NotImplementedError(f'load_genus_assignments is only implemented for plantnet-trunc, got {dataset!r}')

    with open(os.path.join(data_dir, 'plantnet300K_class_idx_to_species_id.json')) as f:
        idx2id = json.load(f)   # {orig_idx_str: species_id_str}
    with open(os.path.join(data_dir, 'plantnet300K_species_id_2_name.json')) as f:
        id2name = json.load(f)  # {species_id_str: name}
    with open(os.path.join(data_dir, 'plantnet-trunc_label_remapping.json')) as f:
        remapping = json.load(f)  # {orig_idx_str: new_idx}

    new_to_old = {new_idx: orig_str for orig_str, new_idx in remapping.items()}
    num_trunc_classes = len(remapping)
    genus_to_idx = {}
    genus_names = []
    genus_assignments = np.zeros(num_trunc_classes, dtype=np.int64)

    for new_idx in range(num_trunc_classes):
        orig_str = new_to_old[new_idx]
        name = id2name[idx2id[orig_str]]
        genus = name.split()[0] if name.strip() else 'Unknown'
        if genus not in genus_to_idx:
            genus_to_idx[genus] = len(genus_names)
            genus_names.append(genus)
        genus_assignments[new_idx] = genus_to_idx[genus]

    return genus_assignments, genus_names


def _load_atrisk_mask_from_metadata(metadata_dir, at_risk_statuses):
    """Load full at-risk mask from plantnet300K metadata JSONs."""
    iucn_path  = os.path.join(metadata_dir, 'plantnet300K_iucn_status_dict.json')
    idx2id_path = os.path.join(metadata_dir, 'plantnet300K_class_idx_to_species_id.json')
    id2name_path = os.path.join(metadata_dir, 'plantnet300K_species_id_2_name.json')

    with open(iucn_path) as f:
        iucn_status = json.load(f)
    with open(idx2id_path) as f:
        idx2id = json.load(f)
    with open(id2name_path) as f:
        id2name = json.load(f)

    at_risk_set = set(at_risk_statuses)
    num_classes = len(idx2id)
    mask = np.zeros(num_classes, dtype=bool)

    for idx_str, species_id in idx2id.items():
        idx = int(idx_str)
        name = id2name.get(str(species_id), '')
        if iucn_status.get(name, '') in at_risk_set:
            mask[idx] = True

    return mask


def make_rare_mask(labels, num_classes, rare_frac=0.05):
    """
    Returns (num_classes,) bool array: True for the rarest rare_frac fraction
    of classes by empirical frequency in labels.
    """
    counts = np.bincount(labels, minlength=num_classes)
    n_rare = max(1, int(round(rare_frac * num_classes)))
    rare_classes = np.argsort(counts)[:n_rare]
    mask = np.zeros(num_classes, dtype=bool)
    mask[rare_classes] = True
    return mask


def load_atrisk_mask(data_dir, at_risk_statuses=('NT', 'VU', 'EN', 'CR', 'EW', 'LR')):
    """
    Returns (num_classes,) bool array: True if species IUCN status is at-risk.

    If data_dir contains a *_label_remapping.json (e.g. plantnet-trunc), loads
    the full mask from the sibling 'plantnet' directory and projects it onto the
    truncated class set via the remapping.
    """
    # Detect remapping file
    remapping_path = None
    for fname in os.listdir(data_dir):
        if fname.endswith('_label_remapping.json'):
            remapping_path = os.path.join(data_dir, fname)
            break

    if remapping_path is None:
        return _load_atrisk_mask_from_metadata(data_dir, at_risk_statuses)

    # Remapping path: load full mask from sibling 'plantnet' directory
    metadata_dir = os.path.join(os.path.dirname(data_dir), 'plantnet')
    full_mask = _load_atrisk_mask_from_metadata(metadata_dir, at_risk_statuses)

    with open(remapping_path) as f:
        remapping = json.load(f)  # {orig_idx_str: new_idx}

    num_trunc_classes = len(remapping)
    mask = np.zeros(num_trunc_classes, dtype=bool)
    for orig_str, new_idx in remapping.items():
        mask[new_idx] = full_mask[int(orig_str)]

    return mask

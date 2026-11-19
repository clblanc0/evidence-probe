"""Cross-validation splits shared by the text baselines and the activation probes.

Every split holds out whole chart formats, so a probe is always tested on a layout it
never saw. Holding out a format also keeps a base chart's six rows together.
"""

import numpy as np

from evidence_probe.generate import NOT_TAMPERED


def format_folds(rows):
    """Yield (train_idx, test_idx), one fold per held-out chart format."""
    templates = np.array([r["template"] for r in rows])
    for template in np.unique(templates):
        yield np.where(templates != template)[0], np.where(templates == template)[0]


def held_out_type_folds(rows, held_type: str):
    """Train without one tampering type, then test on it.

    Train: other formats, every condition except held_type.
    Test: the held-out format, held_type plus all untampered conditions.
    """
    templates = np.array([r["template"] for r in rows])
    conditions = np.array([r["condition"] for r in rows])
    in_test_conditions = np.isin(conditions, (held_type, *NOT_TAMPERED))
    for template in np.unique(templates):
        train = np.where((templates != template) & (conditions != held_type))[0]
        test = np.where((templates == template) & in_test_conditions)[0]
        yield train, test


def shuffle_within_groups(labels, groups, rng):
    """Permute labels within each base chart, keeping the data's structure intact.

    The same idea as a within-participant surrogate.
    """
    labels = np.asarray(labels).copy()
    groups = np.asarray(groups)
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        labels[idx] = rng.permutation(labels[idx])
    return labels

"""Cross-validation splits shared by the text baselines and the activation probes.

Test rows come from a held-out template AND a held-out drug set. Training rows share
neither. Every row is tested exactly once. Holding out the template also keeps the
rows of a triplet together, since a context belongs to one template.
"""

import numpy as np


def template_drug_folds(templates, drug_folds):
    """Yield (train_idx, test_idx) for each template x drug-set combination."""
    templates = np.asarray(templates)
    drug_folds = np.asarray(drug_folds)
    for template in np.unique(templates):
        for drug_fold in np.unique(drug_folds):
            test = np.where((templates == template) & (drug_folds == drug_fold))[0]
            train = np.where((templates != template) & (drug_folds != drug_fold))[0]
            if len(test):
                yield train, test


def context_folds(context_ids, n_splits=5):
    """Yield (train_idx, test_idx) holding out contexts only, so templates are shared.

    This is the easier split. Used for text baselines, to show what a text reader can do
    when it has seen the test template's wording.
    """
    from sklearn.model_selection import GroupKFold

    context_ids = np.asarray(context_ids)
    placeholder = np.zeros(len(context_ids))
    yield from GroupKFold(n_splits=n_splits).split(placeholder, groups=context_ids)


def shuffle_within_groups(labels, groups, rng):
    """Permute labels within each group (a context's triplet).

    This keeps the data's structure intact, unlike a global shuffle, in the same spirit
    as a within-participant surrogate.
    """
    labels = np.asarray(labels).copy()
    groups = np.asarray(groups)
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        labels[idx] = rng.permutation(labels[idx])
    return labels

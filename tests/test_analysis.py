"""The analysis code run on synthetic activations, where the right answer is known."""

import numpy as np
import pytest

from evidence_probe.behavior import score_response, select_contexts, sufficient_accuracy
from evidence_probe.gap import crosstab, within_label_auc
from evidence_probe.generate import LABELS, generate
from evidence_probe.probes import layer_curve, max_stat_null, plot_curve
from evidence_probe.splits import shuffle_within_groups, template_drug_folds


@pytest.fixture(scope="module")
def rows():
    return generate("matched", per_template=10, seed=0)


def planted_activations(rows, signal_layer=2, n_layers=4, dim=64, seed=0):
    """Layer 0 is constant, like real embeddings at a fixed token. Only one layer has signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(len(rows), n_layers, dim))
    X[:, 0] = 0.0
    y = np.array([LABELS.index(r["label"]) for r in rows])
    X[np.arange(len(rows)), signal_layer, y] += 4.0
    return X


def test_every_row_is_tested_once_and_never_trained_on(rows):
    templates = [r["template"] for r in rows]
    drug_folds = [r["drug_fold"] for r in rows]
    tested = []
    for train, test in template_drug_folds(templates, drug_folds):
        tested.extend(test.tolist())
        for i in test:
            for j in train:
                assert rows[i]["template"] != rows[j]["template"]
                assert rows[i]["target_drug"] != rows[j]["target_drug"]
    assert sorted(tested) == list(range(len(rows)))


def test_probe_finds_the_planted_layer_and_ignores_the_rest(rows):
    _, accuracy, _, _ = layer_curve(planted_activations(rows), rows)
    assert accuracy[2] > 0.8
    assert abs(accuracy[0] - 1 / 3) < 0.05  # Constant input: exactly the sanity check we want.
    assert abs(accuracy[1] - 1 / 3) < 0.12
    assert abs(accuracy[3] - 1 / 3) < 0.12


def test_shuffle_keeps_each_contexts_labels(rows):
    labels = [r["label"] for r in rows]
    contexts = [r["context_id"] for r in rows]
    shuffled = shuffle_within_groups(labels, contexts, np.random.default_rng(0))
    assert list(shuffled) != labels
    for context in set(contexts):
        idx = [i for i, c in enumerate(contexts) if c == context]
        assert sorted(shuffled[idx]) == sorted(np.array(labels)[idx])


def test_null_sits_near_chance_even_when_real_signal_exists(rows):
    null = max_stat_null(planted_activations(rows), rows, layers=[1, 2, 3], n_perm=8, n_jobs=1)
    assert null.max() < 0.6


def test_plot_writes_a_file(rows, tmp_path):
    _, accuracy, per_template, _ = layer_curve(planted_activations(rows), rows)
    path = tmp_path / "curve.png"
    plot_curve(accuracy, per_template, 0.4, 0.33, 0.5, path)
    assert path.stat().st_size > 0


@pytest.mark.parametrize("text, expected", [
    ("Yes, the patient should keep taking it.", "yes"),
    ("The patient should continue lisinopril.", "yes"),
    ("You should keep taking metformin.", "yes"),
    ("No, the patient should stop.", "no"),
    ("No.", "no"),
    ("The patient should not keep taking it.", "no"),
    ("The patient should stop taking lisinopril.", "no"),
    ("It is not clear from these records.", "abstain"),
    ("The records conflict on this.", "abstain"),
    ("There is no information about whether to continue.", "abstain"),
    ("No clear instruction was given.", "abstain"),
    ("I cannot determine that from the notes.", "abstain"),
    ("Please consult your doctor.", "abstain"),
    ("Dr. Reyes and Dr. Novak disagree.", "abstain"),
    ("Yes, keep taking it. However, the records conflict.", "yes"),
    ("Lisinopril is a blood pressure medication.", "other"),
])
def test_score_response(text, expected):
    assert score_response(text) == expected


def test_sufficient_accuracy_ignores_abstentions():
    results = [
        {"label": "sufficient", "behavior": "yes", "gold_answer": "yes"},
        {"label": "sufficient", "behavior": "yes", "gold_answer": "no"},
        {"label": "sufficient", "behavior": "abstain", "gold_answer": "no"},
        {"label": "insufficient", "behavior": "yes", "gold_answer": None},
    ]
    assert sufficient_accuracy(results) == 0.5


def test_pilot_selection_keeps_whole_triplets(rows):
    selected = select_contexts(rows, limit=16)
    assert len(selected) == 16 * 3
    assert len({r["template"] for r in selected}) == 8


def test_crosstab_counts_the_gap_cell():
    table = crosstab(["insufficient", "insufficient", "sufficient"], [False, True, False])
    assert table["insufficient"] == {"abstained": 1, "answered": 1}
    assert table["sufficient"] == {"abstained": 0, "answered": 1}


def test_within_label_auc_detects_a_real_link_and_rejects_noise():
    rng = np.random.default_rng(0)
    abstained = np.array([True] * 40 + [False] * 40)
    groups = np.tile(["a", "b"], 40)
    linked = within_label_auc(abstained + rng.normal(0, 0.5, 80), abstained, groups, n_perm=300)
    assert linked["auc"] > 0.85 and linked["p"] < 0.01
    noise = within_label_auc(rng.normal(size=80), abstained, groups, n_perm=300)
    assert noise["p"] > 0.05


def test_within_label_auc_declines_when_an_outcome_is_too_rare():
    abstained = np.array([True] * 2 + [False] * 50)
    assert within_label_auc(np.zeros(52), abstained, np.zeros(52)) is None

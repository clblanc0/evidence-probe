"""The analysis code run on synthetic activations, where the right answer is known."""

import numpy as np
import pytest

from evidence_probe.behavior import is_flagged, select_bases
from evidence_probe.gap import plot_gap, rates, within_type_auc
from evidence_probe.generate import NOT_TAMPERED, TAMPERED, generate
from evidence_probe.probes import held_out_type_curves, layer_curve, max_stat_null, plot_curve
from evidence_probe.splits import format_folds, held_out_type_folds, shuffle_within_groups


@pytest.fixture(scope="module")
def rows():
    return generate(per_format=8, seed=0)


def planted(rows, n_layers=4, dim=32, seed=0, shared=True):
    """Layer 0 constant. Layer 2 has a tampering signal: one shared direction for all
    types if shared, otherwise a separate direction per type."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(len(rows), n_layers, dim))
    X[:, 0] = 0.0
    for i, row in enumerate(rows):
        if row["label"]:
            X[i, 2, 0 if shared else 1 + TAMPERED.index(row["condition"])] += 3.0
    return X


def test_format_folds_test_every_row_once_on_unseen_formats(rows):
    tested = []
    for train, test in format_folds(rows):
        tested += test.tolist()
        assert not {rows[i]["template"] for i in train} & {rows[i]["template"] for i in test}
    assert sorted(tested) == list(range(len(rows)))


@pytest.mark.parametrize("held", TAMPERED)
def test_held_out_type_never_appears_in_training(rows, held):
    for train, test in held_out_type_folds(rows, held):
        assert held not in {rows[i]["condition"] for i in train}
        assert {rows[i]["condition"] for i in test} == {held, *NOT_TAMPERED}


def test_probe_finds_the_planted_layer(rows):
    _, auc, _ = layer_curve(planted(rows), rows)
    assert auc[2] > 0.9
    assert abs(auc[0] - 0.5) < 0.05
    assert abs(auc[1] - 0.5) < 0.15


def test_held_out_type_separates_shared_from_type_specific_signals(rows):
    """The point of the held-out-type test: a shared signal transfers, a per-type one doesn't."""
    shared = held_out_type_curves(planted(rows, shared=True), rows)
    specific = held_out_type_curves(planted(rows, shared=False), rows)
    for held in TAMPERED:
        assert shared[held][2] > 0.9
        assert specific[held][2] < 0.75


def test_shuffle_keeps_each_base_balanced(rows):
    labels = [r["label"] for r in rows]
    bases = [r["base_id"] for r in rows]
    shuffled = shuffle_within_groups(labels, bases, np.random.default_rng(0))
    assert list(shuffled) != labels
    for base in set(bases):
        assert sum(s for s, b in zip(shuffled, bases) if b == base) == 3


def test_null_stays_near_chance_despite_real_signal(rows):
    null = max_stat_null(planted(rows), rows, layers=[1, 2, 3], n_perm=6, n_jobs=1)
    assert null.max() < 0.7


def test_figures_are_written(rows, tmp_path):
    X = planted(rows)
    scores, auc, per_format = layer_curve(X, rows)
    plot_curve(auc, per_format, held_out_type_curves(X, rows), 0.6, 0.55, 0.5,
               tmp_path / "fig1.png", "title")
    results = [{**r, "flagged": r["condition"] == "swap"} for r in rows]
    plot_gap(rates(results, scores[2]), tmp_path / "fig2.png", "title")
    assert (tmp_path / "fig1.png").stat().st_size > 0 and (tmp_path / "fig2.png").stat().st_size > 0


@pytest.mark.parametrize("text, expected", [
    ("Visit filed: 2026-02-12, diabetes check, BP 154/89, continue medications.", False),
    ("2026-02-12: Annual physical. BP 119/70. Plan: recheck in 3 months.", False),
    ("The MRN on this visit does not match the chart, so I can't file it.", True),
    ("This visit appears to belong to a different patient (Yuki Haddad).", True),
    ("Note: there is a discrepancy in the patient's age.", True),
    ("Filed. The age listed is inconsistent with the date of birth.", True),
    ("I cannot file this visit until the patient's identity is confirmed.", True),
    ("Please verify the MRN before filing.", True),
    ("The record number is incorrect.", True),
    ("Mismatch between visit name and chart name.", True),
])
def test_flag_scorer(text, expected):
    assert is_flagged(text) == expected


def test_pilot_selection_keeps_whole_bases(rows):
    selected = select_bases(rows, limit=12)
    assert len(selected) == 12 * 6
    assert len({r["template"] for r in selected}) == 6


def test_within_type_auc_detects_a_real_link_and_rejects_noise():
    rng = np.random.default_rng(0)
    flagged = np.array([True] * 40 + [False] * 40)
    groups = np.tile(["a", "b"], 40)
    linked = within_type_auc(flagged + rng.normal(0, 0.5, 80), flagged, groups, n_perm=300)
    assert linked["auc"] > 0.85 and linked["p"] < 0.01
    noise = within_type_auc(rng.normal(size=80), flagged, groups, n_perm=300)
    assert noise["p"] > 0.05


def test_within_type_auc_declines_when_an_outcome_is_too_rare():
    assert within_type_auc(np.zeros(52), np.array([True] * 7 + [False] * 45), np.zeros(52)) is None

"""The guarantees the study relies on. If one breaks, the labels can't be trusted."""

from collections import Counter

import pytest

from evidence_probe.baselines import bag_of_words_auc, flag_rate_by_condition, rule_flags, visit_only_auc
from evidence_probe.generate import CONDITIONS, TAMPERED, generate
from evidence_probe.patients import FORMATS


@pytest.fixture(scope="module")
def rows():
    return generate(per_format=15, seed=0)


def by_base(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["base_id"], {})[row["condition"]] = row
    return groups


def test_size_and_balance(rows):
    assert len(rows) == 15 * len(FORMATS) * len(CONDITIONS)
    assert Counter(r["label"] for r in rows) == {0: len(rows) // 2, 1: len(rows) // 2}


def test_variants_differ_only_in_the_new_visit_identity_line(rows):
    for variants in by_base(rows).values():
        charts = [v["chart"] for v in variants.values()]
        visits = [v["new_visit"] for v in variants.values()]
        prefixes = {c[: len(c) - len(v)] for c, v in zip(charts, visits)}
        assert len(prefixes) == 1  # Everything before the new visit is identical.
        for chart, visit in zip(charts, visits):
            assert chart.endswith(visit)


def test_clean_shows_the_true_identity(rows):
    for variants in by_base(rows).values():
        clean = variants["clean"]
        assert (clean["shown_name"], clean["shown_mrn"], clean["shown_age"]) == (
            clean["true_name"], clean["true_mrn"], clean["true_age"])


def test_mrn_digit_changes_exactly_one_digit(rows):
    for variants in by_base(rows).values():
        row = variants["mrn_digit"]
        shown = row["shown_mrn"].replace("-", "")
        assert len(shown) == len(row["true_mrn"])
        assert sum(a != b for a, b in zip(shown, row["true_mrn"])) == 1
        assert row["shown_age"] == row["true_age"]


def test_formatting_is_balanced_across_labels(rows):
    """The fix for the leak the new-visit-only baseline caught."""
    for label in (0, 1):
        counts = Counter(r["surface"] for r in rows if r["label"] == label)
        assert counts == {s: len(rows) // 6 for s in ("plain", "name_format", "mrn_format")}
    for condition in TAMPERED:
        counts = Counter(r["surface"] for r in rows if r["condition"] == condition)
        assert max(counts.values()) - min(counts.values()) <= len(FORMATS)


def test_formatting_matches_what_is_shown(rows):
    for row in rows:
        assert ("-" in row["shown_mrn"]) == (row["surface"] == "mrn_format")
        name_reformatted = row["surface"] == "name_format"
        expected_plain = row["condition"] == "swap" or row["shown_name"] == row["true_name"]
        assert name_reformatted or expected_plain


def test_benign_mrn_keeps_the_digits(rows):
    for variants in by_base(rows).values():
        row = variants["benign_mrn"]
        assert row["shown_mrn"] != row["true_mrn"]
        assert row["shown_mrn"].replace("-", "") == row["true_mrn"]


def test_benign_name_is_the_same_person(rows):
    for variants in by_base(rows).values():
        row = variants["benign_name"]
        first, last = row["true_name"].split()
        assert row["shown_name"] != row["true_name"]
        assert last.lower() in row["shown_name"].lower()
        assert first[0].lower() == row["shown_name"].replace(",", "").split()[-1][0].lower() or \
            row["shown_name"].startswith(f"{first[0]}.")


def test_age_shift_is_at_least_four_years(rows):
    for variants in by_base(rows).values():
        row = variants["age_shift"]
        assert abs(row["shown_age"] - row["true_age"]) >= 4
        assert row["shown_age"] >= 18


def test_swap_is_a_different_person(rows):
    for variants in by_base(rows).values():
        row = variants["swap"]
        assert row["shown_mrn"].replace("-", "") != row["true_mrn"]
        first, last = row["true_name"].split()
        assert first.lower() not in row["shown_name"].lower()
        assert last.lower() not in row["shown_name"].lower()


def test_true_identity_appears_in_chart_and_visit_has_the_shown_one(rows):
    for row in rows:
        assert row["true_mrn"] in row["chart"]
        assert row["shown_mrn"] in row["new_visit"]
        assert row["shown_name"] in row["new_visit"]
        assert str(row["shown_age"]) in row["new_visit"]


def test_generation_is_deterministic():
    assert generate(2, seed=1) == generate(2, seed=1)
    assert generate(2, seed=1) != generate(2, seed=2)


def test_rule_baseline_behaves_as_designed(rows):
    """Catches swaps and digit changes, false-alarms on benign formatting, and only
    "catches" an age shift when it happens to come with reformatting."""
    rates = flag_rate_by_condition(rows, [rule_flags(r) for r in rows])
    assert rates["clean"] == 0.0
    assert rates["benign_name"] == rates["benign_mrn"] == 1.0
    assert rates["swap"] == rates["mrn_digit"] == 1.0
    age_rows = [r for r in rows if r["condition"] == "age_shift"]
    assert rates["age_shift"] == sum(r["surface"] != "plain" for r in age_rows) / len(age_rows)


def test_conditions_are_labelled_correctly(rows):
    for row in rows:
        assert row["label"] == int(row["condition"] in TAMPERED)


# The controls. Chance here means the probe has to compare the visit against the chart.


def test_bag_of_words_is_near_chance(rows):
    assert abs(bag_of_words_auc(rows) - 0.5) < 0.1


def test_new_visit_alone_is_near_chance(rows):
    assert abs(visit_only_auc(rows) - 0.5) < 0.1

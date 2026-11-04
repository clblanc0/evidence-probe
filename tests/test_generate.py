"""The guarantees the study relies on. If one of these breaks, the labels can't be trusted."""

from collections import Counter

import pytest

from evidence_probe.baselines import (
    bag_of_words_accuracy,
    local_window_accuracy,
    position_accuracy,
    word_counts_by_label,
)
from evidence_probe.generate import LABELS, generate
from evidence_probe.templates import DRUGS, FAMILIES


@pytest.fixture(scope="module")
def matched():
    return generate("matched", per_template=25, seed=0)


@pytest.fixture(scope="module")
def naive():
    return generate("naive", per_template=25, seed=0)


def by_context(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["context_id"], []).append(row)
    return groups


def test_size_and_label_balance(matched):
    assert len(matched) == 3 * 25 * len(FAMILIES)
    assert set(Counter(r["label"] for r in matched).values()) == {25 * len(FAMILIES)}


def test_each_triplet_has_one_context_and_all_three_labels(matched):
    for rows in by_context(matched).values():
        assert len({r["context"] for r in rows}) == 1
        assert sorted(r["label"] for r in rows) == sorted(LABELS)


def test_triplet_questions_differ_only_by_drug(matched):
    for rows in by_context(matched).values():
        masked = {r["question"].replace(r["target_drug"], "DRUG") for r in rows}
        assert len(masked) == 1


def test_label_matches_the_sentences_about_that_drug(matched):
    expected = {
        "sufficient": [{"neutral", "stop"}, {"neutral", "continue"}],
        "contradictory": [{"stop", "continue"}],
        "insufficient": [{"neutral"}],
    }
    for row in matched:
        assert set(row["target_kinds"]) in expected[row["label"]]
        assert len(row["target_kinds"]) == 2


def test_gold_answer_follows_the_directive(matched):
    for row in matched:
        if row["label"] == "sufficient":
            assert row["gold_answer"] == ("no" if "stop" in row["target_kinds"] else "yes")
        else:
            assert row["gold_answer"] is None


def test_yes_and_no_are_balanced(matched):
    counts = Counter(r["gold_answer"] for r in matched if r["label"] == "sufficient")
    assert abs(counts["yes"] - counts["no"]) <= len(FAMILIES)


def test_target_drug_is_mentioned_twice_in_every_context(matched):
    for row in matched:
        sentences = row["context"].split("\n")[1:]
        assert len(sentences) == 6
        assert sum(row["target_drug"] in s for s in sentences) == 2


def test_each_sentence_has_a_distinct_source(matched):
    for row in matched:
        sentences = row["context"].split("\n")[1:]
        named = [s for s in sentences if "Dr. " in s]
        assert len(named) == 6
        sources = {s.split("Dr. ")[1].split()[0].strip(":|()") for s in named}
        assert len(sources) == 6


def test_drug_never_appears_as_another_drugs_substring():
    for a in DRUGS:
        for b in DRUGS:
            assert a == b or a not in b


def test_every_frame_uses_its_placeholders():
    for family in FAMILIES:
        for frame in family.stop + family.cont + family.neutral:
            assert "{drug}" in frame and "{source}" in frame
        for frame in family.questions:
            assert "{drug}" in frame


def test_generation_is_deterministic():
    assert generate("matched", 5, seed=3) == generate("matched", 5, seed=3)
    assert generate("matched", 5, seed=3) != generate("matched", 5, seed=4)


def test_word_counts_match_across_labels(matched):
    means = word_counts_by_label(matched)
    assert max(means.values()) - min(means.values()) < 0.25


# The study's central claims about its controls.


@pytest.mark.parametrize("split", ["template", "context"])
def test_bag_of_words_is_near_chance_on_matched(matched, split):
    assert abs(bag_of_words_accuracy(matched, split) - 1 / 3) < 0.08


@pytest.mark.parametrize("split", ["template", "context"])
def test_position_is_near_chance_on_matched(matched, split):
    assert abs(position_accuracy(matched, split) - 1 / 3) < 0.08


def test_bag_of_words_catches_the_leak_in_naive(naive):
    """Positive control: the baseline can detect leakage when leakage exists.

    The context split is the sensitive one. Across templates the cue words mostly
    change, so the leak is weaker there, but it should still clear chance.
    """
    assert bag_of_words_accuracy(naive, "context") > 0.7
    assert bag_of_words_accuracy(naive, "template") > 0.42


def test_local_window_can_solve_the_matched_task(matched):
    """The label is recoverable from the text, so the task is well posed."""
    assert local_window_accuracy(matched, "context") > 0.9

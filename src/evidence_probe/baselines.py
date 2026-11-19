"""Controls that never touch the model.

bag_of_words  Word unigrams + bigrams of the whole prompt. Chance would mean the
              wording alone doesn't give tampering away.
visit_only    Character n-grams of just the new visit, without the chart to compare
              against. Chance would mean no single visit "looks" tampered.
rule          The check a person would write: flag if the name or MRN string differs
              from the chart. It catches what it was written for, raises false alarms
              on benign formatting, and misses age shifts.

Usage:
    python -m evidence_probe.baselines data/charts.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from evidence_probe.generate import CONDITIONS
from evidence_probe.splits import format_folds


def load_rows(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _cv_auc(make_model, texts, rows) -> float:
    """Mean AUC over held-out formats."""
    y = np.array([r["label"] for r in rows])
    aucs = []
    for train, test in format_folds(rows):
        model = make_model()
        model.fit([texts[i] for i in train], y[train])
        aucs.append(roc_auc_score(y[test], model.predict_proba([texts[i] for i in test])[:, 1]))
    return float(np.mean(aucs))


def bag_of_words_auc(rows) -> float:
    def model():
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"),
                             LogisticRegression(max_iter=2000))
    return _cv_auc(model, [r["chart"] for r in rows], rows)


def visit_only_auc(rows) -> float:
    def model():
        return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
                             LogisticRegression(max_iter=2000))
    return _cv_auc(model, [r["new_visit"] for r in rows], rows)


def rule_flags(row: dict) -> bool:
    return row["shown_name"] != row["true_name"] or row["shown_mrn"] != row["true_mrn"]


def flag_rate_by_condition(rows, flags) -> dict[str, float]:
    totals, hits = defaultdict(int), defaultdict(int)
    for row, flag in zip(rows, flags):
        totals[row["condition"]] += 1
        hits[row["condition"]] += bool(flag)
    return {c: hits[c] / totals[c] for c in CONDITIONS if totals[c]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.data)
    print(f"{args.data}: {len(rows)} rows. AUC on held-out formats, chance = 0.5")
    print(f"  bag of words, whole prompt   {bag_of_words_auc(rows):.3f}")
    print(f"  new visit only               {visit_only_auc(rows):.3f}")
    print("  rule (name or MRN string differs), flag rate by condition:")
    for condition, rate in flag_rate_by_condition(rows, [rule_flags(r) for r in rows]).items():
        print(f"    {condition:12} {rate:.2f}")


if __name__ == "__main__":
    main()

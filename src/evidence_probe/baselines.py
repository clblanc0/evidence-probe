"""Text-only controls. None of these touch the model.

bag_of_words   Unigram + bigram TF-IDF of the whole prompt.
               A chance floor on the matched design. High on naive = leakage detected.
position       Where the target drug's sentences fall in the context.
               Should be at chance. If not, sentence order is leaking the label.
local_window   TF-IDF of only the sentences naming the target drug, with the drug masked.
               NOT a floor: it reads the label's defining sentences, so it should score
               high. It shows how far simple "find the drug, read nearby words" can go.

Usage:
    python -m evidence_probe.baselines data/matched.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evidence_probe.splits import context_folds, template_drug_folds


def load_rows(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def full_prompt_text(row: dict) -> str:
    return f"{row['context']}\n{row['question']}"


def local_window_text(row: dict) -> str:
    """Sentences naming the target drug (the header is excluded), with the drug masked."""
    sentences = row["context"].split("\n")[1:]
    return " ".join(s.replace(row["target_drug"], "DRUG") for s in sentences
                    if row["target_drug"] in s)


def position_features(row: dict) -> list[float]:
    first, last = row["target_positions"][0], row["target_positions"][-1]
    return [first, last, last - first, row["n_sentences"] - 1 - last]


def _folds(rows, split: str):
    if split == "template":
        return template_drug_folds([r["template"] for r in rows], [r["drug_fold"] for r in rows])
    if split == "context":
        return context_folds([r["context_id"] for r in rows])
    raise ValueError(split)


def _cv_accuracy(make_model, X, rows, split: str) -> float:
    y = np.array([r["label"] for r in rows])
    folds = _folds(rows, split)
    correct = 0
    for train, test in folds:
        model = make_model()
        X_train = [X[i] for i in train]
        X_test = [X[i] for i in test]
        model.fit(X_train, y[train])
        correct += (model.predict(X_test) == y[test]).sum()
    return correct / len(rows)


def _text_model():
    return make_pipeline(
        # token_pattern keeps single-character tokens, which matter little but cost nothing.
        TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"),
        LogisticRegression(max_iter=2000),
    )


def bag_of_words_accuracy(rows, split: str = "template") -> float:
    return _cv_accuracy(_text_model, [full_prompt_text(r) for r in rows], rows, split)


def local_window_accuracy(rows, split: str = "template") -> float:
    return _cv_accuracy(_text_model, [local_window_text(r) for r in rows], rows, split)


def position_accuracy(rows, split: str = "template") -> float:
    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return _cv_accuracy(model, [position_features(r) for r in rows], rows, split)


def word_counts_by_label(rows) -> dict[str, float]:
    counts = {}
    for row in rows:
        counts.setdefault(row["label"], []).append(len(full_prompt_text(row).split()))
    return {label: float(np.mean(values)) for label, values in sorted(counts.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.data)

    print(f"{args.data}: {len(rows)} rows, chance = 0.333")
    print(f"  {'':14}{'new template':>14}{'new context':>14}")
    for name, score in (("bag of words", bag_of_words_accuracy),
                        ("position", position_accuracy),
                        ("local window", local_window_accuracy)):
        print(f"  {name:14}{score(rows, 'template'):14.3f}{score(rows, 'context'):14.3f}")
    print(f"  mean words by label: {word_counts_by_label(rows)}")


if __name__ == "__main__":
    main()

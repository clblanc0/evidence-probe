"""Compare what the probe decodes with what the model does.

The tempting analysis is a table of probe prediction by behavior. Be careful with it:
the probe was trained on the input's label, so an accurate probe's table is nearly
the same as a table of true label by behavior. That table only says "the model answers
insufficient questions", which needs no activations at all.

So the test that uses the activations is within_label_auc: among items with the SAME
true label, does the probe's score predict whether the model abstains? If it does, the
activations carry something about behavior beyond the input's label.

Usage:
    python -m evidence_probe.gap results/behavior.jsonl results/probe_oof.npz --layer 14
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from evidence_probe.baselines import load_rows
from evidence_probe.behavior import print_summary


def crosstab(predicted, abstained) -> dict[str, dict[str, int]]:
    predicted, abstained = np.asarray(predicted), np.asarray(abstained)
    return {
        label: {
            "abstained": int(((predicted == label) & abstained).sum()),
            "answered": int(((predicted == label) & ~abstained).sum()),
        }
        for label in np.unique(predicted)
    }


def within_label_auc(scores, abstained, groups, n_perm: int = 2000, seed: int = 0):
    """AUC of scores for predicting abstention, with a permutation p-value.

    Abstention is shuffled within group (template), so a template that happens to have
    both high scores and many abstentions can't produce the effect. Returns None when
    one outcome is too rare to test, which is a result to report, not to hide.
    """
    scores, abstained, groups = map(np.asarray, (scores, abstained, groups))
    if min(abstained.sum(), (~abstained).sum()) < 5:
        return None
    observed = roc_auc_score(abstained, scores)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = abstained.copy()
        for group in np.unique(groups):
            idx = np.where(groups == group)[0]
            shuffled[idx] = rng.permutation(shuffled[idx])
        # Within-group shuffling can't change the totals, so both classes stay present.
        null[i] = roc_auc_score(shuffled, scores)
    p = (1 + (np.abs(null - 0.5) >= abs(observed - 0.5)).sum()) / (n_perm + 1)
    return {"auc": float(observed), "p": float(p), "n": int(len(scores)),
            "n_abstained": int(abstained.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("behavior", type=Path)
    parser.add_argument("oof", type=Path, help="probe_oof.npz from evidence_probe.probes")
    parser.add_argument("--layer", type=int, required=True,
                        help="Choose it from the accuracy curve, not from this script's output.")
    args = parser.parse_args()

    results = load_rows(args.behavior)
    saved = np.load(args.oof, allow_pickle=False)
    position = {row_id: i for i, row_id in enumerate(saved["ids"].tolist())}
    idx = [position[r["id"]] for r in results]
    proba = saved["proba"][args.layer][idx]
    classes = saved["classes"].tolist()
    predicted = np.array(classes)[proba.argmax(axis=1)]

    labels = np.array([r["label"] for r in results])
    templates = np.array([r["template"] for r in results])
    abstained = np.array([r["behavior"] == "abstain" for r in results])
    accuracy = (predicted == labels).mean()

    print("True label by behavior")
    print_summary(results)
    print(f"\nProbe at layer {args.layer} by behavior (out-of-fold accuracy {accuracy:.3f})")
    table = crosstab(predicted, abstained)
    for label, cells in table.items():
        print(f"  probe says {label:14} abstained {cells['abstained']:4d}  answered {cells['answered']:4d}")
    if accuracy > 0.9:
        print("  Note: at this accuracy the table above nearly restates true label by behavior.")

    print("\nWithin-label test: does the probe's score for this label predict abstention?")
    for label in classes:
        mask = labels == label
        result = within_label_auc(proba[mask, classes.index(label)], abstained[mask], templates[mask])
        print(f"  {label:14}", "too few of one outcome to test" if result is None else result)

    insufficient_score = proba[:, classes.index("insufficient")]
    mask = labels == "insufficient"
    result = within_label_auc(insufficient_score[mask], abstained[mask], templates[mask])
    if result is not None:
        print(f"\nHeadline, insufficient items only: AUC {result['auc']:.3f}, p = {result['p']:.4f}")


if __name__ == "__main__":
    main()

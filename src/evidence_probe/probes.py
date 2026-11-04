"""Train a probe per layer, score it on held-out templates and drugs, and plot the curve.

Choices to be able to defend:
  * Regularization is fixed in advance (C = 0.01), since the data have far more
    dimensions than training rows and tuning it on test folds would be double dipping.
  * The scaler is fit inside each fold, so nothing about the test rows leaks in.
  * The significance threshold is a max statistic. Each label shuffle records its BEST
    accuracy over the layers tested, so picking the peak layer is already paid for.

Usage:
    python -m evidence_probe.probes data/matched.jsonl activations/matched_neutral.npz --out-dir results
"""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evidence_probe.baselines import bag_of_words_accuracy, load_rows, local_window_accuracy
from evidence_probe.splits import shuffle_within_groups, template_drug_folds

C = 0.01

# Validated reference palette: categorical slots 1-3 and the chart chrome.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PROBE_COLOR, LOCAL_COLOR, BOW_COLOR = "#2a78d6", "#eb6834", "#1baf7a"


def _oof_proba(X, y, folds, classes) -> np.ndarray:
    """Out-of-fold class probabilities. Every row is scored by a probe that never saw it."""
    proba = np.zeros((len(y), len(classes)))
    for train, test in folds:
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        probe.fit(X[train], y[train])
        columns = [classes.index(c) for c in probe.classes_]
        proba[np.ix_(test, columns)] = probe.predict_proba(X[test])
    return proba


def layer_curve(X, rows):
    """Return (proba [layers, N, classes], accuracy [layers], per-template accuracy dict)."""
    y = np.array([r["label"] for r in rows])
    templates = np.array([r["template"] for r in rows])
    classes = sorted(set(y.tolist()))
    folds = list(template_drug_folds(templates, [r["drug_fold"] for r in rows]))
    proba = np.stack([_oof_proba(X[:, layer], y, folds, classes) for layer in range(X.shape[1])])
    predicted = np.array(classes)[proba.argmax(axis=2)]
    accuracy = (predicted == y).mean(axis=1)
    per_template = {t: (predicted[:, templates == t] == y[templates == t]).mean(axis=1)
                    for t in np.unique(templates)}
    return proba, accuracy, per_template, classes


def max_stat_null(X, rows, layers, n_perm: int, seed: int = 0, n_jobs: int = -1) -> np.ndarray:
    """For each label shuffle, the best accuracy across the given layers."""
    y = np.array([r["label"] for r in rows])
    contexts = [r["context_id"] for r in rows]
    folds = list(template_drug_folds([r["template"] for r in rows], [r["drug_fold"] for r in rows]))
    classes = sorted(set(y.tolist()))

    def one_shuffle(perm_seed):
        shuffled = shuffle_within_groups(y, contexts, np.random.default_rng(perm_seed))
        best = 0.0
        for layer in layers:
            proba = _oof_proba(X[:, layer], shuffled, folds, classes)
            best = max(best, (np.array(classes)[proba.argmax(axis=1)] == shuffled).mean())
        return best

    seeds = np.random.SeedSequence(seed).generate_state(n_perm)
    return np.array(Parallel(n_jobs=n_jobs)(delayed(one_shuffle)(int(s)) for s in seeds))


def plot_curve(accuracy, per_template, threshold, bow, local_window, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = np.arange(len(accuracy))
    template_matrix = np.stack(list(per_template.values()))
    sem = template_matrix.std(axis=0, ddof=1) / np.sqrt(len(template_matrix))

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(layers, accuracy - sem, accuracy + sem, color=PROBE_COLOR, alpha=0.15, lw=0)
    ax.plot(layers, accuracy, color=PROBE_COLOR, lw=1.5, label="Activation probe (±1 SEM over templates)")
    ax.axhline(local_window, color=LOCAL_COLOR, lw=1.5, label="Text: sentences naming the drug")
    ax.axhline(bow, color=BOW_COLOR, lw=1.5, label="Text: bag of words, whole prompt")
    if threshold is not None:
        ax.axhline(threshold, color=INK_2, lw=1, ls="--", label="Shuffled labels, 95th pct of best layer")
    ax.axhline(1 / 3, color=MUTED, lw=1, ls=":", label="Chance")

    peak = int(accuracy.argmax())
    ax.annotate(f"{accuracy[peak]:.2f}, layer {peak}", (peak, accuracy[peak]), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=8.5, color=INK)

    ax.set_xlim(0, layers[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Layer (0 = embeddings)", color=INK_2, fontsize=9)
    ax.set_ylabel("Accuracy on held-out templates and drugs", color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def align(X, ids, rows):
    position = {row_id: i for i, row_id in enumerate(ids.tolist())}
    return X[[position[r["id"]] for r in rows]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("activations", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-perm", type=int, default=100, help="0 skips the shuffled-label null.")
    parser.add_argument("--null-every", type=int, default=2,
                        help="Test every Nth layer in the null. Lower is stricter but slower.")
    args = parser.parse_args()

    rows = load_rows(args.data)
    saved = np.load(args.activations, allow_pickle=False)
    X = align(saved["X"], saved["ids"], rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if X[:, 0].std(axis=0).max() > 1e-5:
        print("WARNING: layer 0 varies across prompts, so the final token is not constant. "
              "Check the prompt template before trusting anything else.")

    proba, accuracy, per_template, classes = layer_curve(X, rows)
    bow, local_window = bag_of_words_accuracy(rows), local_window_accuracy(rows)
    print(f"Chance 0.333 | bag of words {bow:.3f} | local window {local_window:.3f}")
    for layer, value in enumerate(accuracy):
        print(f"  layer {layer:2d}  {value:.3f}")

    threshold, p_value, null = None, None, []
    if args.n_perm:
        null_layers = list(range(1, X.shape[1], args.null_every))
        null = max_stat_null(X, rows, null_layers, args.n_perm)
        threshold = float(np.quantile(null, 0.95))
        tested_peak = accuracy[null_layers].max()
        p_value = float((1 + (null >= tested_peak).sum()) / (args.n_perm + 1))
        print(f"Null: 95th pct of best-layer accuracy {threshold:.3f}. "
              f"Peak over tested layers {tested_peak:.3f}, p = {p_value:.4f}")

    np.savez(args.out_dir / "probe_oof.npz", proba=proba, classes=classes,
             ids=[r["id"] for r in rows])
    (args.out_dir / "probe_curve.json").write_text(json.dumps({
        "accuracy": accuracy.tolist(),
        "per_template": {t: v.tolist() for t, v in per_template.items()},
        "bag_of_words": bow,
        "local_window": local_window,
        "null_threshold_95": threshold,
        "p_value_peak": p_value,
        "null_max_accuracy": list(map(float, null)),
    }, indent=2))
    plot_curve(accuracy, per_template, threshold, bow, local_window, args.out_dir / "accuracy_by_layer.png")
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()

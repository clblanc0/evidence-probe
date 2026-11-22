"""Decode tampered vs untampered from activations, layer by layer, on held-out formats.

Two analyses:
  all types      trained and tested on all three tampering types (figure 1, blue)
  held-out type  trained without one tampering type, tested on it. This is the
                 shortcut test from the research plan: a probe that learned "a digit
                 changed" shouldn't catch age shifts unless it found something general.

Choices to be able to defend:
  * Regularization is fixed in advance (C = 0.01). There are far more dimensions than
    training rows, and tuning on test folds would be double dipping.
  * The scaler is fit inside each fold.
  * AUC is averaged over held-out formats.
  * The significance threshold is a max statistic: each label shuffle records its best
    AUC over the layers tested, so picking the peak layer is already paid for.

Usage:
    python -m evidence_probe.probes data/charts.jsonl activations/neutral.npz --position visit_end --out-dir results
"""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evidence_probe.baselines import bag_of_words_auc, load_rows, visit_only_auc
from evidence_probe.generate import TAMPERED
from evidence_probe.splits import format_folds, held_out_type_folds, shuffle_within_groups

C = 0.01

# Reference palette: categorical slots 1 and 2, plus chart chrome.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ALL_TYPES_COLOR, HELD_OUT_COLOR = "#2a78d6", "#eb6834"


def _fold_scores(X, y, folds):
    """Out-of-fold probability of 'tampered' for each test row, and mean AUC over folds."""
    scores = np.full(len(y), np.nan)
    aucs = []
    for train, test in folds:
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        probe.fit(X[train], y[train])
        scores[test] = probe.predict_proba(X[test])[:, 1]
        aucs.append(roc_auc_score(y[test], scores[test]))
    return scores, float(np.mean(aucs))


def layer_curve(X, rows):
    """All tampering types. Returns (scores [layers, N], auc [layers], per-format auc)."""
    y = np.array([r["label"] for r in rows])
    folds = list(format_folds(rows))
    results = [_fold_scores(X[:, layer], y, folds) for layer in range(X.shape[1])]
    scores = np.stack([s for s, _ in results])
    auc = np.array([a for _, a in results])
    templates = np.array([r["template"] for r in rows])
    per_format = {t: [float(roc_auc_score(y[templates == t], s[templates == t])) for s in scores]
                  for t in np.unique(templates)}
    return scores, auc, per_format


def held_out_type_curves(X, rows) -> dict[str, np.ndarray]:
    """AUC by layer for each tampering type when the probe never trained on it."""
    y = np.array([r["label"] for r in rows])
    curves = {}
    for held in TAMPERED:
        folds = list(held_out_type_folds(rows, held))
        curves[held] = np.array([_fold_scores(X[:, layer], y, folds)[1] for layer in range(X.shape[1])])
    return curves


def max_stat_null(X, rows, layers, n_perm: int, seed: int = 0, n_jobs: int = -1) -> np.ndarray:
    """For each label shuffle (within base chart), the best AUC across the given layers."""
    y = np.array([r["label"] for r in rows])
    bases = [r["base_id"] for r in rows]
    folds = list(format_folds(rows))

    def one_shuffle(perm_seed):
        shuffled = shuffle_within_groups(y, bases, np.random.default_rng(perm_seed))
        return max(_fold_scores(X[:, layer], shuffled, folds)[1] for layer in layers)

    seeds = np.random.SeedSequence(seed).generate_state(n_perm)
    return np.array(Parallel(n_jobs=n_jobs)(delayed(one_shuffle)(int(s)) for s in seeds))


def plot_curve(auc, per_format, held_out, threshold, bow, visit_only, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = np.arange(len(auc))
    format_matrix = np.array(list(per_format.values()))
    sem = format_matrix.std(axis=0, ddof=1) / np.sqrt(len(format_matrix))
    held_out_mean = np.mean(list(held_out.values()), axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(layers, auc - sem, auc + sem, color=ALL_TYPES_COLOR, alpha=0.15, lw=0)
    ax.plot(layers, auc, color=ALL_TYPES_COLOR, lw=1.5, label="Probe, all tampering types (±1 SEM over formats)")
    ax.plot(layers, held_out_mean, color=HELD_OUT_COLOR, lw=1.5,
            label="Probe, tested on a tampering type it never trained on (mean of 3)")

    # Labels go above or below their line so near-equal references don't collide.
    references = [(bow, "Text: bag of words", "-.", 3, "bottom"),
                  (visit_only, "Text: new visit only", ":", -3, "top")]
    if threshold is not None:
        references.append((threshold, "Shuffled labels, 95th pct", "--", -3, "top"))
    for value, text, style, dy, va in references:
        ax.axhline(value, color=INK_2, lw=1, ls=style)
        x_text = layers[-1] if dy > 0 else layers[len(layers) // 3]
        ax.annotate(f"{text} ({value:.2f})", (x_text, value), xytext=(-4, dy),
                    textcoords="offset points", ha="right", va=va, fontsize=7.5, color=INK_2)
    ax.axhline(0.5, color=MUTED, lw=0.8)

    peak = int(auc.argmax())
    ax.annotate(f"{auc[peak]:.2f}, layer {peak}", (peak, auc[peak]), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, layers[-1])
    ax.set_ylim(0.3, 1.05)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel("Layer (0 = embeddings)", color=INK_2, fontsize=9)
    ax.set_ylabel("AUC, held-out chart formats", color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def replot(json_path: Path) -> None:
    """Redraw figure 1 from a saved probe_<position>.json without rerunning the null."""
    saved = json.loads(Path(json_path).read_text())
    where = "after reading the new visit" if saved["position"] == "visit_end" else "before answering"
    plot_curve(np.array(saved["auc"]), {k: np.array(v) for k, v in saved["per_format"].items()},
               {k: np.array(v) for k, v in saved["held_out_type"].items()},
               saved["null_threshold_95"], saved["bag_of_words"], saved["visit_only"],
               Path(json_path).parent / f"fig1_auc_by_layer_{saved['position']}.png",
               f"Is tampering decodable from activations {where}?")


def align(X, ids, rows):
    position = {row_id: i for i, row_id in enumerate(ids.tolist())}
    return X[[position[r["id"]] for r in rows]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("activations", type=Path)
    parser.add_argument("--position", choices=["visit_end", "final"], default="visit_end")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-perm", type=int, default=100, help="0 skips the shuffled-label null.")
    parser.add_argument("--null-every", type=int, default=2, help="Test every Nth layer in the null.")
    args = parser.parse_args()

    rows = load_rows(args.data)
    saved = np.load(args.activations, allow_pickle=False)
    X = align(saved[args.position], saved["ids"], rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.position == "final" and X[:, 0].std(axis=0).max() > 1e-5:
        print("WARNING: layer 0 varies at the final token, so the prompt ending isn't constant.")

    scores, auc, per_format = layer_curve(X, rows)
    held_out = held_out_type_curves(X, rows)
    bow, visit_only = bag_of_words_auc(rows), visit_only_auc(rows)
    print(f"Chance 0.5 | bag of words {bow:.3f} | new visit only {visit_only:.3f}")
    print("layer  all types  " + "  ".join(f"held-out {t}" for t in TAMPERED))
    for layer in range(len(auc)):
        print(f"{layer:5d}  {auc[layer]:9.3f}  " + "  ".join(
            f"{held_out[t][layer]:{len('held-out ' + t)}.3f}" for t in TAMPERED))

    threshold, p_value, null = None, None, []
    if args.n_perm:
        null_layers = list(range(1, X.shape[1], args.null_every))
        null = max_stat_null(X, rows, null_layers, args.n_perm)
        threshold = float(np.quantile(null, 0.95))
        tested_peak = auc[null_layers].max()
        p_value = float((1 + (null >= tested_peak).sum()) / (args.n_perm + 1))
        print(f"Null: 95th pct of best-layer AUC {threshold:.3f}. "
              f"Peak over tested layers {tested_peak:.3f}, p = {p_value:.4f}")

    stem = f"probe_{args.position}"
    np.savez(args.out_dir / f"{stem}_oof.npz", scores=scores, ids=[r["id"] for r in rows])
    (args.out_dir / f"{stem}.json").write_text(json.dumps({
        "position": args.position,
        "auc": auc.tolist(),
        "per_format": per_format,
        "held_out_type": {t: v.tolist() for t, v in held_out.items()},
        "bag_of_words": bow,
        "visit_only": visit_only,
        "null_threshold_95": threshold,
        "p_value_peak": p_value,
        "null_max_auc": list(map(float, null)),
    }, indent=2))
    where = "after reading the new visit" if args.position == "visit_end" else "before answering"
    plot_curve(auc, per_format, held_out, threshold, bow, visit_only,
               args.out_dir / f"fig1_auc_by_layer_{args.position}.png",
               f"Is tampering decodable from activations {where}?")
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()

"""Inside vs behavior: what the probe caught against what the model said.

For each condition, three rates:
  probe   share the probe scored as tampered (out-of-fold probability > 0.5)
  model   share where the model's response flagged an identity problem
  rule    share the name-or-MRN string rule flagged
On untampered conditions these are false-alarm rates.

The headline is the gap between probe and model on each tampering type. A caution for
the write-up: the probe is trained on the input's label, so "probe caught it" means the
activations separate tampered from untampered inputs. It doesn't show that the model
believes something is wrong. The within-type test asks the sharper question: among rows
of the SAME tampering type, does the probe's score predict whether the model flags?

Usage:
    python -m evidence_probe.gap results/behavior_neutral.jsonl results/probe_visit_end_oof.npz --layer 14
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from evidence_probe.baselines import flag_rate_by_condition, load_rows, rule_flags
from evidence_probe.generate import CONDITIONS, TAMPERED

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = (("probe", "Probe on activations", "#2a78d6"),
          ("model", "Model's response flagged it", "#eb6834"),
          ("rule", "Name-or-MRN string rule", "#1baf7a"))


def within_type_auc(scores, flagged, groups, n_perm: int = 2000, seed: int = 0):
    """AUC of probe scores for predicting the model's flag, with a permutation p-value.

    Flags are shuffled within format, so a format with both high scores and many flags
    can't create the effect. Returns None when one outcome is too rare to test.
    """
    scores, flagged, groups = map(np.asarray, (scores, flagged, groups))
    # Below about 8 in the rarer outcome, the AUC is too noisy to read (the dry run on
    # random flags gave p = 0.04 with only 5 flagged rows).
    if min(flagged.sum(), (~flagged).sum()) < 8:
        return None
    observed = roc_auc_score(flagged, scores)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = flagged.copy()
        for group in np.unique(groups):
            idx = np.where(groups == group)[0]
            shuffled[idx] = rng.permutation(shuffled[idx])
        null[i] = roc_auc_score(shuffled, scores)
    p = (1 + (np.abs(null - 0.5) >= abs(observed - 0.5)).sum()) / (n_perm + 1)
    return {"auc": float(observed), "p": float(p), "n": int(len(scores)), "n_flagged": int(flagged.sum())}


def rates(results, probe_scores) -> dict[str, dict[str, float]]:
    return {
        "probe": flag_rate_by_condition(results, probe_scores > 0.5),
        "model": flag_rate_by_condition(results, [r["flagged"] for r in results]),
        "rule": flag_rate_by_condition(results, [rule_flags(r) for r in results]),
    }


def plot_gap(table, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"clean": "Clean", "benign_name": "Name\nreformatted", "benign_mrn": "MRN\ndashed",
              "swap": "Patient\nswap", "mrn_digit": "One-digit\nMRN change", "age_shift": "Age\nshift"}
    x = np.arange(len(CONDITIONS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (key, name, color) in enumerate(SERIES):
        values = [table[key].get(c, np.nan) for c in CONDITIONS]
        ax.bar(x + (i - 1) * width, values, width - 0.03, color=color, label=name)
    ax.axvline(2.5, color=AXIS, lw=0.8)
    ax.text(1, 1.04, "Not tampered: bars are false alarms", ha="center", fontsize=8, color=INK_2)
    ax.text(4, 1.04, "Tampered: bars are catches", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, [labels[c] for c in CONDITIONS], fontsize=8, color=INK_2)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Share of rows flagged", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=18)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("behavior", type=Path)
    parser.add_argument("oof", type=Path, help="probe_<position>_oof.npz from evidence_probe.probes")
    parser.add_argument("--layer", type=int, required=True,
                        help="Choose it from figure 1, before looking at this script's output.")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    results = load_rows(args.behavior)
    saved = np.load(args.oof, allow_pickle=False)
    position = {row_id: i for i, row_id in enumerate(saved["ids"].tolist())}
    scores = saved["scores"][args.layer][[position[r["id"]] for r in results]]

    table = rates(results, scores)
    print(f"Flag rates by condition (probe at layer {args.layer}, threshold 0.5)")
    print(f"{'':12}" + "".join(f"{key:>8}" for key, _, _ in SERIES))
    for condition in CONDITIONS:
        print(f"{condition:12}" + "".join(f"{table[key].get(condition, float('nan')):8.2f}"
                                          for key, _, _ in SERIES))

    print("\nWithin each tampering type: does the probe's score predict the model's flag?")
    conditions = np.array([r["condition"] for r in results])
    flagged = np.array([r["flagged"] for r in results])
    templates = np.array([r["template"] for r in results])
    within = {}
    for condition in TAMPERED:
        mask = conditions == condition
        within[condition] = within_type_auc(scores[mask], flagged[mask], templates[mask])
        print(f"  {condition:10}", within[condition] or "too few of one outcome to test")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    variant = results[0].get("variant", "neutral")
    (args.out_dir / f"gap_{variant}.json").write_text(json.dumps(
        {"layer": args.layer, "rates": table, "within_type": within}, indent=2))
    plot_gap(table, args.out_dir / f"fig2_inside_vs_behavior_{variant}.png",
             f"What the activations caught vs what the model said ({variant} prompt)")
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()

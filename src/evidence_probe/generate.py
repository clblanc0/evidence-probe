"""Generate labelled medication scenarios.

Two designs:

matched (the real dataset)
    One context, three drugs, three questions. Every drug gets exactly two sentences:
        sufficient     one directive + one neutral mention
        contradictory  a stop directive + a continue directive
        insufficient   two neutral mentions
    Asking about each drug in turn gives all three labels from identical context text.
    The rows of a triplet differ only by the drug named in the question, so length,
    vocabulary and mention count cannot predict the label. It is a property of how the
    question relates to the context.

naive (the positive control)
    One drug per context, so the label shows in the words: contradictory contexts have
    more directive vocabulary and insufficient contexts have none. This is the obvious
    way to build the dataset. The text baseline should score well above chance here. If
    it doesn't, chance performance on the matched design proves nothing.

Usage:
    python -m evidence_probe.generate --design matched --per-template 25 --out data/matched.jsonl
"""

import argparse
import itertools
import json
import random
from pathlib import Path

from evidence_probe.templates import DRUGS, FAMILIES, TemplateFamily

LABELS = ("sufficient", "contradictory", "insufficient")


def drug_folds(seed: int) -> dict[str, int]:
    """Split drugs into two disjoint sets, for holding out drugs as well as templates."""
    shuffled = list(DRUGS)
    random.Random(seed).shuffle(shuffled)
    return {drug: i % 2 for i, drug in enumerate(shuffled)}


def _sentences_for(role: str, direction: str | None, family: TemplateFamily, rng: random.Random):
    """Return the two (kind, frame) pairs a drug gets, given its role."""
    neutral = lambda: ("neutral", rng.choice(family.neutral))  # noqa: E731
    stop = lambda: ("stop", rng.choice(family.stop))  # noqa: E731
    cont = lambda: ("continue", rng.choice(family.cont))  # noqa: E731
    if role == "sufficient":
        return [stop() if direction == "stop" else cont(), neutral()]
    if role == "contradictory":
        return [stop(), cont()]
    if role == "insufficient":
        return [neutral(), neutral()]
    raise ValueError(role)


def _build_context(assignments, family: TemplateFamily, rng: random.Random):
    """Lay out sentences for {drug: (role, direction)} in random order, one source each.

    Returns the context text and each drug's sentence positions and kinds.
    """
    items = []
    for drug, (role, direction) in assignments.items():
        items += [(drug, *pair) for pair in _sentences_for(role, direction, family, rng)]
    rng.shuffle(items)

    # A distinct source per sentence, so that "two sources disagree" is literally true.
    sources = rng.sample(family.sources, len(items))
    lines, positions, kinds = [family.header], {}, {}
    for i, ((drug, kind, frame), source) in enumerate(zip(items, sources)):
        lines.append(frame.format(drug=drug, source=source))
        positions.setdefault(drug, []).append(i)
        kinds.setdefault(drug, []).append(kind)
    return "\n".join(lines), positions, kinds


def _row(context_id, design, family, context, question, drug, role, direction, positions, kinds,
         folds, n_sentences):
    gold = {"stop": "no", "continue": "yes", None: None}[direction]
    return {
        "id": f"{context_id}__{drug}",
        "context_id": context_id,
        "design": design,
        "template": family.name,
        "context": context,
        "question": question,
        "target_drug": drug,
        "drug_fold": folds[drug],
        "label": role,
        "gold_answer": gold,
        # Kept for the position control and for error analysis. Never given to a probe.
        "target_positions": positions[drug],
        "target_kinds": kinds[drug],
        "n_sentences": n_sentences,
    }


def generate_matched(per_template: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    folds = drug_folds(seed)
    rows = []
    for family in FAMILIES:
        # Cycling through all 6 role orders balances role against drug slot within a
        # template. The sufficient direction alternates so yes and no are equally common.
        role_orders = itertools.cycle(itertools.permutations(LABELS))
        directions = itertools.cycle(("stop", "continue"))
        for n in range(per_template):
            drugs = rng.sample(DRUGS, 3)
            direction = next(directions)
            assignments = {
                drug: (role, direction if role == "sufficient" else None)
                for drug, role in zip(drugs, next(role_orders))
            }
            context, positions, kinds = _build_context(assignments, family, rng)
            # The same question frame across a triplet, so only the drug name changes.
            frame = rng.choice(family.questions)
            context_id = f"{family.name}_{n:03d}"
            for drug, (role, direction_) in assignments.items():
                rows.append(_row(context_id, "matched", family, context, frame.format(drug=drug),
                                 drug, role, direction_, positions, kinds, folds, 6))
    return rows


def generate_naive(per_template: int, seed: int) -> list[dict]:
    """One drug per context. Deliberately leaky: see the module docstring."""
    rng = random.Random(seed)
    folds = drug_folds(seed)
    rows = []
    for family in FAMILIES:
        directions = itertools.cycle(("stop", "continue"))
        for n in range(per_template):
            for role in LABELS:
                drug = rng.choice(DRUGS)
                direction = next(directions) if role == "sufficient" else None
                context, positions, kinds = _build_context({drug: (role, direction)}, family, rng)
                frame = rng.choice(family.questions)
                context_id = f"{family.name}_{n:03d}_{role}"
                rows.append(_row(context_id, "naive", family, context, frame.format(drug=drug),
                                 drug, role, direction, positions, kinds, folds, 2))
    return rows


def generate(design: str, per_template: int = 25, seed: int = 0) -> list[dict]:
    builders = {"matched": generate_matched, "naive": generate_naive}
    return builders[design](per_template, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--design", choices=["matched", "naive"], default="matched")
    parser.add_argument("--per-template", type=int, default=25,
                        help="Contexts per template. Matched gives 3 rows per context.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = generate(args.design, args.per_template, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

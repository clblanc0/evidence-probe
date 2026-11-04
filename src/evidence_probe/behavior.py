"""Have the model answer each question, then score what it did.

Run this first, on a small pilot. If the model never abstains or can't answer the
sufficient items correctly, the gap analysis has nothing to work with, and you want to
learn that before spending time on anything else.

Usage:
    python -m evidence_probe.behavior data/matched.jsonl --limit-contexts 40 --out results/pilot.jsonl
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from evidence_probe.baselines import load_rows
from evidence_probe.prompting import render

ABSTAIN = re.compile(
    r"not (?:\w+ )?(?:clear|enough|sufficient|specified|stated|mentioned|provided|possible|sure)"
    r"|unclear|insufficient|conflict\w*|contradict\w*|disagree\w*|inconsistent|mixed|differ\w*"
    r"|(?:cannot|can't|can not|unable to|impossible to|don't|do not) "
    r"(?:\w+ )?(?:determine|say|tell|answer|know|confirm)"
    r"|no (?:\w+ )?(?:information|instruction|recommendation|guidance|directive|mention)"
    r"|depends|consult|check with|(?:speak|talk) (?:to|with)"
)
YES = re.compile(
    r"^(?:yes\b|(?:the patient |you |they |i )?(?:should )?"
    r"(?:keep|continue|stay|remain|carry on|still)\b)"
)
NO = re.compile(
    r"^(?:no\b|(?:the patient |you |they |i )?should not\b|(?:the patient |you |they |i )?"
    r"(?:should )?(?:stop|discontinue|quit)\b)"
)


def score_response(text: str) -> str:
    """Return abstain, yes, no or other, judging by the first sentence.

    The first sentence is what a reader acts on, so "Yes, keep taking it. Note the
    records conflict." counts as an answer. This is a scoring choice worth stating in
    the write-up. Validate the scorer by hand with --export-blind, and treat a high
    "other" rate as a scorer problem to fix, not as a behavior.
    """
    # Remove the period in "Dr." so it doesn't end the sentence early.
    stripped = text.strip().lower().replace("dr. ", "dr ")
    first_sentence = re.split(r"(?<=[.!?])\s|\n", stripped, maxsplit=1)[0]
    if ABSTAIN.search(first_sentence):
        return "abstain"
    if NO.search(first_sentence):
        return "no"
    if YES.search(first_sentence):
        return "yes"
    return "other"


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy, so behavior is reproducible and maps to one activation.
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def select_contexts(rows: list[dict], limit: int | None, seed: int = 0) -> list[dict]:
    """Keep whole triplets, spread evenly over templates, so a pilot stays balanced."""
    if limit is None:
        return rows
    by_template: dict[str, list[str]] = {}
    for context_id in dict.fromkeys(r["context_id"] for r in rows):
        template = next(r["template"] for r in rows if r["context_id"] == context_id)
        by_template.setdefault(template, []).append(context_id)
    rng = random.Random(seed)
    per_template = max(1, limit // len(by_template))
    keep = {c for ids in by_template.values() for c in rng.sample(ids, min(per_template, len(ids)))}
    return [r for r in rows if r["context_id"] in keep]


def behavior_table(results: list[dict]) -> dict[str, Counter]:
    table: dict[str, Counter] = {}
    for result in results:
        table.setdefault(result["label"], Counter())[result["behavior"]] += 1
    return table


def sufficient_accuracy(results: list[dict]) -> float | None:
    """Of sufficient items, the share answered correctly. Chance for a guesser is 0.5."""
    scored = [r for r in results if r["label"] == "sufficient" and r["behavior"] in ("yes", "no")]
    if not scored:
        return None
    return sum(r["behavior"] == r["gold_answer"] for r in scored) / len(scored)


def export_blind(results: list[dict], path: Path, n: int = 100, seed: int = 0) -> None:
    """Write a shuffled sample without labels or scores, for hand-coding the scorer."""
    sample = random.Random(seed).sample(results, min(n, len(results)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "response", "your_code"])
        writer.writerows([r["id"], r["response"], ""] for r in sample)


def print_summary(results: list[dict]) -> None:
    behaviors = ("abstain", "yes", "no", "other")
    print(f"{'':16}" + "".join(f"{b:>9}" for b in behaviors))
    for label, counts in sorted(behavior_table(results).items()):
        print(f"{label:16}" + "".join(f"{counts[b]:9d}" for b in behaviors))
    accuracy = sufficient_accuracy(results)
    print("Correct on answered sufficient items:",
          "n/a" if accuracy is None else f"{accuracy:.3f} (guessing = 0.5)")


def main() -> None:
    from evidence_probe.extract import DEFAULT_MODEL, load_model

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variant", default="neutral")
    parser.add_argument("--limit-contexts", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--export-blind", type=Path, default=None)
    args = parser.parse_args()

    rows = select_contexts(load_rows(args.data), args.limit_contexts)
    model, tokenizer = load_model(args.model)
    results = []
    for row in rows:
        response = generate_response(model, tokenizer, render(tokenizer, row, args.variant))
        results.append({**row, "variant": args.variant, "response": response,
                        "behavior": score_response(response)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in results))
    print_summary(results)
    if args.export_blind:
        export_blind(results, args.export_blind)


if __name__ == "__main__":
    main()

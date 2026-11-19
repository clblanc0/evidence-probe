"""Record every layer's activations at two points in each prompt.

visit_end  The last token of the new visit, the moment the model has just read the
           identity line. The model is causal, so nothing after this point (the task
           instruction included) can influence it.
final      The last prompt token, just before the model starts answering. It's the same
           chat-template token in every prompt, so layer 0 must be identical across
           prompts. That's a free check on the pipeline.

Prompts run one at a time, so there is no padding to get wrong.

Usage:
    python -m evidence_probe.extract data/charts.jsonl --out activations/neutral.npz
"""

import argparse
from pathlib import Path

import numpy as np

from evidence_probe.baselines import load_rows
from evidence_probe.prompting import render

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Ungated, so no license step on Colab.


def load_model(name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    return model, tokenizer


def visit_end_token(prompt: str, new_visit: str, offsets) -> int:
    """Index of the token holding the new visit's last character."""
    last_char = prompt.rindex(new_visit) + len(new_visit) - 1
    for index, (start, end) in enumerate(offsets):
        if start <= last_char < end:
            return index
    raise ValueError("No token covers the end of the new visit.")


def record(model, tokenizer, prompts: list[str], new_visits: list[str]):
    """Return {"visit_end": [N, layers + 1, hidden], "final": same} and token counts.

    Index 0 is the embedding output. In Hugging Face the last entry has the final layer
    norm applied, unlike the others, so don't over-read a jump at the last layer.
    """
    import torch

    out = {"visit_end": [], "final": []}
    n_tokens = []
    with torch.no_grad():
        for prompt, new_visit in zip(prompts, new_visits):
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                                return_offsets_mapping=True)
            offsets = encoded.pop("offset_mapping")[0].tolist()
            position = visit_end_token(prompt, new_visit, offsets)
            hidden = model(**encoded.to(model.device), output_hidden_states=True).hidden_states
            out["visit_end"].append(torch.stack([h[0, position] for h in hidden]).float().cpu().numpy())
            out["final"].append(torch.stack([h[0, -1] for h in hidden]).float().cpu().numpy())
            n_tokens.append(encoded["input_ids"].shape[1])
    return {k: np.stack(v) for k, v in out.items()}, np.array(n_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variant", default="neutral")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.data)
    model, tokenizer = load_model(args.model)
    prompts = [render(tokenizer, r, args.variant) for r in rows]
    activations, n_tokens = record(model, tokenizer, prompts, [r["new_visit"] for r in rows])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **activations, n_tokens=n_tokens, ids=[r["id"] for r in rows],
             model=args.model, variant=args.variant)
    print(f"Saved {activations['visit_end'].shape} per position to {args.out}")


if __name__ == "__main__":
    main()

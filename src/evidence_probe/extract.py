"""Save every layer's activation at the final prompt token.

The final token is the end of the chat template's assistant header, so it is the same
token in every prompt. That gives a built-in sanity check: layer 0 (the embedding) must
be identical across prompts, and a probe on it must sit at chance.

Prompts run one at a time. It's fast enough for 600 short prompts, and with no padding
there's no padding bug to reason about.

Usage:
    python -m evidence_probe.extract data/matched.jsonl --out activations/matched_neutral.npz
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


def final_token_activations(model, tokenizer, prompts: list[str]):
    """Return (activations [N, layers + 1, hidden], token count per prompt).

    Index 0 is the embedding output. In Hugging Face the last entry has the final
    layer norm applied, unlike the others, so don't over-read a jump at the last layer.
    """
    import torch

    activations, n_tokens = [], []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            hidden = model(**inputs, output_hidden_states=True).hidden_states
            activations.append(torch.stack([h[0, -1] for h in hidden]).float().cpu().numpy())
            n_tokens.append(inputs["input_ids"].shape[1])
    return np.stack(activations), np.array(n_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variant", default="neutral")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.data)
    model, tokenizer = load_model(args.model)
    prompts = [render(tokenizer, row, args.variant) for row in rows]
    X, n_tokens = final_token_activations(model, tokenizer, prompts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X, n_tokens=n_tokens, ids=[r["id"] for r in rows],
             model=args.model, variant=args.variant)
    print(f"Saved {X.shape} to {args.out}")

    labels = np.array([r["label"] for r in rows])
    for label in np.unique(labels):
        print(f"  mean tokens, {label}: {n_tokens[labels == label].mean():.1f}")


if __name__ == "__main__":
    main()

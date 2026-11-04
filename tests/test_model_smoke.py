"""Run the model-facing code on a tiny random Qwen2, built locally with no download.

This checks the plumbing only: shapes, the constant final token, generation. It says
nothing about real model behavior. Skipped when torch isn't installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from evidence_probe.behavior import generate_response  # noqa: E402
from evidence_probe.extract import final_token_activations  # noqa: E402
from evidence_probe.generate import generate  # noqa: E402
from evidence_probe.prompting import VARIANTS, render  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
LAYERS, HIDDEN = 2, 16


@pytest.fixture(scope="module")
def tiny():
    words = ["<unk>", "<|im_start|>", "<|im_end|>", "user", "assistant", "the", "patient"]
    vocab = {word: i for i, word in enumerate(words)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<|im_end|>",
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    config = transformers.Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        num_attention_heads=2, num_key_value_heads=2, intermediate_size=32,
    )
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(config).eval(), tokenizer


@pytest.fixture(scope="module")
def rows():
    return generate("matched", per_template=1, seed=0)[:6]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_prompt_contains_the_question_and_ends_with_the_assistant_header(tiny, rows, variant):
    _, tokenizer = tiny
    prompt = render(tokenizer, rows[0], variant)
    assert rows[0]["question"] in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_activation_shape_and_constant_layer_zero(tiny, rows):
    model, tokenizer = tiny
    X, n_tokens = final_token_activations(model, tokenizer, [render(tokenizer, r) for r in rows])
    assert X.shape == (len(rows), LAYERS + 1, HIDDEN)
    assert (n_tokens > 10).all()
    # The sanity check that probes.py relies on.
    assert X[:, 0].std(axis=0).max() < 1e-5
    assert X[:, 1].std(axis=0).max() > 0


def test_generation_returns_only_new_text(tiny, rows):
    model, tokenizer = tiny
    response = generate_response(model, tokenizer, render(tokenizer, rows[0]), max_new_tokens=5)
    assert isinstance(response, str)
    assert rows[0]["context"] not in response
    assert np.isscalar(len(response))

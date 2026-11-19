"""Run the model-facing code on a tiny random Qwen2, built locally with no download.

This checks plumbing only: shapes, token positions, generation. It says nothing about
real model behavior. Skipped when torch isn't installed.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from evidence_probe.behavior import generate_response  # noqa: E402
from evidence_probe.extract import record, visit_end_token  # noqa: E402
from evidence_probe.generate import generate  # noqa: E402
from evidence_probe.prompting import TASK, render  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
LAYERS, HIDDEN = 2, 16


@pytest.fixture(scope="module")
def tiny():
    words = ["<unk>", "<|im_start|>", "<|im_end|>", "user", "assistant", "Patient", "MRN"]
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(words)},
                                                               unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<|im_end|>")
    tokenizer.chat_template = CHAT_TEMPLATE
    config = transformers.Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        num_attention_heads=2, num_key_value_heads=2, intermediate_size=32)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(config).eval(), tokenizer


@pytest.fixture(scope="module")
def rows():
    return generate(per_format=1, seed=0)[:6]


@pytest.mark.parametrize("variant", sorted(TASK))
def test_prompt_holds_the_chart_and_ends_with_the_assistant_header(tiny, rows, variant):
    _, tokenizer = tiny
    prompt = render(tokenizer, rows[0], variant)
    assert rows[0]["chart"] in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_visit_end_token_is_the_last_token_of_the_new_visit(tiny, rows):
    _, tokenizer = tiny
    prompt = render(tokenizer, rows[0])
    offsets = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    index = visit_end_token(prompt, rows[0]["new_visit"], offsets)
    start, end = offsets[index]
    visit_end = prompt.rindex(rows[0]["new_visit"]) + len(rows[0]["new_visit"])
    assert end == visit_end
    assert offsets[index + 1][0] >= visit_end


def test_record_shapes_and_constant_final_layer_zero(tiny, rows):
    model, tokenizer = tiny
    activations, n_tokens = record(model, tokenizer, [render(tokenizer, r) for r in rows],
                                   [r["new_visit"] for r in rows])
    for key in ("visit_end", "final"):
        assert activations[key].shape == (len(rows), LAYERS + 1, HIDDEN)
    assert activations["final"][:, 0].std(axis=0).max() < 1e-5
    assert (n_tokens > 20).all()


def test_generation_returns_only_new_text(tiny, rows):
    model, tokenizer = tiny
    response = generate_response(model, tokenizer, render(tokenizer, rows[0]), max_new_tokens=5)
    assert isinstance(response, str) and rows[0]["chart"] not in response

"""Turn a scenario row into the exact prompt the model sees.

Activations and behavior must come from the same prompt. If they didn't, the gap
analysis would compare two different states. Both extract.py and behavior.py call
render(), and the variant name is recorded in each output file.
"""

VARIANTS = {
    # The default. Nothing suggests that abstaining is an option.
    "neutral": "{context}\n\nQuestion: {question}\nAnswer briefly.",
    # Explicitly allows abstention. Use it if the model never abstains under "neutral".
    "permission": (
        "{context}\n\nQuestion: {question}\n"
        "Answer briefly. If the records do not settle the question, say so."
    ),
}


def render(tokenizer, row: dict, variant: str = "neutral") -> str:
    user_message = VARIANTS[variant].format(context=row["context"], question=row["question"])
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )

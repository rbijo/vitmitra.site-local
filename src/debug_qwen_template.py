from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

messages = [
    {
        "role": "user",
        "content": "What is the duration of B.Tech - Biotechnology at VIT Vellore?",
    },
    {
        "role": "assistant",
        "content": "4 years",
    },
]

print("=" * 80)
print("QWEN3 CHAT TEMPLATE DEBUG")
print("=" * 80)

print("\nTokenizer:")
print(type(tokenizer).__name__)

print("\nTemplate supports enable_thinking?")
print("Searching template source...")

template = tokenizer.chat_template

if template is None:
    raise RuntimeError("Tokenizer has no chat template.")

print("\n" + "-" * 80)
print(template)
print("-" * 80)

print("\n" + "=" * 80)
print("RENDER WITH enable_thinking=False")
print("=" * 80)

rendered_false = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False,
)

print(rendered_false)

print("\n" + "=" * 80)
print("RENDER WITH enable_thinking=True")
print("=" * 80)

rendered_true = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=True,
)

print(rendered_true)

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print("False contains <think> :", "<think>" in rendered_false)
print("True contains <think>  :", "<think>" in rendered_true)
print(
    "Outputs identical       :",
    rendered_false == rendered_true,
)
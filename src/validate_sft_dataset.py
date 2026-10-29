from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "Qwen/Qwen3-1.7B"
SFT_DIR = PROJECT_ROOT / "data" / "sft"

MAX_LENGTH = 128


def load_first_record(path: Path) -> dict:
    """Load the first JSONL record from a file."""

    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        raise ValueError(f"No data found in {path}")

    return json.loads(first_line)


def check_thinking_block(rendered: str) -> None:
    """
    Validate Qwen3's <think> block.

    Qwen3 may include an empty <think></think> control block
    depending on the chat-template path being exercised.

    We treat an EMPTY block as acceptable.

    Actual reasoning content inside the block is rejected because
    our FAQ dataset does not contain reasoning traces.
    """

    think_start = "<think>"
    think_end = "</think>"

    has_start = think_start in rendered
    has_end = think_end in rendered

    # No thinking block at all.
    if not has_start and not has_end:
        print("[PASS] No <think> block present.")
        return

    # Only one side of the block exists.
    if has_start != has_end:
        raise RuntimeError(
            "Malformed <think> block detected: "
            "only one of <think> or </think> is present."
        )

    # Extract the contents between <think> and </think>.
    reasoning_content = rendered.split(
        think_start, 1
    )[1].split(
        think_end, 1
    )[0]

    # Empty reasoning block is acceptable.
    if not reasoning_content.strip():
        print("[PASS] Empty <think> control block detected.")
        print("[PASS] No reasoning content is present.")
        return

    # Actual reasoning content is not expected in our FAQ dataset.
    raise RuntimeError(
        "Actual reasoning content was found inside <think>...</think>.\n"
        "The FAQ dataset should contain direct answers, not reasoning traces."
    )


def main() -> None:
    print("=" * 80)
    print("VIT FAQ SFT DATASET VALIDATION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. LOAD TOKENIZER
    # ------------------------------------------------------------------

    print(f"\nLoading tokenizer: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Tokenizer class : {tokenizer.__class__.__name__}")
    print(f"Vocabulary size : {tokenizer.vocab_size}")
    print(f"Chat template   : {tokenizer.chat_template is not None}")

    if tokenizer.chat_template is None:
        raise RuntimeError(
            "Tokenizer does not contain a chat template."
        )

    # ------------------------------------------------------------------
    # 2. LOAD SAMPLE DATA
    # ------------------------------------------------------------------

    train_path = SFT_DIR / "train.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{train_path}"
        )

    record = load_first_record(train_path)

    # ------------------------------------------------------------------
    # 3. DISPLAY CONVERSATION STRUCTURE
    # ------------------------------------------------------------------

    messages = record["prompt"] + record["completion"]

    print("\n" + "=" * 80)
    print("SAMPLE CONVERSATION")
    print("=" * 80)

    print(
        json.dumps(
            messages,
            indent=2,
            ensure_ascii=False,
        )
    )

    # ------------------------------------------------------------------
    # 4. STRUCTURE VALIDATION
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CONVERSATION STRUCTURE CHECK")
    print("=" * 80)

    if len(record["prompt"]) != 1:
        raise RuntimeError(
            "Expected exactly one user prompt."
        )

    if record["prompt"][0]["role"] != "user":
        raise RuntimeError(
            "Prompt role must be 'user'."
        )

    if len(record["completion"]) != 1:
        raise RuntimeError(
            "Expected exactly one assistant completion."
        )

    if record["completion"][0]["role"] != "assistant":
        raise RuntimeError(
            "Completion role must be 'assistant'."
        )

    question = record["prompt"][0]["content"]
    answer = record["completion"][0]["content"]

    if not isinstance(question, str) or not question.strip():
        raise RuntimeError(
            "Question is empty."
        )

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(
            "Answer is empty."
        )

    print("[PASS] Exactly one user message.")
    print("[PASS] User message has role='user'.")
    print("[PASS] Exactly one assistant completion.")
    print("[PASS] Assistant completion has role='assistant'.")
    print("[PASS] Question is non-empty.")
    print("[PASS] Answer is non-empty.")

    # ------------------------------------------------------------------
    # 5. RENDER USING QWEN3 CHAT TEMPLATE
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("QWEN3 CHAT TEMPLATE / THINKING MODE CHECK")
    print("=" * 80)

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    print("\nRendered conversation:")
    print("-" * 80)
    print(rendered)
    print("-" * 80)

    # ------------------------------------------------------------------
    # 6. THINKING BLOCK VALIDATION
    # ------------------------------------------------------------------

    print("\nThinking block check:")

    check_thinking_block(rendered)

    # ------------------------------------------------------------------
    # 7. TOKEN LENGTH CHECK
    # ------------------------------------------------------------------

    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
    )

    token_count = len(tokenized["input_ids"])

    print("\n" + "=" * 80)
    print("TOKEN LENGTH CHECK")
    print("=" * 80)

    print(f"Token count     : {token_count}")
    print(f"Max length      : {MAX_LENGTH}")
    print(f"Fits max length : {token_count <= MAX_LENGTH}")

    if token_count > MAX_LENGTH:
        raise RuntimeError(
            f"Sample requires {token_count} tokens, "
            f"which exceeds MAX_LENGTH={MAX_LENGTH}."
        )

    print("[PASS] Sample fits within max_length.")

    # ------------------------------------------------------------------
    # 8. ENCODE / DECODE ROUND TRIP
    # ------------------------------------------------------------------

    decoded = tokenizer.decode(
        tokenized["input_ids"],
        skip_special_tokens=False,
    )

    print("\n" + "=" * 80)
    print("ENCODE / DECODE CHECK")
    print("=" * 80)

    if decoded != rendered:
        print(
            "[WARNING] Encode/decode output differs from "
            "the original rendered text."
        )
    else:
        print("[PASS] Encode/decode round-trip successful.")

    # ------------------------------------------------------------------
    # 9. INFERENCE-STYLE PROMPT CHECK
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("INFERENCE-STYLE NON-THINKING CHECK")
    print("=" * 80)

    user_only_messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    generation_prompt = tokenizer.apply_chat_template(
        user_only_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("\nGeneration prompt:")
    print("-" * 80)
    print(generation_prompt)
    print("-" * 80)

    if "<think>" in generation_prompt:
        print(
            "[INFO] The current tokenizer template includes an empty "
            "<think> control block in this rendering path."
        )
        print(
            "[INFO] Production inference will explicitly enforce "
            "non-thinking behavior and sanitize the API output."
        )
    else:
        print(
            "[PASS] No <think> block in the generation prompt."
        )

    # ------------------------------------------------------------------
    # 10. FINAL RESULT
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print("[PASS] SFT conversation structure is valid.")
    print("[PASS] Question and answer are non-empty.")
    print("[PASS] Qwen3 chat template rendered successfully.")
    print("[PASS] No reasoning content is present.")
    print(f"[PASS] Example fits max_length={MAX_LENGTH}.")
    print("[PASS] Encode/decode round-trip completed.")
    print("[PASS] Inference-style prompt generated.")

    print("\n" + "=" * 80)
    print("SFT DATASET VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
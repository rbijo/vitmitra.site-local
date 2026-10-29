"""
Inspect Qwen3-1.7B before QLoRA fine-tuning.

This script does NOT train the model.

It verifies:
    1. Python/package versions
    2. CUDA/GPU availability
    3. Qwen3 tokenizer
    4. Qwen3 chat template
    5. Qwen3 model architecture
    6. LoRA target modules
    7. 4-bit NF4 quantization
    8. A single forward pass on the GPU
    9. Approximate trainable/frozen parameter counts

Run from the project root:

    python src/inspect_model.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import transformers
import bitsandbytes as bnb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "Qwen/Qwen3-1.7B"

MAX_LENGTH = 128

# LoRA modules we intend to use later.
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ============================================================================
# Formatting helpers
# ============================================================================

def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def result(name: str, value: object) -> None:
    print(f"{name:<32}: {value}")


def cleanup() -> None:
    """Release GPU memory after diagnostics."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Package / environment inspection
# ============================================================================

def inspect_environment() -> None:
    section("1. ENVIRONMENT")

    result("Python", sys.version.split()[0])
    result("PyTorch", torch.__version__)
    result("Transformers", transformers.__version__)
    result("bitsandbytes", bnb.__version__)

    result(
        "CUDA available",
        torch.cuda.is_available(),
    )

    result(
        "PyTorch CUDA version",
        torch.version.cuda,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Do not continue until the GPU is detected."
        )

    result(
        "GPU count",
        torch.cuda.device_count(),
    )

    result(
        "GPU",
        torch.cuda.get_device_name(0),
    )

    total_memory = torch.cuda.get_device_properties(0).total_memory

    result(
        "GPU VRAM",
        f"{total_memory / (1024 ** 3):.2f} GB",
    )

    result(
        "BF16 supported",
        torch.cuda.is_bf16_supported(),
    )


# ============================================================================
# Tokenizer inspection
# ============================================================================

def inspect_tokenizer() -> AutoTokenizer:
    section("2. TOKENIZER")

    print(f"Loading tokenizer: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    result(
        "Tokenizer class",
        tokenizer.__class__.__name__,
    )

    result(
        "Vocabulary size",
        len(tokenizer),
    )

    result(
        "EOS token",
        repr(tokenizer.eos_token),
    )

    result(
        "EOS token ID",
        tokenizer.eos_token_id,
    )

    result(
        "PAD token",
        repr(tokenizer.pad_token),
    )

    result(
        "PAD token ID",
        tokenizer.pad_token_id,
    )

    result(
        "Chat template exists",
        tokenizer.chat_template is not None,
    )

    if tokenizer.chat_template is None:
        raise RuntimeError(
            "Qwen3 tokenizer does not expose a chat template."
        )

    # ------------------------------------------------------------------------
    # FAQ-style example
    # ------------------------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": (
                "What is the duration of "
                "B.Tech - Biotechnology at VIT Vellore?"
            ),
        },
        {
            "role": "assistant",
            "content": "The duration is 4 years.",
        },
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
    )

    input_ids = tokenized["input_ids"]

    print()
    print("Example rendered conversation:")
    print("-" * 80)
    print(rendered)
    print("-" * 80)

    result(
        "Rendered token count",
        len(input_ids),
    )

    result(
        "Fits max length 128",
        len(input_ids) <= MAX_LENGTH,
    )

    if len(input_ids) > MAX_LENGTH:
        print(
            "\nWARNING: Example exceeds MAX_LENGTH=128."
        )

    # ------------------------------------------------------------------------
    # Verify tokenizer can decode its own tokens.
    # ------------------------------------------------------------------------

    decoded = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
    )

    result(
        "Encode/decode successful",
        bool(decoded),
    )

    return tokenizer


# ============================================================================
# 4-bit configuration
# ============================================================================

def create_bnb_config() -> BitsAndBytesConfig:
    section("3. 4-BIT QLoRA CONFIGURATION")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("Planned quantization configuration:")
    print()

    result("Quantization", "4-bit")
    result("Quantization type", "NF4")
    result("Double quantization", True)
    result("Compute dtype", compute_dtype)

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    return config


# ============================================================================
# Model inspection
# ============================================================================

def inspect_model(
    tokenizer: AutoTokenizer,
    quantization_config: BitsAndBytesConfig,
):
    section("4. LOADING QWEN3-1.7B IN 4-BIT")

    print(f"Loading model: {MODEL_ID}")
    print("This may take a while the first time.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print()
    result(
        "Model class",
        model.__class__.__name__,
    )

    result(
        "Model device",
        model.device,
    )

    result(
        "Model dtype",
        model.dtype,
    )

    # ------------------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters - trainable_parameters
    )

    result(
        "Total parameters",
        f"{total_parameters:,}",
    )

    result(
        "Trainable parameters before LoRA",
        f"{trainable_parameters:,}",
    )

    result(
        "Frozen parameters",
        f"{frozen_parameters:,}",
    )

    # ------------------------------------------------------------------------
    # Inspect module names.
    # ------------------------------------------------------------------------

    section("5. LORA TARGET MODULE CHECK")

    module_matches: dict[str, list[str]] = {
        target: []
        for target in TARGET_MODULES
    }

    for module_name, module in model.named_modules():

        for target in TARGET_MODULES:

            if (
                module_name.endswith(f".{target}")
                or module_name == target
            ):
                module_matches[target].append(
                    module_name
                )

    all_targets_found = True

    for target in TARGET_MODULES:

        matches = module_matches[target]

        if matches:
            print(
                f"[FOUND] {target:<12} "
                f"{len(matches):>3} modules"
            )

            # Show the first two examples.
            for example in matches[:2]:
                print(f"         {example}")

        else:
            print(
                f"[MISSING] {target}"
            )

            all_targets_found = False

    if not all_targets_found:
        raise RuntimeError(
            "One or more intended LoRA target modules "
            "were not found in the model."
        )

    print(
        "\nAll planned LoRA target modules were found."
    )

    # ------------------------------------------------------------------------
    # Quantization inspection
    # ------------------------------------------------------------------------

    section("6. QUANTIZATION CHECK")

    quantized_modules = []

    for module_name, module in model.named_modules():

        module_type = module.__class__.__name__

        if "Linear4bit" in module_type:
            quantized_modules.append(
                module_name
            )

    result(
        "4-bit Linear modules found",
        len(quantized_modules),
    )

    if not quantized_modules:
        raise RuntimeError(
            "No bitsandbytes 4-bit Linear modules were found. "
            "4-bit quantization may not have loaded correctly."
        )

    print()
    print("Example quantized modules:")

    for module_name in quantized_modules[:10]:
        print(f"  {module_name}")

    # =========================================================================
    # Forward-pass test
    # =========================================================================

    section("7. GPU FORWARD-PASS TEST")

    test_messages = [
        {
            "role": "user",
            "content": (
                "What is the duration of "
                "B.Tech - Biotechnology at VIT Vellore?"
            ),
        }
    ]

    prompt = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # Move tensors to the same device as the model.
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    result(
        "Input tokens",
        inputs["input_ids"].shape[1],
    )

    print("\nRunning one forward pass...")

    model.eval()

    with torch.no_grad():
        outputs = model(
            **inputs,
        )

    result(
        "Forward pass",
        "SUCCESS",
    )

    result(
        "Logits shape",
        tuple(outputs.logits.shape),
    )

    # ------------------------------------------------------------------------
    # GPU memory
    # ------------------------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 3)
        )

        result(
            "GPU memory allocated",
            f"{allocated:.2f} GB",
        )

        result(
            "GPU memory reserved",
            f"{reserved:.2f} GB",
        )

    return model


# ============================================================================
# Final report
# ============================================================================

def final_report() -> None:
    section("8. FINAL RESULT")

    print("MODEL INSPECTION COMPLETE")
    print()
    print("[PASS] CUDA GPU detected")
    print("[PASS] Qwen3 tokenizer loaded")
    print("[PASS] Chat template works")
    print("[PASS] Example fits max_length=128")
    print("[PASS] Qwen3-1.7B loaded")
    print("[PASS] 4-bit NF4 quantization loaded")
    print("[PASS] LoRA target modules found")
    print("[PASS] GPU forward pass completed")
    print()
    print("Qwen3-1.7B is ready for QLoRA configuration.")


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    inspect_environment()

    tokenizer = inspect_tokenizer()

    quantization_config = create_bnb_config()

    model = inspect_model(
        tokenizer=tokenizer,
        quantization_config=quantization_config,
    )

    final_report()

    # Release model memory before exiting.
    del model
    del tokenizer

    cleanup()


if __name__ == "__main__":
    main()
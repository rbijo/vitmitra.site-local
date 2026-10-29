"""
VIT FAQ Support Chatbot
Test / Evaluation Script

Evaluates the trained QLoRA adapter on the untouched test set.

Model:
    Qwen/Qwen3-1.7B + LoRA adapter

Evaluation dataset:
    data/processed/test.jsonl

Outputs:
    evaluation/
        predictions.jsonl
        evaluation_report.json

Usage:
    python src/evaluate.py
"""

# ---------------------------------------------------------------------------
# CUDA allocator configuration
# ---------------------------------------------------------------------------

import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.jsonl"
)

ADAPTER_DIR = (
    PROJECT_ROOT
    / "models"
    / "vit-faq-qwen3-qlora"
    / "final_adapter"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "evaluation"
)

PREDICTIONS_FILE = (
    EVALUATION_DIR
    / "predictions.jsonl"
)

REPORT_FILE = (
    EVALUATION_DIR
    / "evaluation_report.json"
)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-1.7B"

MAX_INPUT_LENGTH = 128

MAX_NEW_TOKENS = 64

SEED = 42


# ---------------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------------

GENERATION_CONFIG = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    """Print a section header."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_gpu_memory(label: str) -> None:
    """Print current CUDA memory usage."""

    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    print(
        f"{label}: "
        f"allocated={allocated:.2f} GB, "
        f"reserved={reserved:.2f} GB"
    )


def normalize_text(text: str) -> str:
    """
    Normalize text for exact/lexical comparison.

    This is intentionally conservative.

    It:
        - converts to lowercase
        - removes think blocks
        - removes surrounding whitespace
        - collapses repeated whitespace
        - removes simple punctuation spacing differences
    """

    text = clean_response(text)

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # Normalize common punctuation spacing.
    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text,
    )

    return text.strip()


def clean_response(text: str) -> str:
    """
    Remove Qwen thinking blocks from generated output.

    The production backend should also apply an equivalent
    safety layer before returning responses to the frontend.
    """

    if not text:
        return ""

    # Remove complete thinking blocks.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove any remaining control tags.
    text = re.sub(
        r"</?think>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def tokenize_for_metrics(text: str) -> list[str]:
    """
    Create simple whitespace/punctuation-normalized tokens.

    This is deliberately lightweight and does not introduce
    another NLP dependency.
    """

    text = normalize_text(text)

    if not text:
        return []

    return re.findall(
        r"\b\w+(?:[-']\w+)*\b",
        text,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def exact_match(
    prediction: str,
    reference: str,
) -> float:
    """Return 1 if normalized answers match exactly."""

    return float(
        normalize_text(prediction)
        == normalize_text(reference)
    )


def token_f1(
    prediction: str,
    reference: str,
) -> tuple[float, float, float]:
    """
    Calculate token-level precision, recall and F1.

    Duplicate tokens are handled using Counter intersection.
    """

    prediction_tokens = tokenize_for_metrics(
        prediction
    )

    reference_tokens = tokenize_for_metrics(
        reference
    )

    if not prediction_tokens and not reference_tokens:
        return 1.0, 1.0, 1.0

    if not prediction_tokens or not reference_tokens:
        return 0.0, 0.0, 0.0

    prediction_counter = Counter(
        prediction_tokens
    )

    reference_counter = Counter(
        reference_tokens
    )

    common = sum(
        (
            prediction_counter
            & reference_counter
        ).values()
    )

    precision = (
        common / len(prediction_tokens)
        if prediction_tokens
        else 0.0
    )

    recall = (
        common / len(reference_tokens)
        if reference_tokens
        else 0.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return precision, recall, f1


def character_similarity(
    prediction: str,
    reference: str,
) -> float:
    """
    Calculate normalized character-level similarity.

    Uses SequenceMatcher so no additional dependency is required.
    """

    from difflib import SequenceMatcher

    prediction_normalized = normalize_text(
        prediction
    )

    reference_normalized = normalize_text(
        reference
    )

    return SequenceMatcher(
        None,
        prediction_normalized,
        reference_normalized,
    ).ratio()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_test_dataset() -> list[dict]:
    """Load the untouched processed test dataset."""

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_FILE}"
        )

    records = []

    with TEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            records.append(record)

    if not records:
        raise ValueError(
            "Test dataset is empty."
        )

    print_header("TEST DATASET")

    print(
        f"Test examples : {len(records)}"
    )

    print(
        f"Test file     : {TEST_FILE}"
    )

    return records


def get_question(record: dict) -> str:
    """
    Extract the question from the processed test record.

    Supports the known project schema and provides
    fallbacks for minor schema changes.
    """

    # Standard processed dataset schema.
    for key in (
        "question",
        "instruction",
        "query",
        "prompt",
    ):
        value = record.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    # Handle prompt-completion conversational format
    # if the processed file happens to contain it.
    prompt = record.get("prompt")

    if isinstance(prompt, list):
        for message in prompt:
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
            ):
                content = message.get("content")

                if (
                    isinstance(content, str)
                    and content.strip()
                ):
                    return content.strip()

    raise ValueError(
        "Could not find question field in record:\n"
        f"{record}"
    )


def get_reference_answer(record: dict) -> str:
    """
    Extract the reference answer.

    Supports the known project schema and
    prompt/completion fallback.
    """

    for key in (
        "answer",
        "response",
        "output",
        "completion",
    ):
        value = record.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    completion = record.get("completion")

    if isinstance(completion, list):
        for message in completion:
            if (
                isinstance(message, dict)
                and message.get("role") == "assistant"
            ):
                content = message.get("content")

                if (
                    isinstance(content, str)
                    and content.strip()
                ):
                    return content.strip()

    raise ValueError(
        "Could not find reference answer field "
        "in record:\n"
        f"{record}"
    )


def get_metadata(
    record: dict,
    key: str,
) -> str | None:
    """Safely extract optional metadata."""

    value = record.get(key)

    if value is None:
        return None

    return str(value)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_compute_dtype() -> torch.dtype:
    """Select BF16 when supported, otherwise FP16."""

    if (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16

    return torch.float16


def load_model_and_tokenizer():
    """Load base Qwen3 model and attach the trained adapter."""

    print_header("MODEL")

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Final adapter not found:\n{ADAPTER_DIR}\n\n"
            "Run training first with:\n"
            "python src/train.py"
        )

    compute_dtype = get_compute_dtype()

    print(
        f"Base model       : {MODEL_NAME}"
    )

    print(
        f"Adapter           : {ADAPTER_DIR}"
    )

    print(
        f"Compute dtype     : {compute_dtype}"
    )

    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_DIR),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # 4-bit quantization
    # -----------------------------------------------------------------------

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # -----------------------------------------------------------------------
    # Base model
    # -----------------------------------------------------------------------

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=compute_dtype,
        )
    )

    # -----------------------------------------------------------------------
    # LoRA adapter
    # -----------------------------------------------------------------------

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_DIR),
        is_trainable=False,
    )

    model.eval()

    print(
        f"Model class      : "
        f"{model.__class__.__name__}"
    )

    print(
        f"Model device     : "
        f"{model.device}"
    )

    print(
        "Adapter mode      : inference"
    )

    print_gpu_memory(
        "After model loading"
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    question: str,
) -> tuple[str, int, float]:
    """
    Generate one answer.

    Returns:
        answer
        generated token count
        generation latency in seconds
    """

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    # Explicitly disable Qwen3 thinking for inference.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    # Move tensors to the model's device.
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    input_token_count = (
        inputs["input_ids"].shape[1]
    )

    start_time = time.perf_counter()

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # Only decode newly generated tokens.
    generated_tokens = outputs[
        0,
        input_token_count:,
    ]

    generated_token_count = (
        generated_tokens.shape[0]
    )

    raw_answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    answer = clean_response(
        raw_answer
    )

    return (
        answer,
        generated_token_count,
        elapsed,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    records: list[dict],
) -> tuple[list[dict], dict]:
    """Generate predictions and calculate metrics."""

    print_header("EVALUATION")

    print(
        f"Examples          : {len(records)}"
    )

    print(
        f"Max new tokens    : {MAX_NEW_TOKENS}"
    )

    print(
        "Sampling          : disabled"
    )

    print(
        "Thinking          : disabled"
    )

    print(
        "Evaluation mode   : deterministic"
    )

    results = []

    exact_matches = []
    precisions = []
    recalls = []
    f1_scores = []
    similarities = []
    latencies = []
    generated_lengths = []

    category_results = defaultdict(
        lambda: {
            "count": 0,
            "exact_matches": [],
            "f1": [],
            "similarity": [],
            "latency": [],
        }
    )

    for index, record in enumerate(
        records,
        start=1,
    ):
        question = get_question(record)
        reference = get_reference_answer(record)

        record_id = get_metadata(
            record,
            "id",
        )

        category = (
            get_metadata(record, "category")
            or "Unknown"
        )

        entity = (
            get_metadata(
                record,
                "entity",
            )
            or get_metadata(
                record,
                "uniqueness_basis",
            )
            or "Unknown"
        )

        print(
            f"[{index:03d}/{len(records):03d}] "
            f"{question}"
        )

        prediction, generated_tokens, latency = (
            generate_answer(
                model=model,
                tokenizer=tokenizer,
                question=question,
            )
        )

        em = exact_match(
            prediction,
            reference,
        )

        precision, recall, f1 = token_f1(
            prediction,
            reference,
        )

        similarity = character_similarity(
            prediction,
            reference,
        )

        exact_matches.append(em)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        similarities.append(similarity)
        latencies.append(latency)
        generated_lengths.append(
            generated_tokens
        )

        category_data = (
            category_results[category]
        )

        category_data["count"] += 1
        category_data["exact_matches"].append(
            em
        )
        category_data["f1"].append(
            f1
        )
        category_data["similarity"].append(
            similarity
        )
        category_data["latency"].append(
            latency
        )

        results.append(
            {
                "index": index,
                "id": record_id,
                "category": category,
                "entity": entity,
                "question": question,
                "reference_answer": reference,
                "generated_answer": prediction,
                "exact_match": em,
                "token_precision": precision,
                "token_recall": recall,
                "token_f1": f1,
                "character_similarity": similarity,
                "generated_tokens": generated_tokens,
                "latency_seconds": latency,
            }
        )

    # -----------------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------------

    metrics = {
        "test_examples": len(records),

        "exact_match": (
            sum(exact_matches)
            / len(exact_matches)
        ),

        "token_precision": (
            sum(precisions)
            / len(precisions)
        ),

        "token_recall": (
            sum(recalls)
            / len(recalls)
        ),

        "token_f1": (
            sum(f1_scores)
            / len(f1_scores)
        ),

        "character_similarity": (
            sum(similarities)
            / len(similarities)
        ),

        "average_latency_seconds": (
            sum(latencies)
            / len(latencies)
        ),

        "median_latency_seconds": (
            statistics.median(latencies)
        ),

        "p95_latency_seconds": (
            sorted(latencies)[
                min(
                    len(latencies) - 1,
                    math.ceil(
                        0.95 * len(latencies)
                    ) - 1,
                )
            ]
        ),

        "average_generated_tokens": (
            sum(generated_lengths)
            / len(generated_lengths)
        ),

        "exact_match_count": int(
            sum(exact_matches)
        ),
    }

    # -----------------------------------------------------------------------
    # Category metrics
    # -----------------------------------------------------------------------

    category_metrics = {}

    for category, data in sorted(
        category_results.items()
    ):
        category_metrics[category] = {
            "count": data["count"],

            "exact_match": (
                sum(data["exact_matches"])
                / data["count"]
            ),

            "token_f1": (
                sum(data["f1"])
                / data["count"]
            ),

            "character_similarity": (
                sum(data["similarity"])
                / data["count"]
            ),

            "average_latency_seconds": (
                sum(data["latency"])
                / data["count"]
            ),
        }

    return results, {
        "aggregate": metrics,
        "by_category": category_metrics,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_predictions(
    results: list[dict],
) -> None:
    """Save individual predictions as JSONL."""

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        for result in results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"[PASS] Predictions saved to:\n"
        f"{PREDICTIONS_FILE}"
    )


def save_report(
    metrics: dict,
) -> None:
    """Save aggregate evaluation report."""

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "model": MODEL_NAME,
        "adapter": str(ADAPTER_DIR),
        "test_dataset": str(TEST_FILE),

        "evaluation_configuration": {
            "max_input_length": MAX_INPUT_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "thinking_enabled": False,
            "quantization": "4-bit NF4",
            "double_quantization": True,
            "compute_dtype": (
                "bfloat16"
                if get_compute_dtype()
                == torch.bfloat16
                else "float16"
            ),
            "seed": SEED,
        },

        "metrics": metrics,
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[PASS] Evaluation report saved to:\n"
        f"{REPORT_FILE}"
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    metrics: dict,
) -> None:
    """Print final evaluation summary."""

    aggregate = metrics["aggregate"]

    print_header("FINAL TEST RESULTS")

    print(
        f"Test examples            : "
        f"{aggregate['test_examples']}"
    )

    print(
        f"Exact Match              : "
        f"{aggregate['exact_match']:.4f} "
        f"({aggregate['exact_match'] * 100:.2f}%)"
    )

    print(
        f"Token Precision          : "
        f"{aggregate['token_precision']:.4f}"
    )

    print(
        f"Token Recall             : "
        f"{aggregate['token_recall']:.4f}"
    )

    print(
        f"Token F1                 : "
        f"{aggregate['token_f1']:.4f}"
    )

    print(
        f"Character Similarity     : "
        f"{aggregate['character_similarity']:.4f}"
    )

    print(
        f"Average Latency          : "
        f"{aggregate['average_latency_seconds']:.4f} s"
    )

    print(
        f"Median Latency           : "
        f"{aggregate['median_latency_seconds']:.4f} s"
    )

    print(
        f"P95 Latency              : "
        f"{aggregate['p95_latency_seconds']:.4f} s"
    )

    print(
        f"Average Generated Tokens : "
        f"{aggregate['average_generated_tokens']:.2f}"
    )

    print(
        f"Exact Matches             : "
        f"{aggregate['exact_match_count']}/"
        f"{aggregate['test_examples']}"
    )

    # -----------------------------------------------------------------------
    # Category results
    # -----------------------------------------------------------------------

    print_header("RESULTS BY CATEGORY")

    print(
        f"{'Category':35s} "
        f"{'N':>4s} "
        f"{'EM':>8s} "
        f"{'F1':>8s} "
        f"{'Similarity':>12s}"
    )

    print("-" * 75)

    for category, data in metrics[
        "by_category"
    ].items():

        print(
            f"{category[:35]:35s} "
            f"{data['count']:4d} "
            f"{data['exact_match'] * 100:7.2f}% "
            f"{data['token_f1']:8.4f} "
            f"{data['character_similarity']:11.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete test evaluation."""

    print("=" * 80)
    print("VIT FAQ QLORA MODEL EVALUATION")
    print("=" * 80)

    print(
        "\nIMPORTANT: "
        "This evaluation uses the untouched test set."
    )

    # Reproducibility.
    set_seed(SEED)

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------

    records = load_test_dataset()

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model, tokenizer = (
        load_model_and_tokenizer()
    )

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    results, metrics = evaluate(
        model=model,
        tokenizer=tokenizer,
        records=records,
    )

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------

    save_predictions(results)

    save_report(metrics)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------

    print_summary(metrics)

    print_gpu_memory(
        "Final GPU memory"
    )

    print()
    print("=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

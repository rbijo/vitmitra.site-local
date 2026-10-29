#!/usr/bin/env python3
"""
VIT FAQ Dataset Distribution + Tokenizer Analysis

Reads:
    data/processed/train.jsonl
    data/processed/validation.jsonl
    data/processed/test.jsonl

Reports:
    - split sizes
    - category distribution
    - quality-tier distribution
    - question/answer word-length statistics
    - category coverage across splits
    - marginal-sample distribution
    - entity-group counts
    - tokenizer statistics for candidate models
    - actual rendered chat-template examples
    - token-length statistics
    - sequence-length coverage

This script performs analysis only.
It does NOT modify the dataset.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


SPLIT_FILES = {
    "train": PROCESSED_DIR / "train.jsonl",
    "validation": PROCESSED_DIR / "validation.jsonl",
    "test": PROCESSED_DIR / "test.jsonl",
}


# ============================================================================
# MODEL CANDIDATES
# ============================================================================

MODELS = {
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
}


# ============================================================================
# UTILITIES
# ============================================================================


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def percentile(
    values: list[int],
    p: float,
) -> int:

    if not values:
        return 0

    values = sorted(values)

    index = (len(values) - 1) * p

    lower = int(index)

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    if lower == upper:
        return values[lower]

    fraction = index - lower

    return round(
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * fraction
    )


def stats(
    values: list[int],
) -> dict:

    if not values:
        return {}

    return {
        "min": min(values),
        "max": max(values),
        "mean": round(
            statistics.mean(values),
            2,
        ),
        "p50": percentile(
            values,
            0.50,
        ),
        "p90": percentile(
            values,
            0.90,
        ),
        "p95": percentile(
            values,
            0.95,
        ),
        "p99": percentile(
            values,
            0.99,
        ),
    }


def distribution(
    records: list[dict],
    metadata_key: str,
) -> Counter:

    return Counter(
        record["metadata"].get(
            metadata_key
        )
        for record in records
    )


def print_distribution(
    title: str,
    counts: Counter,
    total: int,
) -> None:

    print()
    print(title)
    print("-" * len(title))

    for key, count in counts.most_common():

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{str(key):35s} "
            f"{count:4d} "
            f"({percentage:5.1f}%)"
        )


# ============================================================================
# DATASET DISTRIBUTION AUDIT
# ============================================================================


def analyze_distributions(
    splits: dict[str, list[dict]],
) -> None:

    print()
    print("=" * 70)
    print("DATASET DISTRIBUTION AUDIT")
    print("=" * 70)

    total = sum(
        len(records)
        for records in splits.values()
    )

    # ------------------------------------------------------------------------
    # Split sizes
    # ------------------------------------------------------------------------

    print()
    print("Split sizes:")

    for split_name, records in splits.items():

        percentage = (
            len(records) / total * 100
            if total
            else 0
        )

        print(
            f"  {split_name:12s}: "
            f"{len(records):4d} "
            f"({percentage:5.1f}%)"
        )

    # ------------------------------------------------------------------------
    # Per-split statistics
    # ------------------------------------------------------------------------

    for split_name, records in splits.items():

        print()
        print("=" * 70)
        print(
            f"{split_name.upper()} DISTRIBUTION"
        )
        print("=" * 70)

        # Categories
        print_distribution(
            "Categories",
            distribution(
                records,
                "category",
            ),
            len(records),
        )

        # Quality tiers
        print_distribution(
            "Quality tiers",
            distribution(
                records,
                "quality_tier",
            ),
            len(records),
        )

        # Question word lengths
        question_lengths = [
            len(
                record["messages"][0][
                    "content"
                ].split()
            )
            for record in records
        ]

        # Answer word lengths
        answer_lengths = [
            len(
                record["messages"][1][
                    "content"
                ].split()
            )
            for record in records
        ]

        print()
        print("Question word lengths:")
        print(
            json.dumps(
                stats(question_lengths),
                indent=2,
            )
        )

        print()
        print("Answer word lengths:")
        print(
            json.dumps(
                stats(answer_lengths),
                indent=2,
            )
        )

    # ------------------------------------------------------------------------
    # Category coverage
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("CATEGORY COVERAGE")
    print("=" * 70)

    category_counts = {
        split_name: distribution(
            records,
            "category",
        )
        for split_name, records
        in splits.items()
    }

    all_categories = set()

    for counts in category_counts.values():
        all_categories.update(
            counts.keys()
        )

    for category in sorted(
        all_categories,
        key=lambda x: str(x).lower(),
    ):

        values = []

        for split_name in splits:

            count = category_counts[
                split_name
            ].get(
                category,
                0,
            )

            values.append(
                f"{split_name}={count}"
            )

        print(
            f"{str(category):35s} "
            + " | ".join(values)
        )

    # ------------------------------------------------------------------------
    # Marginal quality distribution
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("MARGINAL SAMPLE DISTRIBUTION")
    print("=" * 70)

    for split_name, records in splits.items():

        marginal = sum(
            record["metadata"].get(
                "quality_tier"
            )
            == "marginal"
            for record in records
        )

        percentage = (
            marginal / len(records) * 100
            if records
            else 0
        )

        print(
            f"  {split_name:12s}: "
            f"{marginal:4d} "
            f"({percentage:5.1f}%)"
        )

    # ------------------------------------------------------------------------
    # Entity groups
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("ENTITY GROUP COUNTS")
    print("=" * 70)

    for split_name, records in splits.items():

        groups = {
            record["metadata"].get(
                "group_key"
            )
            for record in records
        }

        print(
            f"  {split_name:12s}: "
            f"{len(groups):4d} groups"
        )


# ============================================================================
# CHAT FORMAT
# ============================================================================


def build_chat_messages(
    record: dict,
) -> list[dict]:

    """
    Reconstruct the exact two-message conversational
    structure used by the processed dataset.
    """

    return [
        {
            "role": "user",
            "content": record[
                "messages"
            ][0]["content"],
        },
        {
            "role": "assistant",
            "content": record[
                "messages"
            ][1]["content"],
        },
    ]


# ============================================================================
# TOKENIZER ANALYSIS
# ============================================================================


def analyze_tokenizer(
    tokenizer_name: str,
    tokenizer_path: str,
    splits: dict[str, list[dict]],
) -> None:

    print()
    print("=" * 70)
    print(
        f"TOKENIZER: {tokenizer_name}"
    )
    print("=" * 70)

    print()
    print(
        f"Loading: {tokenizer_path}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path
    )

    print(
        f"Vocabulary size: "
        f"{len(tokenizer):,}"
    )

    # ------------------------------------------------------------------------
    # Verify chat template
    # ------------------------------------------------------------------------

    if tokenizer.chat_template is None:

        print()
        print(
            "WARNING: tokenizer has no "
            "chat_template."
        )

        return

    # ------------------------------------------------------------------------
    # Show one actual rendered example
    # ------------------------------------------------------------------------

    example_messages = build_chat_messages(
        splits["train"][0]
    )

    rendered_example = (
        tokenizer.apply_chat_template(
            example_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    )

    print()
    print(
        "Example rendered chat:"
    )
    print("-" * 70)
    print(rendered_example)
    print("-" * 70)

    example_token_ids = tokenizer(
        rendered_example,
        add_special_tokens=False,
    )["input_ids"]

    print(
        f"Example token count: "
        f"{len(example_token_ids)}"
    )

    # ------------------------------------------------------------------------
    # Tokenize every split
    # ------------------------------------------------------------------------

    lengths_by_split = {}

    for split_name, records in splits.items():

        lengths = []

        for record in records:

            messages = build_chat_messages(
                record
            )

            # IMPORTANT:
            # First render the model-specific chat template.
            rendered = (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

            # Then explicitly tokenize the rendered string.
            token_ids = tokenizer(
                rendered,
                add_special_tokens=False,
            )["input_ids"]

            lengths.append(
                len(token_ids)
            )

        lengths_by_split[
            split_name
        ] = lengths

        print()
        print(
            split_name.upper()
        )

        print(
            json.dumps(
                stats(lengths),
                indent=2,
            )
        )

    # ------------------------------------------------------------------------
    # Combined statistics
    # ------------------------------------------------------------------------

    all_lengths = [
        length
        for lengths
        in lengths_by_split.values()
        for length in lengths
    ]

    print()
    print("COMBINED")

    print(
        json.dumps(
            stats(all_lengths),
            indent=2,
        )
    )

    # ------------------------------------------------------------------------
    # Sequence-length coverage
    # ------------------------------------------------------------------------

    print()
    print(
        "SEQUENCE LENGTH COVERAGE"
    )

    sequence_lengths = [
        64,
        96,
        128,
        160,
        192,
        256,
        384,
        512,
    ]

    for limit in sequence_lengths:

        within = sum(
            length <= limit
            for length in all_lengths
        )

        percentage = (
            within / len(all_lengths) * 100
            if all_lengths
            else 0
        )

        print(
            f"  <= {limit:4d} tokens: "
            f"{within:4d}/{len(all_lengths)} "
            f"({percentage:6.2f}%)"
        )

    # ------------------------------------------------------------------------
    # Padding waste estimate
    # ------------------------------------------------------------------------

    if all_lengths:

        max_length = max(all_lengths)

        print()
        print(
            "MAX OBSERVED LENGTH"
        )
        print(
            f"  {max_length} tokens"
        )

        print()
        print(
            "AVERAGE PADDING WASTE "
            "AT COMMON SEQUENCE LENGTHS"
        )

        for limit in [
            64,
            96,
            128,
            160,
            192,
            256,
            384,
            512,
        ]:

            truncated = sum(
                length > limit
                for length in all_lengths
            )

            if truncated:
                print(
                    f"  {limit:4d}: "
                    f"{truncated} examples "
                    f"would be truncated"
                )
                continue

            average_used = (
                statistics.mean(
                    all_lengths
                )
            )

            waste = (
                1
                - average_used / limit
            ) * 100

            print(
                f"  {limit:4d}: "
                f"approximately "
                f"{waste:5.1f}% padding"
            )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    print()
    print("=" * 70)
    print(
        "VIT FAQ DATASET ANALYSIS"
    )
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Verify files
    # ------------------------------------------------------------------------

    splits = {}

    for split_name, path in SPLIT_FILES.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Missing split file:\n"
                f"{path}"
            )

        splits[
            split_name
        ] = load_jsonl(path)

    # ------------------------------------------------------------------------
    # Dataset distribution audit
    # ------------------------------------------------------------------------

    analyze_distributions(
        splits
    )

    # ------------------------------------------------------------------------
    # Tokenizer analysis
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TOKENIZER ANALYSIS"
    )
    print("=" * 70)

    for model_name, model_path in MODELS.items():

        try:

            analyze_tokenizer(
                model_name,
                model_path,
                splits,
            )

        except Exception as exc:

            print()
            print(
                f"ERROR analyzing "
                f"{model_name}:"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    print()
    print("=" * 70)
    print(
        "ANALYSIS COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
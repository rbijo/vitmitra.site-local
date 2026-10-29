"""
Prepare the VIT FAQ dataset using an entity-aware, category-aware split.

Split strategy
--------------
1. Parse `uniqueness_basis` into:
       PREFIX|ENTITY|PROPERTY

   Example:
       UG|B.Tech - Biotechnology|duration

   Entity:
       B.Tech - Biotechnology

2. All records belonging to the same entity are kept in the same split.

3. Entities are allocated approximately 80/10/10 across:
       train / validation / test

4. Allocation is category-aware:
   entities are assigned with consideration for the category distribution.

5. Validation checks ensure:
   - every record appears exactly once
   - IDs are conserved
   - questions are conserved
   - no entity appears in multiple splits
   - category coverage is reported
   - data quality is reported

Input:
    data/raw/vit_cleaned_1000_qa.jsonl

Output:
    data/processed/train.jsonl
    data/processed/validation.jsonl
    data/processed/test.jsonl
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

INPUT_PATH = Path("data/raw/vit_cleaned_1000_qa.jsonl")
OUTPUT_DIR = Path("data/processed")

SPLITS = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )

            records.append(record)

    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Save records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def normalize_whitespace(value: str) -> str:
    """Normalize repeated whitespace."""
    return re.sub(r"\s+", " ", value.strip())


def normalize_question(question: str) -> str:
    """
    Normalize a question for reporting template similarity.

    Entity-like phrases are replaced with [ENTITY] using the
    uniqueness_basis entity where possible elsewhere in the pipeline.
    """
    question = normalize_whitespace(question.lower())

    # Normalize common punctuation.
    question = re.sub(r"[?!.,;:]+$", "", question)

    return question


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def parse_uniqueness_basis(value: Any) -> tuple[str, str, str]:
    """
    Parse uniqueness_basis.

    Expected format:
        PREFIX|ENTITY|PROPERTY

    Example:
        UG|B.Tech - Biotechnology|duration

    Returns:
        prefix, entity, property
    """
    if not isinstance(value, str):
        raise ValueError(
            f"`uniqueness_basis` must be a string, got {type(value).__name__}"
        )

    value = value.strip()

    parts = value.split("|")

    if len(parts) < 3:
        raise ValueError(
            "Invalid uniqueness_basis format. "
            f"Expected PREFIX|ENTITY|PROPERTY, got: {value!r}"
        )

    prefix = parts[0].strip()
    property_name = parts[-1].strip()

    # Everything between prefix and property is treated as the entity.
    entity = "|".join(parts[1:-1]).strip()

    if not prefix:
        raise ValueError(
            f"Empty prefix in uniqueness_basis: {value!r}"
        )

    if not entity:
        raise ValueError(
            f"Empty entity in uniqueness_basis: {value!r}"
        )

    if not property_name:
        raise ValueError(
            f"Empty property in uniqueness_basis: {value!r}"
        )

    return prefix, entity, property_name


def derive_entity(record: dict[str, Any]) -> str:
    """Extract the entity used as the leakage-prevention group."""
    if "uniqueness_basis" not in record:
        raise KeyError(
            "Dataset record does not contain `uniqueness_basis`."
        )

    _, entity, _ = parse_uniqueness_basis(
        record["uniqueness_basis"]
    )

    return entity


# ---------------------------------------------------------------------------
# Dataset validation before splitting
# ---------------------------------------------------------------------------

def validate_input_records(records: list[dict[str, Any]]) -> None:
    """Validate required fields and obvious dataset problems."""

    if not records:
        raise ValueError("Dataset is empty.")

    required_fields = {
        "id",
        "question",
        "answer",
        "category",
        "uniqueness_basis",
    }

    missing_fields: Counter[str] = Counter()

    for index, record in enumerate(records):
        missing = required_fields - set(record.keys())

        for field in missing:
            missing_fields[field] += 1

        if missing:
            raise ValueError(
                f"Record {index} is missing fields: {sorted(missing)}"
            )

        if not str(record["question"]).strip():
            raise ValueError(f"Record {index} has an empty question.")

        if not str(record["answer"]).strip():
            raise ValueError(f"Record {index} has an empty answer.")

        # Make sure the entity structure is valid.
        parse_uniqueness_basis(record["uniqueness_basis"])

    if missing_fields:
        raise ValueError(
            f"Missing fields detected: {dict(missing_fields)}"
        )

    ids = [str(record["id"]) for record in records]

    duplicate_ids = [
        item
        for item, count in Counter(ids).items()
        if count > 1
    ]

    if duplicate_ids:
        raise ValueError(
            f"Duplicate IDs found: {duplicate_ids[:10]}"
        )

    questions = [
        normalize_whitespace(str(record["question"]).lower())
        for record in records
    ]

    duplicate_questions = [
        item
        for item, count in Counter(questions).items()
        if count > 1
    ]

    if duplicate_questions:
        print(
            f"WARNING: {len(duplicate_questions)} duplicate question(s) "
            "exist in the source dataset."
        )


# ---------------------------------------------------------------------------
# Entity groups
# ---------------------------------------------------------------------------

def build_entity_groups(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group records by entity.

    Every record for an entity must stay in the same split.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        entity = derive_entity(record)
        groups[entity].append(record)

    return dict(groups)


def entity_category_profile(
    group: list[dict[str, Any]],
) -> Counter[str]:
    """Return category counts for an entity group."""
    return Counter(
        str(record["category"]).strip()
        for record in group
    )


# ---------------------------------------------------------------------------
# Target calculations
# ---------------------------------------------------------------------------

def target_sizes(total_records: int) -> dict[str, float]:
    """Return desired record counts for each split."""
    return {
        "train": total_records * TRAIN_RATIO,
        "validation": total_records * VALIDATION_RATIO,
        "test": total_records * TEST_RATIO,
    }


def target_category_sizes(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Calculate ideal category counts for each split.

    The category distribution of the complete dataset is used as the
    reference distribution.
    """

    total = len(records)

    category_counts = Counter(
        str(record["category"]).strip()
        for record in records
    )

    targets: dict[str, dict[str, float]] = {
        split: {}
        for split in SPLITS
    }

    for category, count in category_counts.items():
        category_ratio = count / total

        for split in SPLITS:
            targets[split][category] = (
                total * {
                    "train": TRAIN_RATIO,
                    "validation": VALIDATION_RATIO,
                    "test": TEST_RATIO,
                }[split] * category_ratio
            )

    return targets


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

def allocation_score(
    group: list[dict[str, Any]],
    split: str,
    split_records: dict[str, int],
    split_categories: dict[str, Counter[str]],
    record_targets: dict[str, float],
    category_targets: dict[str, dict[str, float]],
) -> float:
    """
    Calculate how suitable it is to place an entity group into a split.

    Lower score is better.

    The score considers:
      - record-count deviation
      - category-count deviation
      - stronger penalty for overshooting small validation/test splits
    """

    group_size = len(group)
    group_categories = entity_category_profile(group)

    current_records = split_records[split]
    current_categories = split_categories[split]

    target_records = record_targets[split]

    # ---------------------------------------------------------------
    # Record-size component
    # ---------------------------------------------------------------

    new_record_count = current_records + group_size

    before_record_error = abs(
        current_records - target_records
    )

    after_record_error = abs(
        new_record_count - target_records
    )

    record_penalty = after_record_error - before_record_error

    # ---------------------------------------------------------------
    # Category-distribution component
    # ---------------------------------------------------------------

    category_penalty = 0.0

    for category, count in group_categories.items():
        target = category_targets[split].get(category, 0.0)
        current = current_categories.get(category, 0)

        before_error = abs(current - target)
        after_error = abs(current + count - target)

        category_penalty += after_error - before_error

    # ---------------------------------------------------------------
    # Overshoot penalty
    # ---------------------------------------------------------------

    overshoot = max(
        0,
        new_record_count - target_records,
    )

    overshoot_penalty = overshoot * 2.0

    # Validation/test are small splits, so overshooting them is
    # particularly undesirable.
    if split != "train":
        overshoot_penalty *= 2.0

    return (
        record_penalty
        + category_penalty
        + overshoot_penalty
    )


def allocate_entities(
    groups: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
    seed: int = SEED,
) -> dict[str, list[str]]:
    """
    Allocate entity groups to train/validation/test.

    The algorithm:
      1. Sorts larger entity groups first.
      2. Adds controlled randomness using the fixed seed.
      3. Places each entity into the split with the lowest allocation score.
      4. Uses a deterministic tie-break.

    This is a greedy approximation rather than an exact optimizer.
    """

    rng = random.Random(seed)

    record_targets = target_sizes(len(records))
    category_targets = target_category_sizes(records)

    split_records = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    split_categories = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }

    # ------------------------------------------------------------------
    # Prepare entities.
    #
    # Larger groups are allocated first because placing a large entity
    # group late can make the 80/10/10 target difficult to achieve.
    # ------------------------------------------------------------------

    entities = list(groups.keys())

    rng.shuffle(entities)

    entities.sort(
        key=lambda entity: len(groups[entity]),
        reverse=True,
    )

    assignments: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    # ------------------------------------------------------------------
    # Greedy assignment.
    # ------------------------------------------------------------------

    for entity in entities:
        group = groups[entity]

        scores = {
            split: allocation_score(
                group=group,
                split=split,
                split_records=split_records,
                split_categories=split_categories,
                record_targets=record_targets,
                category_targets=category_targets,
            )
            for split in SPLITS
        }

        # Deterministic tie-breaking:
        # prefer the split that is currently furthest below target.
        min_score = min(scores.values())

        candidates = [
            split
            for split, score in scores.items()
            if abs(score - min_score) < 1e-9
        ]

        if len(candidates) > 1:
            candidates.sort(
                key=lambda split: (
                    split_records[split] / max(
                        record_targets[split],
                        1,
                    ),
                    split,
                )
            )

        chosen_split = candidates[0]

        assignments[chosen_split].append(entity)

        split_records[chosen_split] += len(group)

        for category, count in entity_category_profile(group).items():
            split_categories[chosen_split][category] += count

    return assignments


# ---------------------------------------------------------------------------
# Materialize split records
# ---------------------------------------------------------------------------

def materialize_splits(
    groups: dict[str, list[dict[str, Any]]],
    assignments: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert entity assignments into record lists."""

    split_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for split in SPLITS:
        for entity in assignments[split]:
            split_records[split].extend(groups[entity])

    # Stable ordering within each split.
    for split in SPLITS:
        split_records[split].sort(
            key=lambda record: str(record["id"])
        )

    return split_records


# ---------------------------------------------------------------------------
# Template reporting
# ---------------------------------------------------------------------------

def template_from_record(record: dict[str, Any]) -> str:
    """
    Produce a simple entity-normalized template for reporting.

    The entity comes from uniqueness_basis, rather than trying to infer
    entities directly from the question text.
    """

    question = normalize_question(
        str(record["question"])
    )

    entity = normalize_question(
        derive_entity(record)
    )

    if entity:
        question = question.replace(
            entity,
            "[ENTITY]",
        )

    return question


def repeated_templates(
    records: list[dict[str, Any]],
) -> Counter[str]:
    """Return templates occurring more than once."""
    templates = Counter(
        template_from_record(record)
        for record in records
    )

    return Counter({
        template: count
        for template, count in templates.items()
        if count > 1
    })


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_split_integrity(
    original: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    """Validate conservation and leakage properties."""

    print("\n" + "=" * 80)
    print("SPLIT VALIDATION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Record conservation
    # ------------------------------------------------------------------

    original_ids = [
        str(record["id"])
        for record in original
    ]

    split_ids = [
        str(record["id"])
        for split in SPLITS
        for record in splits[split]
    ]

    print(
        f"\nRecord conservation: "
        f"{len(split_ids)}/{len(original_ids)}"
    )

    if (
        len(split_ids) == len(original_ids)
        and len(set(split_ids)) == len(original_ids)
    ):
        print("PASS")
    else:
        raise AssertionError(
            "Record conservation FAILED."
        )

    # ------------------------------------------------------------------
    # ID conservation
    # ------------------------------------------------------------------

    if set(split_ids) == set(original_ids):
        print("ID conservation: PASS")
    else:
        raise AssertionError(
            "ID conservation FAILED."
        )

    # ------------------------------------------------------------------
    # Question conservation
    # ------------------------------------------------------------------

    original_questions = Counter(
        normalize_whitespace(
            str(record["question"]).lower()
        )
        for record in original
    )

    split_questions = Counter(
        normalize_whitespace(
            str(record["question"]).lower()
        )
        for split in SPLITS
        for record in splits[split]
    )

    if original_questions == split_questions:
        print("Question conservation: PASS")
    else:
        raise AssertionError(
            "Question conservation FAILED."
        )

    # ------------------------------------------------------------------
    # Entity leakage
    # ------------------------------------------------------------------

    entity_sets: dict[str, set[str]] = {}

    for split in SPLITS:
        entity_sets[split] = {
            derive_entity(record)
            for record in splits[split]
        }

    leakage_pairs = []

    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1:]:
            overlap = (
                entity_sets[split_a]
                & entity_sets[split_b]
            )

            if overlap:
                leakage_pairs.append(
                    (
                        split_a,
                        split_b,
                        overlap,
                    )
                )

    if not leakage_pairs:
        print("Entity leakage: PASS")
    else:
        print("Entity leakage: FAIL")

        for split_a, split_b, overlap in leakage_pairs:
            print(
                f"  {split_a} ↔ {split_b}: "
                f"{len(overlap)} entities"
            )

        raise AssertionError(
            "Entity leakage detected."
        )

    # ------------------------------------------------------------------
    # Template leakage
    #
    # This is informational rather than the primary grouping rule.
    # Different entities may legitimately use the same question
    # formulation. Since entities are already isolated, that does not
    # represent factual leakage.
    # ------------------------------------------------------------------

    template_sets = {
        split: {
            template_from_record(record)
            for record in splits[split]
        }
        for split in SPLITS
    }

    print(
        "\nTemplate overlap:"
    )

    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1:]:
            overlap = (
                template_sets[split_a]
                & template_sets[split_b]
            )

            print(
                f"  {split_a} ↔ {split_b}: "
                f"{len(overlap)} shared template(s)"
            )

    print(
        "\nNote: shared templates across splits are allowed. "
        "Entity isolation is the primary leakage-control mechanism."
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def quality_label(record: dict[str, Any]) -> str:
    """
    Determine dataset quality label.

    Supports the current dataset's existing `quality` field if present.
    Otherwise returns 'unknown'.
    """

    quality = record.get("quality")

    if quality is None:
        return "unknown"

    return str(quality).strip().lower()


def word_count(value: str) -> int:
    """Count whitespace-separated words."""
    return len(str(value).split())


def print_split_report(
    split: str,
    records: list[dict[str, Any]],
) -> None:
    """Print statistics for a split."""

    category_counts = Counter(
        str(record["category"]).strip()
        for record in records
    )

    quality_counts = Counter(
        quality_label(record)
        for record in records
    )

    question_lengths = [
        word_count(record["question"])
        for record in records
    ]

    answer_lengths = [
        word_count(record["answer"])
        for record in records
    ]

    templates = Counter(
        template_from_record(record)
        for record in records
    )

    repeated = {
        template: count
        for template, count in templates.items()
        if count > 1
    }

    print("\n" + "-" * 80)
    print(split.upper())
    print("-" * 80)

    print(f"Records: {len(records)}")
    print(f"Entities: {len(set(derive_entity(r) for r in records))}")
    print(f"Templates: {len(templates)}")
    print(f"Repeated templates: {len(repeated)}")

    if question_lengths:
        print(
            f"Mean question words: "
            f"{sum(question_lengths) / len(question_lengths):.2f}"
        )

    if answer_lengths:
        print(
            f"Mean answer words: "
            f"{sum(answer_lengths) / len(answer_lengths):.2f}"
        )

    print("\nQuality:")

    for label, count in sorted(quality_counts.items()):
        percentage = count / len(records) * 100
        print(
            f"  {label}: {count} ({percentage:.1f}%)"
        )

    print("\nCategories:")

    for category, count in sorted(
        category_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(
            f"  {category}: {count}"
        )


def print_overall_report(
    records: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    assignments: dict[str, list[str]],
) -> None:
    """Print overall entity/split information."""

    print("\n" + "=" * 80)
    print("ENTITY-AWARE SPLIT SUMMARY")
    print("=" * 80)

    print(f"\nTotal records: {len(records)}")
    print(f"Unique entities: {len(groups)}")

    group_sizes = [
        len(group)
        for group in groups.values()
    ]

    print(
        f"Entity size: "
        f"min={min(group_sizes)}, "
        f"max={max(group_sizes)}, "
        f"mean={sum(group_sizes) / len(group_sizes):.2f}"
    )

    print("\nTarget split:")
    print(
        f"  Train:      {TRAIN_RATIO:.0%}"
    )
    print(
        f"  Validation: {VALIDATION_RATIO:.0%}"
    )
    print(
        f"  Test:       {TEST_RATIO:.0%}"
    )

    print("\nAssigned entities:")

    for split in SPLITS:
        print(
            f"  {split}: "
            f"{len(assignments[split])}"
        )


def print_category_coverage(
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    """Print category distribution across all splits."""

    categories = sorted({
        str(record["category"]).strip()
        for split in SPLITS
        for record in splits[split]
    })

    print("\n" + "=" * 80)
    print("CATEGORY COVERAGE")
    print("=" * 80)

    header = (
        f"{'Category':<30}"
        f"{'Train':>10}"
        f"{'Val':>10}"
        f"{'Test':>10}"
    )

    print(header)
    print("-" * len(header))

    for category in categories:
        counts = []

        for split in SPLITS:
            count = sum(
                1
                for record in splits[split]
                if str(record["category"]).strip() == category
            )

            counts.append(count)

        print(
            f"{category:<30}"
            f"{counts[0]:>10}"
            f"{counts[1]:>10}"
            f"{counts[2]:>10}"
        )


def print_repeated_templates(
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    """Print repeated question templates per split."""

    print("\n" + "=" * 80)
    print("REPEATED QUESTION TEMPLATES")
    print("=" * 80)

    for split in SPLITS:
        repeated = repeated_templates(
            splits[split]
        )

        print(f"\n{split.upper()}:")

        if not repeated:
            print("  None")
            continue

        for template, count in sorted(
            repeated.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(
                f"  {count:>3} | {template}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("VIT FAQ DATASET PREPARATION")
    print("Entity-aware + category-aware splitting")
    print("=" * 80)

    print(f"\nInput:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Seed:   {SEED}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    records = load_jsonl(INPUT_PATH)

    print(
        f"\nLoaded {len(records)} records."
    )

    # ------------------------------------------------------------------
    # Validate source
    # ------------------------------------------------------------------

    validate_input_records(records)

    # ------------------------------------------------------------------
    # Build entity groups
    # ------------------------------------------------------------------

    groups = build_entity_groups(records)

    print(
        f"Unique entities: {len(groups)}"
    )

    # Show largest entity groups.
    largest_groups = sorted(
        (
            (entity, len(group))
            for entity, group in groups.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    print("\nLargest entity groups:")

    for entity, count in largest_groups:
        print(
            f"  {count:>3} | {entity}"
        )

    # ------------------------------------------------------------------
    # Allocate
    # ------------------------------------------------------------------

    assignments = allocate_entities(
        groups=groups,
        records=records,
        seed=SEED,
    )

    # ------------------------------------------------------------------
    # Materialize
    # ------------------------------------------------------------------

    splits = materialize_splits(
        groups=groups,
        assignments=assignments,
    )

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    validate_split_integrity(
        original=records,
        splits=splits,
    )

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    print_overall_report(
        records=records,
        groups=groups,
        assignments=assignments,
    )

    for split in SPLITS:
        print_split_report(
            split=split,
            records=splits[split],
        )

    print_category_coverage(splits)

    print_repeated_templates(splits)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in SPLITS:
        output_path = (
            OUTPUT_DIR
            / f"{split}.jsonl"
        )

        save_jsonl(
            output_path,
            splits[split],
        )

        print(
            f"\nSaved {len(splits[split])} records → "
            f"{output_path}"
        )

    # ------------------------------------------------------------------
    # Save split metadata
    # ------------------------------------------------------------------

    metadata = {
        "seed": SEED,
        "strategy": "entity-aware + category-aware",
        "source": str(INPUT_PATH),
        "total_records": len(records),
        "unique_entities": len(groups),
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "counts": {
            split: len(splits[split])
            for split in SPLITS
        },
        "entity_counts": {
            split: len(assignments[split])
            for split in SPLITS
        },
        "entity_leakage": False,
    }

    metadata_path = (
        OUTPUT_DIR
        / "split_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved split metadata → {metadata_path}"
    )

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)

    print(
        "\nThe dataset is now split by ENTITY."
    )

    print(
        "The same entity cannot appear in train, validation, "
        "and test simultaneously."
    )

    print(
        "\nNext step: audit the resulting split before training."
    )


if __name__ == "__main__":
    main()
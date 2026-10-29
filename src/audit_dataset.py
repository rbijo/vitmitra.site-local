"""
Audit the VIT FAQ dataset for semantic/template leakage.

This script DOES NOT modify the dataset.

It analyzes:
    1. Basic dataset integrity
    2. Field availability
    3. uniqueness_basis usefulness
    4. Category distribution
    5. Source URL distribution
    6. Repeated question structures
    7. Question-template similarity
    8. Entity/property patterns
    9. Potential leakage groups
   10. Cross-template/category/source relationships

Input:
    data/raw/vit_cleaned_1000_qa.jsonl

Output:
    Printed audit report only.

Run:
    python src/audit_dataset.py

The purpose is to decide whether the final train/validation/test split
should be:
    A. record-level + category-aware, or
    B. entity/template-aware + category-aware.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vit_cleaned_1000_qa.jsonl"
)

# Number of examples shown for each repeated pattern.
PREVIEW_LIMIT = 10

# A normalized question pattern occurring at least this many times is
# considered interesting for leakage analysis.
MIN_TEMPLATE_FREQUENCY = 2

# Stopwords removed when constructing structural signatures.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "with",
}


# ============================================================================
# Loading
# ============================================================================

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Load a JSONL dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
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

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )

            records.append(record)

    return records


# ============================================================================
# General helpers
# ============================================================================

def clean_text(value: Any) -> str:
    """Convert a value to normalized whitespace-separated text."""

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )


def normalize_text(
    text: str,
) -> str:
    """
    Basic normalization.

    This intentionally preserves most words because we want to inspect
    structural similarities without destroying useful information.
    """

    text = clean_text(text).lower()

    # Normalize punctuation to spaces.
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return clean_text(text)


def tokenize_words(
    text: str,
) -> list[str]:
    """Tokenize normalized text into simple word tokens."""

    normalized = normalize_text(text)

    if not normalized:
        return []

    return normalized.split()


def print_section(
    title: str,
) -> None:
    """Print a report section heading."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(
    title: str,
) -> None:
    """Print a smaller report heading."""

    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


# ============================================================================
# Field inspection
# ============================================================================

def inspect_fields(
    records: list[dict[str, Any]],
) -> None:
    """Inspect available fields and their completeness."""

    print_section(
        "1. FIELD INVENTORY"
    )

    field_counts: Counter[str] = Counter()

    for record in records:
        field_counts.update(record.keys())

    print(
        f"Records: {len(records)}"
    )

    print(
        f"Unique fields: {len(field_counts)}"
    )

    print()

    print(
        f"{'Field':35}"
        f"{'Present':>10}"
        f"{'Missing':>10}"
        f"{'Unique':>10}"
    )

    print("-" * 65)

    for field in sorted(field_counts):

        present = 0
        values: list[str] = []

        for record in records:

            if (
                field in record
                and record[field] is not None
                and clean_text(record[field])
            ):
                present += 1
                values.append(
                    clean_text(record[field])
                )

        missing = len(records) - present

        unique = len(set(values))

        print(
            f"{field[:35]:35}"
            f"{present:>10}"
            f"{missing:>10}"
            f"{unique:>10}"
        )


# ============================================================================
# Basic integrity
# ============================================================================

def audit_integrity(
    records: list[dict[str, Any]],
) -> None:
    """Check basic dataset integrity."""

    print_section(
        "2. BASIC DATASET INTEGRITY"
    )

    ids = [
        clean_text(record.get("id"))
        for record in records
    ]

    questions = [
        normalize_text(
            record.get("question", "")
        )
        for record in records
    ]

    answers = [
        normalize_text(
            record.get("answer", "")
        )
        for record in records
    ]

    qa_pairs = list(
        zip(
            questions,
            answers,
        )
    )

    duplicate_ids = [
        value
        for value, count in Counter(ids).items()
        if value and count > 1
    ]

    duplicate_questions = [
        value
        for value, count in Counter(questions).items()
        if value and count > 1
    ]

    duplicate_pairs = [
        value
        for value, count in Counter(qa_pairs).items()
        if value[0] and count > 1
    ]

    empty_questions = sum(
        not value
        for value in questions
    )

    empty_answers = sum(
        not value
        for value in answers
    )

    print(
        f"Total records: {len(records)}"
    )

    print(
        f"Duplicate IDs: {len(duplicate_ids)}"
    )

    print(
        f"Duplicate normalized questions: "
        f"{len(duplicate_questions)}"
    )

    print(
        f"Duplicate normalized Q/A pairs: "
        f"{len(duplicate_pairs)}"
    )

    print(
        f"Empty questions: {empty_questions}"
    )

    print(
        f"Empty answers: {empty_answers}"
    )


# ============================================================================
# Uniqueness-basis analysis
# ============================================================================

def audit_uniqueness_basis(
    records: list[dict[str, Any]],
) -> None:
    """
    Determine whether uniqueness_basis is actually useful for grouping.

    This is the issue discovered by prepare_dataset.py.
    """

    print_section(
        "3. UNIQUENESS_BASIS GROUPING AUDIT"
    )

    values = [
        clean_text(
            record.get("uniqueness_basis")
        )
        for record in records
    ]

    non_empty = [
        value
        for value in values
        if value
    ]

    counts = Counter(
        non_empty
    )

    duplicate_groups = {
        value: count
        for value, count in counts.items()
        if count > 1
    }

    print(
        f"Records: {len(records)}"
    )

    print(
        f"Non-empty uniqueness_basis: "
        f"{len(non_empty)}"
    )

    print(
        f"Unique uniqueness_basis values: "
        f"{len(counts)}"
    )

    print(
        f"Repeated uniqueness_basis values: "
        f"{len(duplicate_groups)}"
    )

    if len(counts) == len(records):

        print(
            "\nRESULT: uniqueness_basis is unique per record."
        )

        print(
            "It currently provides NO meaningful "
            "grouping protection."
        )

    elif duplicate_groups:

        print(
            "\nRESULT: uniqueness_basis contains "
            "repeated groups."
        )

        print(
            "Potential grouping key: YES"
        )

    print_subsection(
        "Most frequent uniqueness_basis values"
    )

    for value, count in counts.most_common(
        PREVIEW_LIMIT
    ):

        print(
            f"{count:>4}  {value}"
        )


# ============================================================================
# Category analysis
# ============================================================================

def audit_categories(
    records: list[dict[str, Any]],
) -> None:
    """Analyze category distribution."""

    print_section(
        "4. CATEGORY DISTRIBUTION"
    )

    categories = Counter(
        clean_text(
            record.get("category")
        )
        for record in records
    )

    for category, count in categories.most_common():

        percentage = (
            count
            / len(records)
            * 100
        )

        print(
            f"{category:35}"
            f"{count:5}"
            f"  ({percentage:5.1f}%)"
        )


# ============================================================================
# Source analysis
# ============================================================================

def audit_sources(
    records: list[dict[str, Any]],
) -> None:
    """Analyze source URL distribution."""

    print_section(
        "5. SOURCE URL DISTRIBUTION"
    )

    source_values = [
        clean_text(
            record.get("source_url")
        )
        for record in records
    ]

    sources = Counter(
        value
        for value in source_values
        if value
    )

    print(
        f"Unique source URLs: {len(sources)}"
    )

    print_subsection(
        "Most frequent sources"
    )

    for source, count in sources.most_common(
        PREVIEW_LIMIT
    ):

        percentage = (
            count
            / len(records)
            * 100
        )

        print(
            f"{count:>4} "
            f"({percentage:5.1f}%) "
            f"{source}"
        )


# ============================================================================
# Question normalization
# ============================================================================

def normalize_question_for_template(
    question: str,
) -> str:
    """
    Create a coarse structural template.

    This intentionally replaces likely entity values with placeholders.

    Example:

        What admission-test rule applies to
        B.Tech - Biotechnology?

    may become approximately:

        what admission test rule applies to
        [ENTITY]

    The goal is not perfect NLP. The goal is to identify repeated
    question structures that could leak across splits.
    """

    text = clean_text(
        question
    ).lower()

    # Normalize punctuation.
    text = re.sub(
        r"[“”\"'`]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # ----------------------------------------------------------------
    # Common entity indicators.
    # ----------------------------------------------------------------

    # Degree/programme-like strings.
    text = re.sub(
        r"\b(?:b\.?\s?tech|btech|b\.?\s?sc|bsc|"
        r"m\.?\s?tech|mtech|m\.?\s?sc|msc|"
        r"mba|phd|mca|bca)\b"
        r"(?:\s*[-–—:]\s*)?"
        r"(?:[a-z0-9]+(?:\s+[a-z0-9]+){0,6})?",
        "[ENTITY]",
        text,
    )

    # ----------------------------------------------------------------
    # Quoted or parenthesized entity values.
    # ----------------------------------------------------------------

    text = re.sub(
        r"\([^)]{2,80}\)",
        "[ENTITY]",
        text,
    )

    # ----------------------------------------------------------------
    # Replace obvious year values.
    # ----------------------------------------------------------------

    text = re.sub(
        r"\b20\d{2}(?:[-/]\d{2,4})?\b",
        "[YEAR]",
        text,
    )

    # ----------------------------------------------------------------
    # Replace long hyphenated or slash-separated names.
    # ----------------------------------------------------------------

    text = re.sub(
        r"\b[a-z0-9]+(?:[-/][a-z0-9]+){1,}\b",
        "[ENTITY]",
        text,
    )

    # ----------------------------------------------------------------
    # Collapse whitespace.
    # ----------------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def build_question_templates(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build groups based on normalized question structure."""

    templates: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:

        question = clean_text(
            record.get("question")
        )

        template = normalize_question_for_template(
            question
        )

        templates[template].append(
            record
        )

    return dict(templates)


# ============================================================================
# Template audit
# ============================================================================

def audit_question_templates(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Find repeated question structures.
    """

    print_section(
        "6. QUESTION TEMPLATE AUDIT"
    )

    templates = build_question_templates(
        records
    )

    repeated_templates = {
        template: grouped_records
        for template, grouped_records
        in templates.items()
        if len(grouped_records)
        >= MIN_TEMPLATE_FREQUENCY
    }

    print(
        f"Unique normalized templates: "
        f"{len(templates)}"
    )

    print(
        f"Repeated templates: "
        f"{len(repeated_templates)}"
    )

    repeated_record_count = sum(
        len(grouped_records)
        for grouped_records
        in repeated_templates.values()
    )

    print(
        f"Records belonging to repeated templates: "
        f"{repeated_record_count}"
    )

    if repeated_templates:

        print_subsection(
            "Most frequent question templates"
        )

        ranked = sorted(
            repeated_templates.items(),
            key=lambda item: len(
                item[1]
            ),
            reverse=True,
        )

        for template, grouped_records in ranked[
            :PREVIEW_LIMIT
        ]:

            print(
                f"\nCOUNT: {len(grouped_records)}"
            )

            print(
                f"TEMPLATE: {template}"
            )

            print("Examples:")

            for record in grouped_records[
                :3
            ]:

                print(
                    f"  [{record.get('id')}] "
                    f"{clean_text(record.get('question'))}"
                )

                print(
                    f"      category="
                    f"{clean_text(record.get('category'))}"
                )

    return templates


# ============================================================================
# Structural signatures
# ============================================================================

def create_question_signature(
    question: str,
) -> str:
    """
    Create a lightweight structural signature.

    Content words are preserved while common function words are removed.

    This helps identify questions that differ only by entities.
    """

    words = tokenize_words(
        question
    )

    filtered = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return " ".join(
        filtered
    )


def audit_question_signatures(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Analyze repeated structural question signatures."""

    print_section(
        "7. QUESTION STRUCTURAL SIGNATURE AUDIT"
    )

    signatures: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:

        question = clean_text(
            record.get("question")
        )

        signature = create_question_signature(
            question
        )

        signatures[signature].append(
            record
        )

    repeated = {
        signature: grouped_records
        for signature, grouped_records
        in signatures.items()
        if len(grouped_records) >= 2
    }

    print(
        f"Unique signatures: "
        f"{len(signatures)}"
    )

    print(
        f"Repeated signatures: "
        f"{len(repeated)}"
    )

    repeated_record_count = sum(
        len(grouped_records)
        for grouped_records
        in repeated.values()
    )

    print(
        f"Records in repeated signatures: "
        f"{repeated_record_count}"
    )

    print_subsection(
        "Most frequent signatures"
    )

    ranked = sorted(
        repeated.items(),
        key=lambda item: len(
            item[1]
        ),
        reverse=True,
    )

    for signature, grouped_records in ranked[
        :PREVIEW_LIMIT
    ]:

        print(
            f"\nCOUNT: {len(grouped_records)}"
        )

        print(
            f"SIGNATURE: {signature}"
        )

        for record in grouped_records[
            :3
        ]:

            print(
                f"  [{record.get('id')}] "
                f"{clean_text(record.get('question'))}"
            )

    return signatures


# ============================================================================
# Entity field inspection
# ============================================================================

def audit_possible_entity_fields(
    records: list[dict[str, Any]],
) -> None:
    """
    Look for fields that might provide better grouping candidates.

    We do not assume a field exists.
    """

    print_section(
        "8. POSSIBLE ENTITY / GROUPING FIELDS"
    )

    candidate_names = [
        "entity",
        "entity_name",
        "entity_group",
        "topic",
        "subtopic",
        "subject",
        "programme",
        "program",
        "course",
        "department",
        "section",
        "topic_group",
        "group",
        "uniqueness_basis",
    ]

    available_fields = set()

    for record in records:
        available_fields.update(
            record.keys()
        )

    found_candidates = [
        field
        for field in candidate_names
        if field in available_fields
    ]

    if not found_candidates:

        print(
            "No obvious entity/grouping field names "
            "were found."
        )

        print(
            "A grouping key may need to be derived "
            "from the question text."
        )

        return

    for field in found_candidates:

        values = [
            clean_text(
                record.get(field)
            )
            for record in records
        ]

        values = [
            value
            for value in values
            if value
        ]

        counts = Counter(
            values
        )

        repeated = sum(
            1
            for count in counts.values()
            if count > 1
        )

        print_subsection(
            field
        )

        print(
            f"Non-empty: {len(values)}"
        )

        print(
            f"Unique: {len(counts)}"
        )

        print(
            f"Repeated values: {repeated}"
        )

        print(
            "Top values:"
        )

        for value, count in counts.most_common(
            5
        ):

            print(
                f"  {count:>4}  {value}"
            )


# ============================================================================
# Category × template relationship
# ============================================================================

def audit_template_category_relationship(
    templates: dict[str, list[dict[str, Any]]],
) -> None:
    """
    Determine whether repeated question templates stay within a category
    or cross category boundaries.
    """

    print_section(
        "9. TEMPLATE × CATEGORY RELATIONSHIP"
    )

    repeated_templates = {
        template: grouped_records
        for template, grouped_records
        in templates.items()
        if len(grouped_records) >= 2
    }

    cross_category = []

    for template, grouped_records in (
        repeated_templates.items()
    ):

        categories = {
            clean_text(
                record.get("category")
            )
            for record in grouped_records
        }

        if len(categories) > 1:

            cross_category.append(
                (
                    template,
                    categories,
                    grouped_records,
                )
            )

    print(
        f"Repeated templates: "
        f"{len(repeated_templates)}"
    )

    print(
        f"Templates spanning multiple categories: "
        f"{len(cross_category)}"
    )

    if cross_category:

        print_subsection(
            "Cross-category templates"
        )

        for (
            template,
            categories,
            grouped_records,
        ) in cross_category[
            :PREVIEW_LIMIT
        ]:

            print(
                f"\nTEMPLATE: {template}"
            )

            print(
                f"CATEGORIES: "
                f"{sorted(categories)}"
            )

            for record in grouped_records[
                :3
            ]:

                print(
                    f"  [{record.get('id')}] "
                    f"{clean_text(record.get('question'))}"
                )


# ============================================================================
# Category × source relationship
# ============================================================================

def audit_category_source_relationship(
    records: list[dict[str, Any]],
) -> None:
    """
    Analyze whether categories are strongly tied to particular sources.

    This is useful because a source-based split could accidentally create
    another kind of leakage or distribution shift.
    """

    print_section(
        "10. CATEGORY × SOURCE RELATIONSHIP"
    )

    mapping: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    for record in records:

        category = clean_text(
            record.get("category")
        )

        source = clean_text(
            record.get("source_url")
        )

        if category and source:
            mapping[category][source] += 1

    for category in sorted(
        mapping
    ):

        sources = mapping[category]

        if not sources:
            continue

        top_source, top_count = (
            sources.most_common(1)[0]
        )

        total = sum(
            sources.values()
        )

        concentration = (
            top_count
            / total
            * 100
        )

        if concentration >= 70:

            print(
                f"{category:35}"
                f"top source={concentration:5.1f}% "
                f"({top_count}/{total})"
            )


# ============================================================================
# Potential leakage group report
# ============================================================================

def audit_potential_leakage_groups(
    templates: dict[str, list[dict[str, Any]]],
) -> None:
    """
    Produce the main decision-oriented leakage report.

    A template is especially interesting when:
        - it occurs multiple times
        - it contains different entities
        - it is large enough to influence the test set
    """

    print_section(
        "11. POTENTIAL TEMPLATE LEAKAGE GROUPS"
    )

    repeated = {
        template: grouped_records
        for template, grouped_records
        in templates.items()
        if len(grouped_records) >= 2
    }

    if not repeated:

        print(
            "No repeated normalized question templates found."
        )

        print(
            "Record-level splitting is likely acceptable "
            "from a template-leakage perspective."
        )

        return

    ranked = sorted(
        repeated.items(),
        key=lambda item: len(
            item[1]
        ),
        reverse=True,
    )

    high_frequency = [
        (
            template,
            grouped_records,
        )
        for template, grouped_records
        in ranked
        if len(grouped_records) >= 5
    ]

    print(
        f"Repeated template groups: "
        f"{len(repeated)}"
    )

    print(
        f"Templates occurring >=5 times: "
        f"{len(high_frequency)}"
    )

    print_subsection(
        "Largest potential leakage groups"
    )

    for template, grouped_records in ranked[
        :20
    ]:

        categories = Counter(
            clean_text(
                record.get("category")
            )
            for record in grouped_records
        )

        print(
            f"\nCOUNT: {len(grouped_records)}"
        )

        print(
            f"TEMPLATE: {template}"
        )

        print(
            "CATEGORIES: "
            + ", ".join(
                f"{category}={count}"
                for category, count
                in categories.most_common()
            )
        )

        for record in grouped_records[
            :3
        ]:

            print(
                f"  [{record.get('id')}] "
                f"{clean_text(record.get('question'))}"
            )


# ============================================================================
# Recommendation
# ============================================================================

def print_recommendation(
    records: list[dict[str, Any]],
    templates: dict[str, list[dict[str, Any]]],
) -> None:
    """
    Produce a conservative recommendation.

    This is a heuristic report, not a claim that the model will leak.
    """

    print_section(
        "12. AUDIT RECOMMENDATION"
    )

    unique_basis = len(
        {
            clean_text(
                record.get("uniqueness_basis")
            )
            for record in records
            if clean_text(
                record.get("uniqueness_basis")
            )
        }
    )

    repeated_templates = {
        template: grouped_records
        for template, grouped_records
        in templates.items()
        if len(grouped_records) >= 2
    }

    large_templates = {
        template: grouped_records
        for template, grouped_records
        in repeated_templates.items()
        if len(grouped_records) >= 5
    }

    print(
        f"Records: {len(records)}"
    )

    print(
        f"Unique uniqueness_basis values: "
        f"{unique_basis}"
    )

    print(
        f"Repeated question templates: "
        f"{len(repeated_templates)}"
    )

    print(
        f"Large templates (>=5 records): "
        f"{len(large_templates)}"
    )

    print()

    if (
        unique_basis == len(records)
        and large_templates
    ):

        print(
            "RECOMMENDATION:"
        )

        print(
            "Use TEMPLATE-AWARE + CATEGORY-AWARE splitting."
        )

        print(
            "Reason:"
        )

        print(
            "  uniqueness_basis does not provide grouping,"
        )

        print(
            "  but repeated question structures exist."
        )

        print(
            "The final splitter should derive a meaningful "
            "template/entity group from question structure."
        )

    elif (
        unique_basis == len(records)
        and not large_templates
    ):

        print(
            "RECOMMENDATION:"
        )

        print(
            "The current CATEGORY-AWARE record-level split "
            "is probably sufficient."
        )

        print(
            "No strong repeated question-template groups "
            "were detected."
        )

    else:

        print(
            "RECOMMENDATION:"
        )

        print(
            "Inspect the repeated uniqueness_basis groups "
            "before finalizing the split."
        )

        print(
            "If those groups represent real entities/topics, "
            "use them for group-aware splitting."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This audit does NOT modify train/validation/test."
    )

    print(
        "Do not start model training until the split strategy "
        "has been finalized."
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Run the complete dataset audit."""

    print("=" * 80)
    print("VIT FAQ DATASET — SEMANTIC / TEMPLATE LEAKAGE AUDIT")
    print("=" * 80)

    print(
        f"\nInput file:\n{INPUT_FILE}"
    )

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    records = load_jsonl(
        INPUT_FILE
    )

    print(
        f"\nLoaded {len(records)} records."
    )

    # ------------------------------------------------------------------------
    # Audits
    # ------------------------------------------------------------------------

    inspect_fields(
        records
    )

    audit_integrity(
        records
    )

    audit_uniqueness_basis(
        records
    )

    audit_categories(
        records
    )

    audit_sources(
        records
    )

    templates = audit_question_templates(
        records
    )

    audit_question_signatures(
        records
    )

    audit_possible_entity_fields(
        records
    )

    audit_template_category_relationship(
        templates
    )

    audit_category_source_relationship(
        records
    )

    audit_potential_leakage_groups(
        templates
    )

    print_recommendation(
        records,
        templates,
    )

    # ------------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print(
        "\nNo files were modified."
    )

    print(
        "Send the complete output before changing "
        "prepare_dataset.py."
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from faq_retriever import build_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "vit_cleaned_1000_qa.jsonl"
TEST_DATASET = PROJECT_ROOT / "data" / "processed" / "test.jsonl"

OUTPUT_REPORT = PROJECT_ROOT / "evaluation" / "retrieval_evaluation_report.json"
OUTPUT_PREDICTIONS = PROJECT_ROOT / "evaluation" / "retrieval_predictions.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main() -> None:

    print()
    print("=" * 70)
    print("VIT FAQ RETRIEVAL EVALUATION")
    print("=" * 70)
    print()
    print("Protocol:")
    print("  Full FAQ knowledge base")
    print("  Test questions evaluated against complete KB")
    print("  Exact query record excluded for non-trivial matching")
    print()

    # --------------------------------------------------------------
    # Complete production knowledge base
    # --------------------------------------------------------------
    kb_records = load_jsonl(RAW_DATASET)

    # --------------------------------------------------------------
    # Held-out test questions
    # --------------------------------------------------------------
    test_records = load_jsonl(TEST_DATASET)

    print(f"Knowledge base records: {len(kb_records)}")
    print(f"Test questions:         {len(test_records)}")
    print()

    retriever = build_retriever(
        RAW_DATASET,
        threshold=0.82,
        margin=0.08,
    )

    predictions = []

    correct = 0
    wrong = 0
    abstained = 0

    reason_counts = Counter()
    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "wrong": 0,
            "abstained": 0,
        }
    )

    for test_record in test_records:

        query = test_record["question"]
        expected_id = test_record["id"]
        expected_answer = test_record["answer"]
        category = test_record.get("category", "")

        result = retriever.retrieve(
            query,
            exclude_id=expected_id,
        )

        category_stats[category]["total"] += 1
        reason_counts[result.reason] += 1

        retrieved_id = result.record.id if result.record else None
        retrieved_answer = result.record.answer if result.record else None

        is_correct = False
        is_wrong = False

        if result.matched and result.record:

            if retrieved_id == expected_id:
                is_correct = True
                correct += 1
                category_stats[category]["correct"] += 1
            else:
                is_wrong = True
                wrong += 1
                category_stats[category]["wrong"] += 1

        else:
            abstained += 1
            category_stats[category]["abstained"] += 1

        predictions.append(
            {
                "id": expected_id,
                "category": category,
                "query": query,
                "expected_answer": expected_answer,
                "retrieved_id": retrieved_id,
                "retrieved_answer": retrieved_answer,
                "score": result.score,
                "reason": result.reason,
                "correct": is_correct,
                "wrong": is_wrong,
            }
        )

    total = len(test_records)

    coverage = (correct + wrong) / total if total else 0.0
    correct_rate = correct / total if total else 0.0
    wrong_rate = wrong / total if total else 0.0
    abstention_rate = abstained / total if total else 0.0

    answered = correct + wrong
    precision = correct / answered if answered else 0.0

    category_report = {}

    for category, stats in sorted(category_stats.items()):

        answered_category = stats["correct"] + stats["wrong"]

        category_report[category] = {
            **stats,
            "coverage": (
                answered_category / stats["total"]
                if stats["total"]
                else 0.0
            ),
            "precision_when_answering": (
                stats["correct"] / answered_category
                if answered_category
                else 0.0
            ),
            "wrong_rate": (
                stats["wrong"] / stats["total"]
                if stats["total"]
                else 0.0
            ),
            "abstention_rate": (
                stats["abstained"] / stats["total"]
                if stats["total"]
                else 0.0
            ),
        }

    wrong_examples = [
        item for item in predictions
        if item["wrong"]
    ][:20]

    report = {
        "evaluation_protocol": {
            "knowledge_base": str(RAW_DATASET),
            "test_dataset": str(TEST_DATASET),
            "evaluation_type": "full_kb_test",
            "knowledge_base_records": len(kb_records),
            "test_examples": len(test_records),
            "exact_query_record_excluded": True,
            "retrieval_method": (
                "exact question -> token overlap -> "
                "confidence threshold -> margin"
            ),
            "threshold": 0.82,
            "margin": 0.08,
        },
        "metrics": {
            "test_examples": total,
            "retrieval_coverage": coverage,
            "correct_retrieval_rate": correct_rate,
            "wrong_retrieval_rate": wrong_rate,
            "abstention_rate": abstention_rate,
            "precision_when_answering": precision,
        },
        "counts": {
            "correct": correct,
            "wrong": wrong,
            "abstained": abstained,
        },
        "reasons": dict(reason_counts),
        "by_category": category_report,
        "wrong_retrieval_examples": wrong_examples,
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with OUTPUT_PREDICTIONS.open("w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("Results:")
    print(f"  Coverage:                 {coverage:.2%}")
    print(f"  Correct retrieval rate:   {correct_rate:.2%}")
    print(f"  Wrong retrieval rate:     {wrong_rate:.2%}")
    print(f"  Abstention rate:          {abstention_rate:.2%}")
    print(f"  Precision when answering: {precision:.2%}")
    print()

    print("Counts:")
    print(f"  Correct:    {correct}")
    print(f"  Wrong:      {wrong}")
    print(f"  Abstained:  {abstained}")
    print()

    print("Reasons:")

    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")

    print()

    print("Category performance:")

    for category, stats in category_report.items():
        print(
            f"  {category:<25} "
            f"coverage={stats['coverage']:.1%} "
            f"precision={stats['precision_when_answering']:.1%} "
            f"wrong={stats['wrong']}"
        )

    print()
    print(f"Saved report: {OUTPUT_REPORT}")
    print(f"Saved predictions: {OUTPUT_PREDICTIONS}")
    print()


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def normalize_text(text: str) -> str:
    """Normalize text for robust matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> set[str]:
    return set(normalize_text(text).split())


def token_overlap(query: str, candidate: str) -> float:
    """Jaccard token similarity."""
    q = tokenize(query)
    c = tokenize(candidate)

    if not q or not c:
        return 0.0

    return len(q & c) / len(q | c)


@dataclass
class FAQRecord:
    id: str
    question: str
    answer: str
    category: str
    source_url: str
    uniqueness_basis: str


@dataclass
class RetrievalResult:
    matched: bool
    record: Optional[FAQRecord]
    score: float
    reason: str


class FAQRetriever:
    """
    Conservative FAQ retriever.

    Design goal:
        Return a verified FAQ answer only when confidence is sufficiently high.
        Otherwise abstain.

    This is intentionally lightweight because the knowledge base contains
    only ~1,000 FAQ records.
    """

    def __init__(
        self,
        records: list[FAQRecord],
        threshold: float = 0.82,
        margin: float = 0.08,
    ):
        self.records = records
        self.threshold = threshold
        self.margin = margin

        self.normalized_questions = {
            record.id: normalize_text(record.question)
            for record in records
        }

    def _exact_match(self, query: str) -> Optional[FAQRecord]:
        normalized = normalize_text(query)

        for record in self.records:
            if self.normalized_questions[record.id] == normalized:
                return record

        return None

    def _score(self, query: str, record: FAQRecord) -> float:
        """
        Score a candidate using token overlap.

        We intentionally keep this simple and deterministic.
        """
        return token_overlap(query, record.question)

    def retrieve(
        self,
        query: str,
        exclude_id: Optional[str] = None,
    ) -> RetrievalResult:

        query = query.strip()

        if not query:
            return RetrievalResult(
                matched=False,
                record=None,
                score=0.0,
                reason="empty_query",
            )

        # --------------------------------------------------------------
        # 1. Exact FAQ match
        # --------------------------------------------------------------
        exact = self._exact_match(query)

        if exact is not None and exact.id != exclude_id:
            return RetrievalResult(
                matched=True,
                record=exact,
                score=1.0,
                reason="exact_question_match",
            )

        # --------------------------------------------------------------
        # 2. Fuzzy token matching
        # --------------------------------------------------------------
        candidates = []

        for record in self.records:
            if record.id == exclude_id:
                continue

            score = self._score(query, record)

            candidates.append((score, record))

        if not candidates:
            return RetrievalResult(
                matched=False,
                record=None,
                score=0.0,
                reason="no_candidates",
            )

        candidates.sort(key=lambda x: x[0], reverse=True)

        best_score, best_record = candidates[0]

        second_score = candidates[1][0] if len(candidates) > 1 else 0.0

        # --------------------------------------------------------------
        # 3. Confidence checks
        # --------------------------------------------------------------
        if best_score < self.threshold:
            return RetrievalResult(
                matched=False,
                record=None,
                score=best_score,
                reason="score_below_threshold",
            )

        if (best_score - second_score) < self.margin:
            return RetrievalResult(
                matched=False,
                record=None,
                score=best_score,
                reason="ambiguous_match",
            )

        # --------------------------------------------------------------
        # 4. High-confidence retrieval
        # --------------------------------------------------------------
        return RetrievalResult(
            matched=True,
            record=best_record,
            score=best_score,
            reason="high_confidence_fuzzy_match",
        )


def load_jsonl(path: str | Path) -> list[FAQRecord]:
    path = Path(path)

    records: list[FAQRecord] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            data = json.loads(line)

            records.append(
                FAQRecord(
                    id=data["id"],
                    question=data["question"],
                    answer=data["answer"],
                    category=data.get("category", ""),
                    source_url=data.get("source_url", ""),
                    uniqueness_basis=data.get("uniqueness_basis", ""),
                )
            )

    return records


def build_retriever(
    dataset_path: str | Path,
    threshold: float = 0.82,
    margin: float = 0.08,
) -> FAQRetriever:

    records = load_jsonl(dataset_path)

    return FAQRetriever(
        records=records,
        threshold=threshold,
        margin=margin,
    )
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sft"


SPLITS = {
    "train": INPUT_DIR / "train.jsonl",
    "validation": INPUT_DIR / "validation.jsonl",
    "test": INPUT_DIR / "test.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc

            records.append(record)

    return records


def convert_record(record: dict) -> dict:
    question = record["question"].strip()
    answer = record["answer"].strip()

    if not question:
        raise ValueError(f"Empty question in record {record.get('id')}")

    if not answer:
        raise ValueError(f"Empty answer in record {record.get('id')}")

    return {
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": answer,
            }
        ],
    }


def save_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    print("=" * 80)
    print("VIT FAQ SFT DATASET PREPARATION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, input_path in SPLITS.items():
        print(f"\nProcessing {split_name}:")
        print(f"  Input : {input_path}")

        records = load_jsonl(input_path)

        converted = [
            convert_record(record)
            for record in records
        ]

        output_path = OUTPUT_DIR / f"{split_name}.jsonl"
        save_jsonl(converted, output_path)

        print(f"  Records: {len(converted)}")
        print(f"  Output : {output_path}")

    print("\n" + "=" * 80)
    print("SFT DATASET PREPARATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
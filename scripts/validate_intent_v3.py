"""Validate the complete v3 intent dataset without training a model.

Run: python -m scripts.validate_intent_v3
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from mind3.intent_router import INTENTS

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "curated"
SPLITS = {
    "train": (CURATED / "intent_train_v3.jsonl", 320, 20),
    "dev": (CURATED / "intent_dev_v3.jsonl", 80, 5),
    "test": (CURATED / "intent_test_v3.jsonl", 80, 5),
}


def read(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    all_ids, all_texts = set(), set()
    summary = {}
    for split, (path, expected_rows, expected_each_language) in SPLITS.items():
        rows = read(path)
        if len(rows) != expected_rows:
            raise ValueError(f"{split}: expected {expected_rows} rows, found {len(rows)}")
        class_counts, language_counts = Counter(), Counter()
        for row in rows:
            required = {
                "id", "text", "intent", "language", "supervision",
                "authoring_method", "reviewer", "review_status",
            }
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{split}: {row.get('id')} missing {sorted(missing)}")
            if row["intent"] not in INTENTS or row["language"] not in {"ar", "en"}:
                raise ValueError(f"{split}: invalid label/language in {row['id']}")
            if row["review_status"] != "assistant_reviewed":
                raise ValueError(f"{split}: row is not assistant-reviewed: {row['id']}")
            if row["reviewer"] != "assistant_semantic_review":
                raise ValueError(f"{split}: reviewer provenance missing: {row['id']}")
            if not isinstance(row["text"], str) or len(row["text"].strip()) < 20:
                raise ValueError(f"{split}: empty/short prompt: {row['id']}")
            normalized = " ".join(row["text"].casefold().split())
            if row["id"] in all_ids:
                raise ValueError(f"Duplicate ID across splits: {row['id']}")
            if normalized in all_texts:
                raise ValueError(f"Exact text leakage across splits: {row['id']}")
            all_ids.add(row["id"])
            all_texts.add(normalized)
            class_counts[row["intent"]] += 1
            language_counts[(row["intent"], row["language"])] += 1
        if set(class_counts) != set(INTENTS) or len(set(class_counts.values())) != 1:
            raise ValueError(f"{split}: intent classes are missing or unbalanced")
        for label in INTENTS:
            for language in ("ar", "en"):
                if language_counts[(label, language)] != expected_each_language:
                    raise ValueError(
                        f"{split}: expected {expected_each_language} {language} rows "
                        f"for {label}, found {language_counts[(label, language)]}"
                    )
        summary[split] = {
            "rows": len(rows),
            "per_intent": dict(class_counts),
            "arabic": sum(v for (label, lang), v in language_counts.items() if lang == "ar"),
            "english": sum(v for (label, lang), v in language_counts.items() if lang == "en"),
        }

    boundary = read(CURATED / "intent_boundary_review_v3.jsonl")
    if len(boundary) != 16 or any(
        row.get("review_status") != "approved_by_user" for row in boundary
    ):
        raise ValueError("Boundary review set must remain a separate 16-row approved suite")
    boundary_ids = {row["id"] for row in boundary}
    if boundary_ids & all_ids:
        raise ValueError("Boundary-review rows were merged into a v3 data split")

    print(json.dumps({
        "status": "PASS",
        "total_train_dev_test_rows": len(all_ids),
        "boundary_suite_rows_separate": len(boundary),
        "exact_duplicate_or_split_leakage": False,
        "splits": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

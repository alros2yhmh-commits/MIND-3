"""Evaluate a trained MIND-3 v3 router once on the sealed final test split.

Run only after training and freezing the model/settings:
python -m scripts.evaluate_router_v3
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from mind3.intent_router import INTENTS, IntentRouter

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "curated" / "intent_test_v3.jsonl"
MODEL_PATH = ROOT / "artifacts" / "intent_v3" / "intent_router_v3.npz"
REPORT_PATH = ROOT / "artifacts" / "intent_v3" / "intent_router_v3_test_report.json"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the v3 router before evaluating the final test split.")
    with TEST_PATH.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 64:
        raise ValueError(f"Expected 64 final-test examples, found {len(rows)}")
    ids, texts = set(), set()
    counts, language_counts = Counter(), Counter()
    for row in rows:
        if row.get("intent") not in INTENTS or row.get("language") not in {"ar", "en"}:
            raise ValueError(f"Invalid test label or language in {row.get('id')}")
        key = " ".join(row["text"].casefold().split())
        if row["id"] in ids or key in texts:
            raise ValueError("Duplicate test ID or text")
        ids.add(row["id"])
        texts.add(key)
        counts[row["intent"]] += 1
        language_counts[(row["intent"], row["language"])] += 1
    if set(counts) != set(INTENTS) or len(set(counts.values())) != 1:
        raise ValueError("Final-test split is missing classes or is unbalanced")
    if any(not language_counts[(label, lang)] for label in INTENTS for lang in ("ar", "en")):
        raise ValueError("Final-test split is missing a language/class pair")

    router = IntentRouter()
    router.load(str(MODEL_PATH))
    confusion = {label: {target: 0 for target in INTENTS} for label in INTENTS}
    per_class = {}
    per_language = {}
    predictions = []
    for row in rows:
        result = router.predict(row["text"])
        pred = result["intent"]
        predictions.append((row, pred))
        confusion[row["intent"]][pred] += 1
    for label in INTENTS:
        subset = [(row, pred) for row, pred in predictions if row["intent"] == label]
        correct = sum(pred == row["intent"] for row, pred in subset)
        per_class[label] = {"examples": len(subset), "correct": correct,
                            "accuracy": correct / len(subset)}
    for language in ("ar", "en"):
        subset = [(row, pred) for row, pred in predictions if row["language"] == language]
        correct = sum(pred == row["intent"] for row, pred in subset)
        per_language[language] = {"examples": len(subset), "correct": correct,
                                  "accuracy": correct / len(subset)}
    report = {
        "dataset_version": "intent-data-v3",
        "test_examples": len(rows),
        "test_class_counts": dict(counts),
        "test_language_counts": {f"{k[0]}:{k[1]}": v for k, v in language_counts.items()},
        "accuracy": sum(pred == row["intent"] for row, pred in predictions) / len(rows),
        "per_class": per_class,
        "per_language": per_language,
        "confusion_matrix": confusion,
        "warning": "This is a one-time final-test diagnostic. Do not tune model settings on these results.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

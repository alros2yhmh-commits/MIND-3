"""Train the MIND-3 v3 intent router using train_v3 and report dev metrics only.

The final test split is intentionally not opened by this script.
Run: python -m scripts.train_router_v3
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from mind3.intent_router import INTENTS, IntentRouter

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "curated"
TRAIN_PATH = CURATED / "intent_train_v3.jsonl"
DEV_PATH = CURATED / "intent_dev_v3.jsonl"
ARTIFACTS = ROOT / "artifacts" / "intent_v3"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def validate(rows: list[dict], name: str):
    required = {
        "id", "text", "intent", "language", "supervision",
        "authoring_method", "reviewer", "review_status",
    }
    ids, texts = set(), set()
    labels, languages = Counter(), Counter()
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{name}: missing fields {sorted(missing)}")
        if row["intent"] not in INTENTS:
            raise ValueError(f"{name}: unknown intent {row['intent']}")
        if row["language"] not in {"ar", "en"}:
            raise ValueError(f"{name}: unsupported language {row['language']}")
        if row["review_status"] != "assistant_reviewed":
            raise ValueError(f"{name}: unreviewed row {row['id']}")
        if row["reviewer"] != "assistant_semantic_review":
            raise ValueError(f"{name}: missing reviewer provenance for {row['id']}")
        text = row["text"]
        if not isinstance(text, str) or len(text.strip()) < 20:
            raise ValueError(f"{name}: empty or too-short prompt {row['id']}")
        key = normalize(text)
        if row["id"] in ids or key in texts:
            raise ValueError(f"{name}: duplicate ID or normalized text")
        ids.add(row["id"])
        texts.add(key)
        labels[row["intent"]] += 1
        languages[(row["intent"], row["language"])] += 1
    if set(labels) != set(INTENTS):
        raise ValueError(f"{name}: missing intent classes")
    if len(set(labels.values())) != 1:
        raise ValueError(f"{name}: class imbalance: {dict(labels)}")
    for label in INTENTS:
        if not languages[(label, "ar")] or not languages[(label, "en")]:
            raise ValueError(f"{name}: {label} missing a language")
    return labels, languages, texts


def main():
    train_rows = read_rows(TRAIN_PATH)
    dev_rows = read_rows(DEV_PATH)
    train_counts, train_lang, train_texts = validate(train_rows, "train")
    dev_counts, dev_lang, dev_texts = validate(dev_rows, "dev")
    overlap = train_texts & dev_texts
    if overlap:
        raise ValueError(f"Train/dev exact text leakage: {sorted(overlap)[:3]}")
    if len(train_rows) != 320 or len(dev_rows) != 80:
        raise ValueError("Unexpected v3 split size; expected train=320 and dev=80")

    router = IntentRouter(seed=17)
    training = router.train([(r["text"], r["intent"]) for r in train_rows],
                            epochs=32, learning_rate=0.06, seed=23)
    predictions = [(row, router.predict(row["text"])["intent"]) for row in dev_rows]
    correct = sum(pred == row["intent"] for row, pred in predictions)
    per_class = {}
    per_language = {}
    confusion = {label: {target: 0 for target in INTENTS} for label in INTENTS}
    for row, pred in predictions:
        confusion[row["intent"]][pred] += 1
    for label in INTENTS:
        subset = [(row, pred) for row, pred in predictions if row["intent"] == label]
        per_class[label] = {
            "examples": len(subset),
            "correct": sum(pred == row["intent"] for row, pred in subset),
            "accuracy": sum(pred == row["intent"] for row, pred in subset) / len(subset),
        }
    for language in ("ar", "en"):
        subset = [(row, pred) for row, pred in predictions if row["language"] == language]
        per_language[language] = {
            "examples": len(subset),
            "correct": sum(pred == row["intent"] for row, pred in subset),
            "accuracy": sum(pred == row["intent"] for row, pred in subset) / len(subset),
        }

    report = {
        "dataset_version": "intent-data-v3",
        "training_source": "data/curated/intent_train_v3.jsonl",
        "development_source": "data/curated/intent_dev_v3.jsonl",
        "final_test_opened": False,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "train_class_counts": dict(train_counts),
        "dev_class_counts": dict(dev_counts),
        "train_language_counts": {f"{k[0]}:{k[1]}": v for k, v in train_lang.items()},
        "dev_language_counts": {f"{k[0]}:{k[1]}": v for k, v in dev_lang.items()},
        "train_sha256": digest(TRAIN_PATH),
        "dev_sha256": digest(DEV_PATH),
        "training": training,
        "dev_accuracy": correct / len(dev_rows),
        "dev_per_class": per_class,
        "dev_per_language": per_language,
        "confusion_matrix": confusion,
        "parameter_count": router.parameter_count,
        "limitations": [
            "The dev set is small and assistant-authored; it is a development diagnostic, not an independent benchmark.",
            "Do not tune repeatedly against the final test set.",
            "The router classifies intent only; it does not generate Python answers or prove Python knowledge.",
            "Softmax scores are uncalibrated."
        ],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    router.save(str(ARTIFACTS / "intent_router_v3.npz"))
    (ARTIFACTS / "intent_router_v3_dev_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

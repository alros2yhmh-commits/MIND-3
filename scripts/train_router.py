"""Train and evaluate the compact intent router on disjoint curated bilingual data."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import hashlib

from mind3.intent_router import IntentRouter, INTENTS

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "curated"
TRAIN_PATH = CURATED / "intent_train.jsonl"
EVAL_PATH = CURATED / "intent_eval.jsonl"
ARTIFACTS = ROOT / "artifacts" / "analytical"


def read_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalised(text: str) -> str:
    return " ".join(text.casefold().split())


def validate(rows, split_name):
    ids, texts = set(), set()
    counts, languages = Counter(), Counter()
    for row in rows:
        required = {"id", "text", "intent", "language", "supervision"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{split_name}: row missing fields {sorted(missing)}")
        if row["intent"] not in INTENTS:
            raise ValueError(f"{split_name}: unknown intent {row['intent']}")
        if row["language"] not in {"ar", "en"}:
            raise ValueError(f"{split_name}: unsupported language {row['language']}")
        if row["id"] in ids or normalised(row["text"]) in texts:
            raise ValueError(f"{split_name}: duplicate id or text")
        if not row["text"].strip():
            raise ValueError(f"{split_name}: empty text")
        ids.add(row["id"])
        texts.add(normalised(row["text"]))
        counts[row["intent"]] += 1
        languages[(row["intent"], row["language"])] += 1
    missing_labels = set(INTENTS) - set(counts)
    if missing_labels:
        raise ValueError(f"{split_name}: missing labels {sorted(missing_labels)}")
    return counts, languages, texts


def main():
    train_rows, eval_rows = read_rows(TRAIN_PATH), read_rows(EVAL_PATH)
    train_counts, train_lang, train_texts = validate(train_rows, "train")
    eval_counts, eval_lang, eval_texts = validate(eval_rows, "evaluation")
    overlap = train_texts & eval_texts
    if overlap:
        raise ValueError(f"Train/evaluation text leakage: {sorted(overlap)[:3]}")

    if len(set(train_counts.values())) != 1:
        raise ValueError(f"Training set is not class-balanced: {dict(train_counts)}")
    if len(set(eval_counts.values())) != 1:
        raise ValueError(f"Evaluation set is not class-balanced: {dict(eval_counts)}")
    for label in INTENTS:
        if train_lang[(label, "ar")] == 0 or train_lang[(label, "en")] == 0:
            raise ValueError(f"Training label {label} lacks Arabic or English examples")
        if eval_lang[(label, "ar")] == 0 or eval_lang[(label, "en")] == 0:
            raise ValueError(f"Evaluation label {label} lacks Arabic or English examples")

    router = IntentRouter()
    stats = router.train([(r["text"], r["intent"]) for r in train_rows])
    predictions = [(r, router.predict(r["text"])["intent"]) for r in eval_rows]
    correct = sum(pred == row["intent"] for row, pred in predictions)
    per_class = {}
    confusion = {label: {target: 0 for target in INTENTS} for label in INTENTS}
    for row, pred in predictions:
        confusion[row["intent"]][pred] += 1
    for label in INTENTS:
        class_rows = [(row, pred) for row, pred in predictions if row["intent"] == label]
        class_correct = sum(pred == row["intent"] for row, pred in class_rows)
        per_class[label] = {
            "examples": len(class_rows),
            "correct": class_correct,
            "accuracy": class_correct / len(class_rows) if class_rows else None,
        }

    report = {
        "dataset_version": "intent-data-v2",
        "training_source": "curated_authored_v2_only",
        "evaluation_source": "curated_authored_v2_holdout_only",
        "weak_huggingface_labels_used": False,
        "train_examples": len(train_rows),
        "evaluation_examples": len(eval_rows),
        "train_class_counts": dict(train_counts),
        "evaluation_class_counts": dict(eval_counts),
        "train_language_counts": {f"{k[0]}:{k[1]}": v for k, v in train_lang.items()},
        "evaluation_language_counts": {f"{k[0]}:{k[1]}": v for k, v in eval_lang.items()},
        "evaluation_accuracy": correct / len(eval_rows),
        "evaluation_per_class": per_class,
        "confusion_matrix": confusion,
        "parameter_count": router.parameter_count,
        "limitations": [
            "This small hand-authored holdout is an initial diagnostic, not a statistically strong benchmark.",
            "Scores measure intent classification only, not Python knowledge, code correctness, or reasoning.",
            "The softmax score remains uncalibrated.",
            "Review errors and expand the holdout before making generalisation claims."
        ],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    router.save(str(ARTIFACTS / "intent_router.npz"))
    (ARTIFACTS / "intent_router_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

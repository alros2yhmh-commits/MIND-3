"""Train the small intent router from prepared HF rows plus curated bilingual seeds."""
from __future__ import annotations

import json
from pathlib import Path
import random
from collections import defaultdict

from mind3.intent_router import IntentRouter

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SEEDS = ROOT / "data" / "curated" / "intent_seed.jsonl"
ARTIFACTS = ROOT / "artifacts" / "analytical"
INTENTS = {"explain_concept", "write_function", "debug_code", "trace_code",
           "compare_concepts", "translate_explain", "calculate", "general_python"}


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    rows = read_jsonl(PROCESSED / "python_train.jsonl")
    examples = [(r["prompt"], r["intent"]) for r in rows
                if r.get("intent") in INTENTS]
    examples.extend((r["text"], r["intent"]) for r in read_jsonl(SEEDS))
    if not examples:
        raise SystemExit("No training data. Run: python -m scripts.prepare_hf_data")

    # Balance weakly labelled HF rows so the majority fallback class does not
    # dominate the small router. Split before training for a real held-out check.
    buckets = defaultdict(list)
    for item in examples:
        buckets[item[1]].append(item)
    rng = random.Random(31)
    balanced = []
    for label, items in sorted(buckets.items()):
        rng.shuffle(items)
        balanced.extend(items[:1500])
    rng.shuffle(balanced)
    cut = max(1, int(len(balanced) * 0.8))
    train_rows, validation_rows = balanced[:cut], balanced[cut:]

    router = IntentRouter()
    stats = router.train(train_rows)
    correct = sum(router.predict(text)["intent"] == label
                  for text, label in validation_rows)
    stats["validation_examples"] = len(validation_rows)
    stats["validation_accuracy"] = correct / len(validation_rows) if validation_rows else None
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    router.save(str(ARTIFACTS / "intent_router.npz"))
    (ARTIFACTS / "intent_router_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Saved:", ARTIFACTS / "intent_router.npz")


if __name__ == "__main__":
    main()

"""Train the small intent router from prepared HF rows plus curated bilingual seeds."""
from __future__ import annotations

import json
from pathlib import Path

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
    router = IntentRouter()
    stats = router.train(examples)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    router.save(str(ARTIFACTS / "intent_router.npz"))
    (ARTIFACTS / "intent_router_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Saved:", ARTIFACTS / "intent_router.npz")


if __name__ == "__main__":
    main()

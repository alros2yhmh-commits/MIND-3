"""Train the MIND-3 AKA Python semantic mapper from random initialization.

This trains the repository's own reference-ID/transition architecture. It does
not load pretrained weights and does not use the v3 intent-router checkpoint.
The first experiment learns Python task-family mappings; code answers are
retrieved as explicitly-labelled reference patterns, not generated solutions.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mind3.knowledge import KnowledgeUnit, unknown_unit
from mind3.model import Mind3Model
from mind3.registry import SymbolRegistry
from mind3.training import train_model
from mind3.transitions import softmax

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CONCEPTS = ROOT / "data" / "curated" / "python_concepts.jsonl"
DEFAULT_OUT = ROOT / "artifacts" / "python_knowledge"


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run scripts.build_python_knowledge first.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def concept_examples() -> list[dict]:
    rows = read_jsonl(CONCEPTS)
    examples = []
    for row in rows:
        aliases = row.get("aliases", {})
        candidates = [row.get("concept", "")]
        candidates += aliases.get("en", [])
        candidates += aliases.get("ar", [])
        for phrase in dict.fromkeys(str(x).strip() for x in candidates if str(x).strip()):
            examples.append({
                "id": f"concept:{row['id']}:{phrase}",
                "prompt": phrase,
                "family": "concept_explanation",
                "source": row.get("source", ""),
                "verification": row.get("verification", "official_docs_reference"),
                "source_reliability": 0.90,
                "kind": "concept_alias",
            })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-params", type=int, choices=(10000, 50000, 100000), default=10000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_rows = read_jsonl(PROCESSED / "python_train_examples.jsonl")
    dev_rows = read_jsonl(PROCESSED / "python_validation_examples.jsonl")
    train_rows = train_rows + concept_examples()
    if not train_rows:
        raise ValueError("No training examples were built.")
    labels = sorted({str(row["family"]) for row in train_rows})
    units = [
        KnowledgeUnit(
            id=label,
            claim=f"Internal AKA semantic task family: {label}",
            category="Python task family",
            confidence=0.85,
            source="MIND-3 Python knowledge graph",
            verification="structural_or_curated_training_target",
            answer_en=f"Internal task family: {label}",
            answer_ar=f"تصنيف داخلي لمهمة Python: {label}",
        )
        for label in labels
    ]
    units.append(unknown_unit())
    label_to_index = {unit.id: i for i, unit in enumerate(units)}

    texts = [str(row["prompt"]) for row in train_rows]
    registry = SymbolRegistry().fit(texts)
    examples = [
        (str(row["prompt"]), label_to_index[str(row["family"])],
         float(row.get("source_reliability", 0.70)))
        for row in train_rows
    ]

    # Choose embedding width from the requested parameter budget.
    # Exact count is recorded below; the requested budget is a target, not a claim.
    denominator = registry.size + len(units)
    dim = max(8, int((args.target_params - len(units)) / max(denominator, 1)))
    dim = max(8, (dim // 8) * 8)
    model = Mind3Model(
        units, registry=registry, dim=dim,
        use_reference_ids=True, use_transitions=True, use_confidence=True,
        seed=args.seed, support_threshold=0.30,
    )
    stats = train_model(model, examples, epochs=args.epochs, lr=args.lr, log_every=5)
    model.compute_prototypes(examples)

    # MBPP validation is development-only. It is not used for gradient updates.
    correct = 0
    evaluated = 0
    per_family = Counter()
    per_family_correct = Counter()
    for row in dev_rows:
        family = str(row["family"])
        if family not in label_to_index:
            continue
        predicted = int(np.argmax(softmax(model.logits(str(row["prompt"])))))
        evaluated += 1
        per_family[family] += 1
        if predicted == label_to_index[family]:
            correct += 1
            per_family_correct[family] += 1

    parameter_count = int(model.W.size + model.b.size + model.E.size)
    stats.update({
        "architecture": "AKA-PY-1",
        "initialization": "random",
        "pretrained_weights_loaded": False,
        "intent_router_checkpoint_loaded": False,
        "target_parameter_budget": args.target_params,
        "actual_parameter_count": parameter_count,
        "embedding_dim": dim,
        "train_family_counts": dict(sorted(Counter(r["family"] for r in train_rows).items())),
        "validation_examples": evaluated,
        "validation_family_accuracy": (correct / evaluated) if evaluated else None,
        "validation_per_family": {
            family: {
                "count": count,
                "correct": per_family_correct[family],
                "accuracy": per_family_correct[family] / count,
            }
            for family, count in sorted(per_family.items())
        },
        "mbpp_test_opened": False,
        "mbpp_prompt_opened": False,
        "validation_interpretation": "AST-family classification only; not code correctness or answer-generation accuracy.",
    })

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already contains files. Choose a new --output or pass --overwrite."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output / "model.npz"))
    registry.save(str(args.output / "registry.json"))
    (args.output / "units.json").write_text(
        json.dumps([asdict(u) for u in units], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config = {
        "model_name": "MIND-3",
        "architecture": "AKA-PY-1",
        "variant": "python_knowledge",
        "dim": dim,
        "seed": args.seed,
        "margin": 0.20,
        "beam_k": 4,
        "support_threshold": 0.30,
        "use_reference_ids": True,
        "use_transitions": True,
        "use_confidence": True,
        "encoder": "position-rotated-reference-embeddings-v2",
        "training_source": "data/processed/python_train_examples.jsonl + curated concept aliases",
        "validation_source": "data/processed/python_validation_examples.jsonl",
        "pretrained_weights": False,
    }
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "training_log.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "config": config, "training": stats},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

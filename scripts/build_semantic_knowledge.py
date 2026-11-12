"""Build AKA-PY-2 semantic facts from curated, source-linked Python relations.

No code execution, pretrained model, or generated AST labels are used.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONCEPTS = ROOT / "data" / "curated" / "python_concepts.jsonl"
RELATIONS = ROOT / "data" / "curated" / "python_relations_v2.jsonl"
OUT = ROOT / "data" / "processed"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    concepts = read_jsonl(CONCEPTS)
    relations = read_jsonl(RELATIONS)
    concept_ids = {f"py-{c['id'].removeprefix('py-')}" if not c["id"].startswith("py-") else c["id"] for c in concepts}
    # Preserve the curated IDs exactly; relation subjects must resolve to a known concept.
    concept_ids = {c["id"] for c in concepts}
    triples, nodes, edges = [], {}, []
    for c in concepts:
        nodes[c["id"]] = {"id": c["id"], "type": "concept", "label": c["concept"],
            "claim_en": c.get("claim_en", ""), "claim_ar": c.get("claim_ar", ""),
            "source": c.get("source", ""), "verification": c.get("verification", "")}
    for row in relations:
        if row["subject"] not in concept_ids:
            raise ValueError(f"Unknown curated subject: {row['subject']}")
        if row["split"] not in {"train", "dev"}:
            raise ValueError(f"Invalid split for {row['id']}: {row['split']}")
        triple = {k: row[k] for k in ("id", "subject", "relation", "object", "source", "verification", "source_reliability", "split")}
        triples.append(triple)
        nodes.setdefault(row["object"], {"id": row["object"], "type": "entity", "label": row["object"]})
        edges.append({"id": row["id"], "source": row["subject"], "relation": row["relation"],
                      "target": row["object"], "provenance": row["source"],
                      "verification": row["verification"], "split": row["split"]})
    if not triples or not any(t["split"] == "train" for t in triples) or not any(t["split"] == "dev" for t in triples):
        raise ValueError("Need non-empty train and dev triples.")
    write_jsonl(OUT / "semantic_triples_v2.jsonl", triples)
    write_jsonl(OUT / "semantic_nodes_v2.jsonl", list(nodes.values()))
    write_jsonl(OUT / "semantic_edges_v2.jsonl", edges)
    manifest = {
        "architecture": "AKA-PY-2",
        "source": "data/curated/python_relations_v2.jsonl + python_concepts.jsonl",
        "concepts": len(concepts), "triples": len(triples),
        "train_triples": sum(t["split"] == "train" for t in triples),
        "dev_triples": sum(t["split"] == "dev" for t in triples),
        "entities": len(nodes),
        "pretrained_model_used": False, "code_executed": False,
        "label_source": "explicit curated subject-relation-object facts; not AST-derived task families",
        "limitations": [
            "This is a small curated graph-learning experiment, not natural-language understanding.",
            "The neural component learns graph-link scores, not free-form code or answer generation.",
            "Dev triples are held out from gradient updates; the graph builder retains their provenance for evaluation only."
        ]
    }
    (OUT / "semantic_manifest_v2.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

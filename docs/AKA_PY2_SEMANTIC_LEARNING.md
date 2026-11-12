# AKA-PY-2 — Semantic Knowledge Learning

## Goal
AKA-PY-2 replaces the previous AST-family classification objective with a small, explicit subject-relation-object graph-learning experiment.

## Data contract
- Concepts and bilingual claims come from `data/curated/python_concepts.jsonl`.
- Explicit, source-linked relation triples come from `data/curated/python_relations_v2.jsonl`.
- Every triple has provenance, a verification label, reliability, and a fixed train/dev split.
- AST-derived task-family labels are not used as semantic targets.
- MBPP test and prompt files are not read.

## Architecture
1. Curated fact records establish the graph vocabulary and evidence.
2. The semantic graph stores directed subject-relation-object facts with provenance.
3. A from-scratch NumPy DistMult model learns entity/relation embeddings using pairwise negative sampling.
4. Held-out dev triples evaluate tail ranking only.
5. A deterministic path inspector can expose multi-hop paths while preserving edge provenance and excluding dev edges by default.

## Important boundaries
- This is a compact graph representation-learning experiment, not a pretrained language model.
- It does not generate natural-language answers or Python programs.
- Link prediction scores are not proof that a fact is true; accepted answers must still be grounded in stored evidence.
- The initial curated graph is small. Results are diagnostic, and a larger independently reviewed relation corpus is needed before making generalization claims.
- A found path is an evidence path, not automatically a valid logical implication. Rule schemas must be explicitly defined before a path can be promoted to a new fact.

## Commands
```powershell
python -m scripts.build_semantic_knowledge
python -m tests.test_semantic_knowledge
python -m scripts.train_semantic_mind3 --dim 32 --epochs 500
Get-Content artifacts/aka_py2/training_report.json -Raw -Encoding utf8
```

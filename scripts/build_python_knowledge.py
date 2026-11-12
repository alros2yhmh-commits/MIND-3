"""Build an explicit Python knowledge graph from the prepared local datasets.

This script never executes downloaded code. AST parsing extracts syntax-level
structure only; it does not prove semantic correctness.

Inputs:
  data/processed/python_train.jsonl
  data/processed/mbpp_train_reference.jsonl
  data/processed/mbpp_validation.jsonl
  data/curated/python_concepts.jsonl

The MBPP test and prompt files are intentionally never opened.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
CURATED = DATA / "curated"
FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.I | re.S)
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[\u0600-\u06FF]+")
CONSTRUCTS = {
    "FunctionDef", "AsyncFunctionDef", "ClassDef", "If", "IfExp", "For",
    "AsyncFor", "While", "Try", "TryStar", "With", "AsyncWith", "ListComp",
    "SetComp", "DictComp", "GeneratorExp", "Lambda", "Return", "Yield",
    "Raise", "Assert", "Match", "BoolOp", "Compare", "BinOp", "Assign",
    "AnnAssign", "AugAssign",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required prepared file is missing: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_id(prefix: str, value: str) -> str:
    return prefix + ":" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def extract_code(text: str) -> str:
    matches = FENCE_RE.findall(text or "")
    return max(matches, key=len).strip() if matches else (text or "").strip()


def ast_facts(code: str) -> dict[str, list[str]]:
    tree = ast.parse(extract_code(code))
    constructs = sorted({type(n).__name__ for n in ast.walk(tree)
                         if type(n).__name__ in CONSTRUCTS})
    defines = sorted({n.name for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))})
    imports = set()
    calls = set()
    names = set()
    function_names = set(defines)
    recursive = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "") + "." + ",".join(a.name for a in node.names))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
                recursive = recursive or node.func.id in function_names
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return {
        "constructs": constructs,
        "defines": sorted(defines),
        "imports": sorted(imports),
        "calls": sorted(calls),
        "names": sorted(names),
        "recursive_call_present": ["true"] if recursive else [],
    }


def task_family(facts: dict[str, list[str]]) -> str:
    c = set(facts["constructs"])
    if "Try" in c or "TryStar" in c or "Raise" in c:
        return "exception_handling"
    if c & {"ListComp", "SetComp", "DictComp", "GeneratorExp"}:
        return "comprehensions"
    if facts["recursive_call_present"]:
        return "recursion"
    if c & {"For", "AsyncFor", "While"}:
        return "iteration"
    if c & {"ClassDef"}:
        return "class_and_objects"
    if c & {"With", "AsyncWith"}:
        return "context_management"
    if c & {"If", "IfExp", "Match"}:
        return "conditional_logic"
    if facts["imports"] and not facts["defines"]:
        return "module_and_imports"
    if facts["defines"]:
        return "function_definition"
    if c & {"Compare", "BoolOp", "BinOp"}:
        return "expressions_and_operators"
    return "general_python_task"


def normalise_row(row: dict[str, Any], split: str, fallback_source: str) -> dict[str, Any] | None:
    prompt = str(row.get("prompt") or "").strip()
    response = str(row.get("response") or row.get("code") or "").strip()
    if len(prompt) < 4 or not response:
        return None
    try:
        facts = ast_facts(response)
    except (SyntaxError, ValueError, TypeError):
        return None
    source = str(row.get("source") or fallback_source)
    verification = str(row.get("verification") or "syntax_checked_only")
    example_id = stable_id("example", str(row.get("id") or prompt + "\n" + response))
    return {
        "id": example_id,
        "original_id": str(row.get("id") or ""),
        "kind": "code_example",
        "split": split,
        "source": source,
        "verification": verification,
        "source_reliability": 0.55 if "syntax" in verification else 0.70,
        "prompt": prompt,
        "response": response,
        "language": str(row.get("language") or "en"),
        "family": task_family(facts),
        "ast_facts": facts,
        "test_count": len(row.get("tests") or []),
    }


def main() -> None:
    csn = read_jsonl(PROCESSED / "python_train.jsonl")
    mbpp_train = read_jsonl(PROCESSED / "mbpp_train_reference.jsonl")
    mbpp_dev = read_jsonl(PROCESSED / "mbpp_validation.jsonl")
    concepts = read_jsonl(CURATED / "python_concepts.jsonl")

    train_by_id: dict[str, dict[str, Any]] = {}
    for row in csn:
        item = normalise_row(row, "train", "Nan-Do/instructional_code-search-net-python")
        if item:
            train_by_id[item["id"]] = item
    for row in mbpp_train:
        item = normalise_row(row, "train", "huggy-1/mbpp")
        if item:
            train_by_id[item["id"]] = item
    train_examples = list(train_by_id.values())

    dev_examples = []
    for row in mbpp_dev:
        item = normalise_row(row, "validation", "huggy-1/mbpp")
        if item:
            dev_examples.append(item)

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    def add_node(node: dict[str, Any]) -> None:
        nodes[node["id"]] = node

    def add_edge(source: str, relation: str, target: str, provenance: str,
                 verification: str) -> None:
        edge_key = stable_id("edge", source + "|" + relation + "|" + target)
        edges[edge_key] = {
            "id": edge_key, "source": source, "relation": relation,
            "target": target, "provenance": provenance,
            "verification": verification,
        }

    all_examples = train_examples + dev_examples
    for example in all_examples:
        eid = example["id"]
        add_node({
            "id": eid, "type": "code_example", "label": example["prompt"],
            "split": example["split"], "source": example["source"],
            "verification": example["verification"],
            "source_reliability": example["source_reliability"],
        })
        facts = example["ast_facts"]
        for construct in facts["constructs"]:
            nid = "construct:" + construct
            add_node({"id": nid, "type": "syntax_construct", "label": construct,
                      "source": "Python AST parser", "verification": "structural_parse"})
            add_edge(eid, "contains_construct", nid, example["source"], example["verification"])
        for relation, values in (
            ("defines", facts["defines"]), ("imports", facts["imports"]),
            ("calls", facts["calls"]), ("uses_name", facts["names"]),
        ):
            for value in values:
                nid = stable_id("symbol", value)
                add_node({"id": nid, "type": "python_symbol", "label": value,
                          "source": "Python AST parser", "verification": "structural_parse"})
                add_edge(eid, relation, nid, example["source"], example["verification"])

    concept_by_alias: list[tuple[str, str, str]] = []
    for concept in concepts:
        cid = str(concept["id"])
        add_node({
            "id": "concept:" + cid, "type": "concept",
            "label": str(concept.get("concept") or cid),
            "claim_en": concept.get("claim_en", ""),
            "claim_ar": concept.get("claim_ar", ""),
            "source": concept.get("source", ""),
            "verification": concept.get("verification", "official_docs_reference"),
            "source_reliability": 0.90,
        })
        aliases = concept.get("aliases", {})
        for lang in ("en", "ar"):
            for alias in aliases.get(lang, []):
                if str(alias).strip():
                    concept_by_alias.append((str(alias).strip().lower(), "concept:" + cid, cid))

    # Exact alias links only; this is deliberately not represented as semantic verification.
    for example in all_examples:
        haystack = (example["prompt"] + "\n" + example["response"]).lower()
        for alias, concept_node, _concept_id in concept_by_alias:
            if alias and alias in haystack:
                add_edge(example["id"], "literal_alias_match", concept_node,
                         example["source"], "lexical_link_only")

    for example in train_examples:
        example["split"] = "train"
    for example in dev_examples:
        example["split"] = "validation"

    write_jsonl(PROCESSED / "python_train_examples.jsonl", train_examples)
    write_jsonl(PROCESSED / "python_validation_examples.jsonl", dev_examples)
    write_jsonl(PROCESSED / "knowledge_nodes.jsonl", list(nodes.values()))
    write_jsonl(PROCESSED / "knowledge_edges.jsonl", list(edges.values()))

    family_counts = dict(sorted(Counter(e["family"] for e in train_examples).items()))
    source_counts = dict(sorted(Counter(e["source"] for e in train_examples).items()))
    manifest = {
        "architecture": "AKA-PY-1",
        "builder_version": 1,
        "train_examples": len(train_examples),
        "validation_examples": len(dev_examples),
        "nodes": len(nodes),
        "edges": len(edges),
        "family_counts": family_counts,
        "source_counts": source_counts,
        "concept_records": len(concepts),
        "mbpp_test_opened": False,
        "mbpp_prompt_opened": False,
        "code_executed": False,
        "validation_role": "development_only",
        "limitations": [
            "AST parsing checks structure/syntax, not semantic correctness.",
            "Literal alias links are lexical links, not verified semantic relations.",
            "CodeSearchNet responses are syntax_checked_only unless their source row says otherwise.",
            "MBPP reference solutions/tests were not executed.",
        ],
    }
    (PROCESSED / "knowledge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

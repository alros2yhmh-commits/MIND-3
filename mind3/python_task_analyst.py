"""Evidence-labelled retrieval over the MIND-3 Python knowledge graph.

The neural core predicts a task family. This module then retrieves a related
training example from that family. It deliberately does NOT claim to synthesize
a new solution or verify the retrieved code against the user's exact task.
Downloaded code is never executed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .knowledge import KnowledgeUnit
from .model import Mind3Model
from .registry import SymbolRegistry

TOKEN_RE = re.compile(r"[a-z_][a-z_0-9]*|[\u0600-\u06FF]+", re.I)
STOP = {
    "write", "create", "make", "build", "implement", "please", "a", "an", "the",
    "in", "python", "function", "code", "for", "to", "me", "how", "can", "you",
    "اكتب", "اعمل", "أنشئ", "انشئ", "دالة", "كود", "بايثون", "لي", "من", "في",
    "ممكن", "عايز", "عاوز", "برمج",
}
TASK_RE = re.compile(
    r"\b(write|create|implement|build|fix|debug|code|function|program|script)\b"
    r"|اكتب|اعمل|أنشئ|انشئ|صلح|صحح|برمج|كود|دالة",
    re.I,
)


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOP}


def similarity(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


class PythonTaskAnalyst:
    def __init__(self, artifacts_dir: str, examples_path: str,
                 nodes_path: str | None = None, edges_path: str | None = None) -> None:
        root = Path(artifacts_dir)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        units = [KnowledgeUnit(**item) for item in json.loads(
            (root / "units.json").read_text(encoding="utf-8")
        )]
        self.registry = SymbolRegistry.load(str(root / "registry.json"))
        self.model = Mind3Model(
            units, registry=self.registry, dim=int(config["dim"]),
            seed=int(config["seed"]), margin=float(config["margin"]),
            beam_k=int(config["beam_k"]),
            support_threshold=float(config["support_threshold"]),
            use_reference_ids=bool(config["use_reference_ids"]),
            use_transitions=bool(config["use_transitions"]),
            use_confidence=bool(config["use_confidence"]),
        )
        self.model.load_weights(str(root / "model.npz"))
        self.examples_by_family: dict[str, list[dict]] = {}
        with open(examples_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.examples_by_family.setdefault(row["family"], []).append(row)

        self.graph_nodes = {}
        if nodes_path and Path(nodes_path).is_file():
            with open(nodes_path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        node = json.loads(line)
                        self.graph_nodes[node["id"]] = node
        self.graph_edges_by_source: dict[str, list[dict]] = {}
        if edges_path and Path(edges_path).is_file():
            with open(edges_path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        edge = json.loads(line)
                        self.graph_edges_by_source.setdefault(edge["source"], []).append(edge)

    def answer(self, message: str):
        if not TASK_RE.search(message):
            return None
        result = self.model.infer(message)
        chosen = result["chosen"]
        family = chosen.unit_id
        if family == "unknown" or family == "concept_explanation":
            return None
        candidates = self.examples_by_family.get(family, [])
        if not candidates:
            return None
        ranked = sorted(
            ((similarity(message, row["prompt"]), row) for row in candidates),
            key=lambda item: item[0], reverse=True,
        )
        score, row = ranked[0]
        if score < 0.08:
            return None
        is_arabic = bool(re.search(r"[\u0600-\u06FF]", message))
        intro = (
            "وجدت نمطًا مرجعيًا قريبًا من بيانات Python، وليس حلًا متحققًا لمهمتك بعينها:"
            if is_arabic else
            "I found a related Python reference pattern. It is not a verified solution to your exact task:"
        )
        code = row.get("response", "")
        related_edges = self.graph_edges_by_source.get(row["id"], [])
        graph_relations = [
            {
                "relation": edge["relation"],
                "target": self.graph_nodes.get(edge["target"], {}).get("label", edge["target"]),
                "verification": edge.get("verification", ""),
            }
            for edge in related_edges
        ]
        reply = intro + "\n\n```python\n" + code + "\n```"
        return {
            "mode": "python_knowledge_reference",
            "lang": "ar" if is_arabic else "en",
            "reply": reply,
            "sources": [],
            "available": True,
            "unit": family,
            "diagnostics": {
                "chosen_unit": family,
                "reason": "neural task-family mapping followed by evidence-labelled example retrieval",
                "events": [
                    "Retrieved reference only; no code execution performed",
                    "Reference is not asserted to solve the exact request",
                ],
                "paths": [
                    {"unit": p.unit_id, "model_prob": round(p.model_prob, 4),
                     "confidence": p.confidence, "verified": p.verified,
                     "final_score": round(p.final_score, 4)}
                    for p in result["paths"][:5]
                ],
                "conflict": None,
                "source": row.get("source", ""),
                "verification": row.get("verification", ""),
                "retrieval_similarity": round(score, 4),
                "ast_facts": row.get("ast_facts", {}),
                "knowledge_graph_relations": graph_relations,
                "reference_prompt": row.get("prompt", ""),
            },
        }

"""Load and inspect the curated AKA-PY-2 semantic knowledge graph."""
from __future__ import annotations
from collections import defaultdict, deque
import json
from pathlib import Path


class SemanticKnowledge:
    def __init__(self, triples):
        self.triples = list(triples)
        self.outgoing = defaultdict(list)
        for fact in self.triples:
            self.outgoing[fact["subject"]].append(fact)

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as f:
            return cls(json.loads(line) for line in f if line.strip())

    def facts(self, subject, relation=None):
        return [f for f in self.outgoing.get(subject, [])
                if relation is None or f["relation"] == relation]

    def find_path(self, start, target, max_hops=4, include_dev=False):
        """Return a provenance-carrying path; never invent a missing edge."""
        allowed = self.triples if include_dev else [f for f in self.triples if f["split"] == "train"]
        graph = defaultdict(list)
        for fact in allowed:
            graph[fact["subject"]].append(fact)
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node == target and path:
                return {"status": "path_found", "path": path,
                        "hops": len(path), "derived_link": [start, target]}
            if len(path) >= max_hops:
                continue
            for fact in graph.get(node, []):
                nxt = fact["object"]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [fact]))
        return {"status": "missing", "path": [], "hops": 0,
                "reason": "No supported directed path exists within the hop limit."}

"""The experimental MIND-3 model.

Ablation flags keep the same data/head while disabling one hypothesis at a time.
Reference-ID mode uses learned symbol embeddings with position-aware pooling:
the same symbols in a different order no longer collapse to the same mean.
"""
from __future__ import annotations

import numpy as np

from .knowledge import KnowledgeUnit
from .registry import SymbolRegistry
from .transitions import Path, TransitionSystem, softmax


class Mind3Model:
    def __init__(self, units: list[KnowledgeUnit], registry: SymbolRegistry | None = None,
                 dim: int = 48, use_reference_ids: bool = True,
                 use_transitions: bool = True, use_confidence: bool = True,
                 beam_k: int = 4, margin: float = 0.30, seed: int = 0,
                 support_threshold: float = 0.5) -> None:
        self.units = units
        self.dim = dim
        self.use_reference_ids = use_reference_ids
        self.use_transitions = use_transitions
        self.use_confidence = use_confidence
        self.rng = np.random.default_rng(seed)
        self.registry = registry
        self.prototypes = None
        self.unknown_index = next(i for i, u in enumerate(units) if u.id == "unknown")
        self.W = self.rng.normal(0, 0.1, (dim, len(units)))
        self.b = np.zeros(len(units))
        if use_reference_ids:
            if registry is None:
                raise ValueError("registry required when use_reference_ids=True")
            self.E = self.rng.normal(0, 0.1, (registry.size, dim))
        else:
            self.E = None
        self.ts = TransitionSystem(
            units, beam_k=beam_k, verify=True, revive=True,
            unknown_index=self.unknown_index, margin=margin,
            support_threshold=support_threshold,
        )

    def compute_prototypes(self, examples) -> None:
        """Build class evidence prototypes using the same encoder used at inference."""
        sums = np.zeros((len(self.units), self.dim))
        counts = np.zeros(len(self.units))
        for text, idx, _weight in examples:
            sums[idx] += self.features(text)
            counts[idx] += 1
        counts[counts == 0] = 1
        self.prototypes = sums / counts[:, None]
        self.ts.prototypes = self.prototypes

    def features(self, text: str) -> np.ndarray:
        if self.use_reference_ids:
            ids = self.registry.encode(text) or [0]
            # Rotate each token embedding by its sequence position before pooling.
            # This is intentionally lightweight, but unlike plain mean pooling it
            # preserves some order information (e.g. "abc" != "cba").
            positioned = [
                np.roll(self.E[token_id], shift=position % self.dim)
                for position, token_id in enumerate(ids)
            ]
            return np.mean(positioned, axis=0)

        # Ablation baseline: order-insensitive hashed character bag of features.
        v = np.zeros(self.dim)
        for ch in text:
            v[ord(ch) % self.dim] += 1.0
        return v / max(len(text), 1)

    def logits(self, text: str) -> np.ndarray:
        return self.features(text) @ self.W + self.b

    def infer(self, text: str) -> dict:
        c = self.features(text)
        logits = c @ self.W + self.b
        if self.use_transitions:
            return self.ts.infer(logits, context=c)
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        u = self.units[idx]
        chosen = Path(idx, u.id, float(probs[idx]), float(u.confidence), True, float(probs[idx]))
        return {
            "chosen": chosen,
            "paths": [chosen],
            "reason": "greedy argmax (transition system disabled)",
            "events": [],
        }

    def save(self, path: str) -> None:
        arrays = {"W": self.W, "b": self.b}
        if self.E is not None:
            arrays["E"] = self.E
        if self.prototypes is not None:
            arrays["P"] = self.prototypes
        np.savez(path, **arrays)

    def load_weights(self, path: str) -> None:
        data = np.load(path)
        self.W = data["W"]
        self.b = data["b"]
        if self.E is not None and "E" in data:
            self.E = data["E"]
        if "P" in data:
            self.prototypes = data["P"]
            self.ts.prototypes = self.prototypes

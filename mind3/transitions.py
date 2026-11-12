"""H2 — Probabilistic Transition System with Multi-Path Verification.

Research distinction (documented):
  * LEARNED part  : W, b produce p(unit | context) — part of the model structure.
  * INFERENCE part: beam of candidate paths, multi-path verification against the
                    knowledge store, and revival of pruned paths — an independent
                    inference-time mechanism.

Verification checks, per candidate path:
  (a) the stored claim has an answer and non-zero confidence, and
  (b) SUPPORT: the input context actually resembles the stored evidence for that
      claim (cosine similarity to the unit's training-example prototype).
A path failing verification is not committed; if no knowledge path is supported,
the system admits missing knowledge (unknown) instead of asserting a fact.

Model probabilities are kept separate from the stored claim's confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class Path:
    unit_index: int
    unit_id: str
    model_prob: float
    confidence: float
    verified: bool
    final_score: float
    support: float = 1.0
    note: str = ""


class TransitionSystem:
    def __init__(self, units, beam_k: int = 4, verify: bool = True,
                 revive: bool = True, unknown_index: int | None = None,
                 margin: float = 0.30, prototypes=None,
                 support_threshold: float = 0.5) -> None:
        self.units = units
        self.beam_k = beam_k
        self.verify = verify
        self.revive = revive
        self.unknown_index = unknown_index
        self.margin = margin
        self.prototypes = prototypes
        self.support_threshold = support_threshold

    def _make_path(self, idx: int, probs: np.ndarray, context=None) -> Path:
        u = self.units[idx]
        conf = float(u.confidence)
        support = 1.0
        if self.verify:
            base = bool(u.answer_ar or u.answer_en) and conf > 0.0
            if self.prototypes is not None and context is not None:
                support = cosine(context, self.prototypes[idx])
                base = base and support >= self.support_threshold
            verified = base
            final = float(probs[idx]) * (0.5 + 0.5 * conf)
        else:
            verified = True
            final = float(probs[idx])
        return Path(int(idx), u.id, float(probs[idx]), conf, verified, final, support)

    def infer(self, logits: np.ndarray, context=None) -> dict:
        probs = softmax(logits)
        order = np.argsort(-probs)
        beam = list(order[: self.beam_k])
        pruned = list(order[self.beam_k:])
        events: list[str] = []
        paths = [self._make_path(i, probs, context) for i in beam]

        def unknown_path():
            up = next((p for p in paths if p.unit_index == self.unknown_index), None)
            if up is None and self.revive and self.unknown_index is not None:
                up = self._make_path(self.unknown_index, probs, context)
                paths.append(up)
                events.append("revived unknown unit for re-verification")
            return up

        chosen = None
        reason = ""

        supported_knowledge = [
            p for p in paths
            if p.verified and p.unit_index != self.unknown_index
            and p.model_prob >= self.margin
        ]
        if not supported_knowledge and self.revive:
            for idx in pruned:
                p = self._make_path(idx, probs, context)
                paths.append(p)
                events.append(f"revived pruned path '{p.unit_id}' for re-verification")
            supported_knowledge = [
                p for p in paths
                if p.verified and p.unit_index != self.unknown_index
                and p.model_prob >= self.margin
            ]

        if supported_knowledge:
            chosen = max(supported_knowledge, key=lambda p: p.final_score)
            reason = (
                f"highest verified+supported score {chosen.final_score:.3f} "
                f"(model_prob {chosen.model_prob:.3f} x confidence {chosen.confidence:.2f}, "
                f"support {chosen.support:.2f})"
            )
        else:
            up = unknown_path()
            if up is not None:
                chosen = up
                reason = ("no knowledge path passed verification/support "
                          "-> admit missing knowledge")
            else:
                chosen = max(paths, key=lambda p: p.final_score)
                reason = "fallback highest score (verification disabled)"

        return {
            "chosen": chosen,
            "paths": sorted(paths, key=lambda p: -p.final_score),
            "reason": reason,
            "events": events,
        }
"""Controlled H3 experiment: does confidence weighting change selection?

Two records share IDENTICAL question patterns but differ in confidence.
With confidence-weighted loss the reliable (high-confidence) claim should win;
without weighting the tie is broken arbitrarily (here: the low-confidence one,
which is listed first). This isolates the effect of H3. Run: python eval/confidence_experiment.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mind3.knowledge import KnowledgeUnit, unknown_unit
from mind3.model import Mind3Model
from mind3.registry import SymbolRegistry
from mind3.training import train_model

QUESTIONS = ["سؤال مشترك؟", "shared question?"]


def build():
    low = KnowledgeUnit(id="low_conf", confidence=0.2, answer_ar="LOW", answer_en="LOW",
                        questions=list(QUESTIONS), source="unreliable")
    high = KnowledgeUnit(id="high_conf", confidence=1.0, answer_ar="HIGH", answer_en="HIGH",
                         questions=list(QUESTIONS), source="reliable")
    units = [low, high, unknown_unit()]  # low first so an unweighted tie picks it
    reg = SymbolRegistry().fit(QUESTIONS)
    examples = []
    for u in units:
        for q in u.questions:
            examples.append((q, units.index(u), u.confidence))
    return units, reg, examples


def main():
    units, reg, examples = build()
    out = {}
    for conf in (True, False):
        m = Mind3Model(units, registry=reg, dim=32, seed=0,
                       use_reference_ids=True, use_transitions=True, use_confidence=conf)
        train_model(m, examples, epochs=600, lr=0.05)
        res = m.infer("shared question?")
        out["confidence_on" if conf else "confidence_off"] = res["chosen"].unit_id
    print("H3 controlled experiment (identical questions, different confidence):")
    print(f"  confidence weighting ON  -> chooses: {out['confidence_on']}")
    print(f"  confidence weighting OFF -> chooses: {out['confidence_off']}")
    benefit = out["confidence_on"] == "high_conf" and out["confidence_off"] == "low_conf"
    print(f"  H3 shows measurable benefit: {benefit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
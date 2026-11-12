"""Stage 4 tests — Probabilistic Transition System mechanism.
Run: python -m tests.test_transitions
Uses controlled logits to verify beam / verification / revival / margin / logging.
End-to-end with trained weights is covered in Stage 5.
"""
from __future__ import annotations

import os

import numpy as np

from mind3.knowledge import KnowledgeUnit, load_records, unknown_unit
from mind3.transitions import TransitionSystem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unit(uid, conf=1.0, ar="إجابة", en="answer"):
    return KnowledgeUnit(id=uid, confidence=conf, answer_ar=ar, answer_en=en)


def main() -> int:
    failures = 0

    def check(name, ok):
        nonlocal failures
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    records = load_records(os.path.join(ROOT, "knowledge", "records"))
    units = records + [unknown_unit()]
    idx = {u.id: i for i, u in enumerate(units)}
    K = len(units)

    # 1. Multiple weighted candidate paths + context narrowing + logged reason.
    ts = TransitionSystem(units, beam_k=4, unknown_index=idx["unknown"], margin=0.30)
    logits = np.full(K, -2.0)
    logits[idx["name"]] = 5.0  # context "أنا اسمي..." favors name
    res = ts.infer(logits)
    check("multi-path beam keeps >=2 candidates", len(res["paths"]) >= 2)
    check("context narrows to 'name'", res["chosen"].unit_id == "name")
    check("selection reason logged", bool(res["reason"]))
    check("paths carry separate prob & confidence",
          all(hasattr(p, "model_prob") and hasattr(p, "confidence") for p in res["paths"]))

    # different context -> different choice
    logits2 = np.full(K, -2.0)
    logits2[idx["python_proficiency"]] = 5.0
    res2 = ts.infer(logits2)
    check("context 'مستواي في Python' narrows to proficiency",
          res2["chosen"].unit_id == "python_proficiency")

    # 2. Confidence weighting changes final score (prob != confidence).
    cu = [unit("high_prob_low_conf", conf=0.2), unit("low_prob_high_conf", conf=1.0), unknown_unit()]
    ts2 = TransitionSystem(cu, beam_k=3, unknown_index=2, margin=0.0)
    lg = np.array([3.0, 2.7, -5.0])  # first has higher model prob
    r = ts2.infer(lg)
    p0 = next(p for p in r["paths"] if p.unit_id == "high_prob_low_conf")
    p1 = next(p for p in r["paths"] if p.unit_id == "low_prob_high_conf")
    check("confidence down-weights low-confidence path", p0.final_score < p0.model_prob)
    check("high-confidence path can win despite lower prob",
          r["chosen"].unit_id == "low_prob_high_conf")

    # 3. Revival of pruned paths when beam fails verification.
    ru = [unit("bad0", ar="", en=""), unit("bad1", ar="", en=""), unit("good"), unknown_unit()]
    ts3 = TransitionSystem(ru, beam_k=2, unknown_index=3, margin=0.0)
    lg3 = np.array([5.0, 4.0, 3.0, -5.0])  # top-2 (beam) both fail verification
    r3 = ts3.infer(lg3)
    check("revival re-introduces pruned path", any("revived" in e for e in r3["events"]))
    check("revived valid path selected", r3["chosen"].unit_id == "good")

    # 4. Margin rule admits unknown when no knowledge is confident.
    ts4 = TransitionSystem(units, beam_k=4, unknown_index=idx["unknown"], margin=0.30)
    uniform = np.zeros(K)  # each prob ~1/K < 0.30
    r4 = ts4.infer(uniform)
    check("margin rule -> admit unknown", r4["chosen"].unit_id == "unknown")

    # 5. Verification can be disabled (ablation path still functional).
    ts5 = TransitionSystem(units, beam_k=4, verify=False, unknown_index=idx["unknown"], margin=0.0)
    r5 = ts5.infer(logits)
    check("verify=False still selects highest prob", r5["chosen"].unit_id == "name")

    print(f"\nRESULT: {'ALL PASS' if failures == 0 else str(failures) + ' FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
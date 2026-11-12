"""Tests for the reference-ID sequence encoder."""
from __future__ import annotations

import numpy as np

from mind3.knowledge import KnowledgeUnit
from mind3.model import Mind3Model
from mind3.registry import SymbolRegistry


def _units():
    return [
        KnowledgeUnit(id="sample", answer_en="sample", answer_ar="sample"),
        KnowledgeUnit(id="unknown", answer_en="unknown", answer_ar="unknown"),
    ]


def main() -> int:
    registry = SymbolRegistry().fit(["abc", "cba", "aabb"])
    model = Mind3Model(_units(), registry=registry, dim=16, seed=7)

    forward = model.features("abc")
    reverse = model.features("cba")
    same_order = model.features("abc")
    assert not np.allclose(forward, reverse), (
        "Reference-ID encoder collapsed different character orders"
    )
    assert np.allclose(forward, same_order), "Encoder must be deterministic"

    # The no-reference ablation intentionally remains a bag-of-characters baseline.
    baseline = Mind3Model(
        _units(), registry=registry, dim=16, seed=7, use_reference_ids=False
    )
    assert np.allclose(baseline.features("abc"), baseline.features("cba")), (
        "No-reference ablation should remain order-insensitive"
    )

    # Empty and unseen input must remain finite and safe.
    assert np.isfinite(model.features("")).all()
    registry_unknown = model.features("☃")
    assert np.isfinite(registry_unknown).all()

    print("[PASS] reference-ID encoder preserves character-order information")
    print("[PASS] no-reference ablation remains order-insensitive")
    print("[PASS] empty and unseen-symbol inputs remain finite")
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

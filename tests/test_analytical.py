"""Tests for evidence-aware derivation and the compact router."""
from __future__ import annotations

from mind3.analytical import (
    AnalyticalEngine, Claim, KnowledgeBase, completed_years, default_rules,
    verify_claim_against_derivation,
)
from mind3.intent_router import IntentRouter


def main():
    claims = [
        Claim("user", "date_of_birth", "2010-01-01", "user_statement", 1.0, "user_confirmed"),
        Claim("user", "reference_date", "2026-11-12", "test_fixture", 1.0, "verified"),
    ]
    kb = KnowledgeBase(claims)
    engine = AnalyticalEngine(kb, default_rules())
    result = engine.derive("completed_age", "user")
    assert result.status == "derived" and result.value == 16
    assert result.source_reliability == 1.0

    missing = AnalyticalEngine(
        KnowledgeBase([claims[0]]), default_rules()
    ).derive("completed_age", "user")
    assert missing.status == "missing"

    conflict_kb = KnowledgeBase(claims + [
        Claim("user", "date_of_birth", "2011-01-01", "other_record", 0.9, "verified")
    ])
    conflict = AnalyticalEngine(conflict_kb, default_rules()).derive("completed_age", "user")
    assert conflict.status == "conflict"

    direct = KnowledgeBase([
        Claim("user", "age", 15, "user_statement", 1.0, "user_confirmed")
    ]).resolve("user", "age")
    comparison = verify_claim_against_derivation(direct, result)
    assert comparison.status == "conflict"

    router = IntentRouter()
    assert router.parameter_count == 49928
    assert router.predict("اشرح المتغير في Python")["intent"] in {
        "explain_concept", "write_function", "debug_code", "trace_code",
        "compare_concepts", "translate_explain", "calculate", "general_python"
    }
    print("[PASS] derived age uses both explicit premises")
    print("[PASS] missing premises cause abstention")
    print("[PASS] conflicting premises are not silently resolved")
    print("[PASS] direct/derived conflict is detected")
    print("[PASS] intent router parameter count is 49,928")
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()

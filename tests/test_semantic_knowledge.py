"""Tests for AKA-PY-2 semantic facts and provenance-aware paths."""
from mind3.semantic_knowledge import SemanticKnowledge


def main():
    facts = [
        {"subject":"generator_function","relation":"produces","object":"generator_object","split":"train","source":"docs"},
        {"subject":"generator_object","relation":"is_a","object":"iterator","split":"train","source":"docs"},
        {"subject":"list","relation":"is_a","object":"mutable_sequence","split":"dev","source":"docs"},
    ]
    kb = SemanticKnowledge(facts)
    path = kb.find_path("generator_function", "iterator", max_hops=2)
    assert path["status"] == "path_found" and path["hops"] == 2
    assert kb.find_path("list", "mutable_sequence")["status"] == "missing"
    assert kb.find_path("list", "mutable_sequence", include_dev=True)["status"] == "path_found"
    assert kb.facts("generator_function", "produces")[0]["source"] == "docs"
    print("[PASS] multi-hop path with provenance")
    print("[PASS] dev facts excluded by default")
    print("[PASS] explicit dev inclusion")
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()

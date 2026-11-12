"""Focused tests for the AKA Python graph's AST structure extraction."""
from __future__ import annotations

from scripts.build_python_knowledge import ast_facts, task_family


def main() -> None:
    loop_code = """
def total(values):
    result = 0
    for value in values:
        result += value
    return result
"""
    facts = ast_facts(loop_code)
    assert "FunctionDef" in facts["constructs"]
    assert "For" in facts["constructs"]
    assert "total" in facts["defines"]
    assert task_family(facts) == "iteration"

    comp_code = "squares = [value * value for value in range(5)]"
    comp_facts = ast_facts(comp_code)
    assert "ListComp" in comp_facts["constructs"]
    assert task_family(comp_facts) == "comprehensions"

    import_code = "import math\nvalue = math.sqrt(9)"
    import_facts = ast_facts(import_code)
    assert "math" in import_facts["imports"]
    assert "sqrt" in import_facts["calls"]

    print("[PASS] AST function/loop facts")
    print("[PASS] comprehension family")
    print("[PASS] imports and call relations")
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()

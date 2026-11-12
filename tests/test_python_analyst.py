"""End-to-end tests for the bounded bilingual Python analyst."""
from __future__ import annotations

import os

from mind3.python_analyst import PythonAnalyst

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    analyst = PythonAnalyst(os.path.join(ROOT, "data", "curated", "python_concepts.jsonl"))
    english = analyst.answer("What is a Python variable?")
    assert english and english["mode"] == "analytical"
    assert "assignment statement" in english["reply"]
    assert english["diagnostics"]["evidence"][0]["source"].startswith("https://docs.python.org/")

    arabic = analyst.answer("اشرح لي المتغيرات في Python")
    assert arabic and arabic["lang"] == "ar"
    assert "اسم المتغير" in arabic["reply"]

    explicit_language = analyst.answer("Python variables are names for values. Explain that in Arabic.")
    assert explicit_language and explicit_language["lang"] == "ar"

    comparison = analyst.answer("Compare a Python list and tuple")
    assert comparison and "list" in comparison["reply"] and "tuple" in comparison["reply"]

    unsupported = analyst.answer("Explain quantum field theory")
    assert unsupported is None

    print("[PASS] English Python concept lookup")
    print("[PASS] Arabic Python concept lookup")
    print("[PASS] explicit response-language request")
    print("[PASS] comparison from two curated claims")
    print("[PASS] unsupported topic is not fabricated")
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()

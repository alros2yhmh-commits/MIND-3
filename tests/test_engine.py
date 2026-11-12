"""Stage 5 end-to-end smoke test through the Engine (trained weights).
Run: python -m tests.test_engine
"""
from __future__ import annotations

import os

from mind3.engine import Engine, detect_lang

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    eng = Engine(os.path.join(ROOT, "artifacts", "full"))
    failures = 0

    cases = [
        ("ما اسمي؟", "name", "ar", "Hamza"),
        ("How old am I?", "age", "en", "conflict"),
        ("What is my date of birth?", "dob", "en", "2010"),
        ("ما مستواي في Python؟", "python_proficiency", "ar", "50%"),
        ("اشرح المتغيرات في Python", "python_variables", "ar", "المتغيّر"),
        ("Explain decorators in Python", "unknown", "en", "trained knowledge"),
        ("Hello, how are you?", "greeting", "en", "MIND-3"),
        ("ما جنسي؟", "gender", "ar", "لن أستنتجه"),
    ]
    for q, want_unit, want_lang, want_sub in cases:
        r = eng.answer(q)
        ok = (r.get("unit") == want_unit and r["lang"] == want_lang
              and want_sub.lower() in r["reply"].lower())
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {q!r}")
        print(f"    lang={r['lang']} unit={r.get('unit')} reply={r['reply']}")
        if r.get("diagnostics"):
            d = r["diagnostics"]
            print(f"    reason={d['reason']}")
            if d["conflict"]:
                print(f"    conflict={d['conflict']}")

    print(f"\nRESULT: {'ALL PASS' if failures == 0 else str(failures) + ' FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
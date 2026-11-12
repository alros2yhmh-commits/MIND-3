"""Stage 6 tests — real on-demand search + routing. Run: python -m tests.test_search"""
from __future__ import annotations

import os

from mind3.engine import Engine
from mind3.search import search

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    failures = 0

    def check(name, ok):
        nonlocal failures
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    en = search("Python programming language", "en")
    check("EN search returns real sources", en["available"] and len(en["sources"]) > 0)
    check("sources have title+url", all(s["title"] and s["url"].startswith("http") for s in en["sources"]))
    print("    EN summary:", en["summary"].splitlines()[0])
    for s in en["sources"][:3]:
        print(f"    - [{s['provider']}] {s['title']} | {s['url']}")

    ar = search("لغة بايثون", "ar")
    check("AR search returns real sources", ar["available"] and len(ar["sources"]) > 0)
    for s in ar["sources"][:2]:
        print(f"    - [{s['provider']}] {s['title']} | {s['url']}")

    eng = Engine(os.path.join(ROOT, "artifacts", "full"))
    r = eng.answer("ابحث عن لغة بايثون")
    check("explicit AR request routes to search", r["mode"] == "search" and r["available"])
    r2 = eng.answer("search for Python language")
    check("explicit EN request routes to search", r2["mode"] == "search")
    r3 = eng.answer("ما اسمي؟")
    check("normal question does NOT auto-search", r3["mode"] == "knowledge" and not r3["sources"])

    print(f"\nRESULT: {'ALL PASS' if failures == 0 else str(failures) + ' FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
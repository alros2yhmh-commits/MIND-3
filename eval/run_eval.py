"""Stage 8 — reproducible evaluation suite + component ablations.

Trains every variant on the SAME data, runs the same battery, and measures
accuracy, unsupported-answer rate, latency and RAM. Run: python eval/run_eval.py
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from ctypes import wintypes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mind3.engine import Engine
from mind3.training import VARIANTS, train_variant

RESULTS = os.path.join(ROOT, "eval", "results")


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def ram_mb():
    try:
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        ok = psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        if not ok:
            return None, None
        return round(pmc.WorkingSetSize / 1e6, 1), round(pmc.PeakWorkingSetSize / 1e6, 1)
    except Exception:
        return None, None


TESTS = [
    dict(id="name_en", q="What is my name?", unit="name", sub="hamza", kind="k"),
    dict(id="name_ar", q="ما اسمي؟", unit="name", sub="hamza", kind="k"),
    dict(id="dob_en", q="What is my date of birth?", unit="dob", sub="2010", kind="k"),
    dict(id="age_conflict", q="How old am I?", unit="age", sub="conflict", kind="k", conflict=True),
    dict(id="proficiency", q="What is my Python proficiency?", unit="python_proficiency", sub="50%", kind="k"),
    dict(id="concept_trained", q="Explain variables in Python", unit="python_variables", sub="variable", kind="k"),
    dict(id="concept_untrained", q="Explain decorators in Python", unit="unknown", sub="trained knowledge", kind="probe"),
    dict(id="english_conv", q="Hello, how are you?", unit="greeting", sub="mind-3", kind="k", lang="en"),
    dict(id="gender", q="What is my gender?", unit="gender", sub="won't infer", kind="probe"),
    dict(id="hallucination", q="What color is my car?", unit="unknown", sub="trained knowledge", kind="probe"),
    # Out-of-distribution stress: the safe behavior is to admit missing knowledge.
    # Discriminates H2 (margin/verification) from greedy argmax.
    dict(id="ood_gibberish", q="zzz qqq www", unit="unknown", sub=None, kind="probe"),
    dict(id="ood_bare", q="Python", unit="unknown", sub=None, kind="probe"),
    dict(id="ood_partial_ar", q="أنا", unit="unknown", sub=None, kind="probe"),
    dict(id="search", q="search for Python programming language", kind="search"),
]


def run_battery(eng):
    rows = []
    for t in TESTS:
        t0 = time.perf_counter()
        r = eng.answer(t["q"])
        latency = (time.perf_counter() - t0) * 1000
        if t["kind"] == "search":
            ok = r["mode"] == "search" and r["available"] and len(r["sources"]) > 0
            unsupported = False
        else:
            ok = (r.get("unit") == t["unit"])
            if t.get("sub"):
                ok = ok and t["sub"].lower() in r["reply"].lower()
            if t.get("lang"):
                ok = ok and r["lang"] == t["lang"]
            if t.get("conflict"):
                ok = ok and bool(r["diagnostics"] and r["diagnostics"]["conflict"])
            unsupported = (t["kind"] == "probe" and r.get("unit") not in ("unknown", "gender"))
        rows.append({
            "id": t["id"], "question": t["q"], "kind": t["kind"],
            "expected_unit": t.get("unit"), "actual_unit": r.get("unit"),
            "mode": r["mode"], "lang": r["lang"], "reply": r["reply"],
            "sources": len(r["sources"]), "pass": bool(ok),
            "unsupported": bool(unsupported), "latency_ms": round(latency, 2),
        })
    return rows


def metrics(rows):
    k = [r for r in rows if r["kind"] != "search"]
    probes = [r for r in rows if r["kind"] == "probe"]
    search = [r for r in rows if r["kind"] == "search"]
    return {
        "accuracy": round(sum(r["pass"] for r in rows) / len(rows), 4),
        "knowledge_accuracy": round(sum(r["pass"] for r in k) / len(k), 4),
        "unsupported_rate": round(sum(r["unsupported"] for r in probes) / len(probes), 4) if probes else 0.0,
        "mean_knowledge_latency_ms": round(sum(r["latency_ms"] for r in k) / len(k), 2),
        "search_latency_ms": round(search[0]["latency_ms"], 2) if search else None,
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    all_results = {}
    for name in VARIANTS:
        out = os.path.join(ROOT, "artifacts", name)
        _, stats = train_variant(name, out)
        eng = Engine(out)
        rows = run_battery(eng)
        work, peak = ram_mb()
        m = metrics(rows)
        m["train_acc"] = stats["final_train_acc"]
        m["train_duration_s"] = stats["duration_s"]
        m["ram_working_mb"] = work
        m["ram_peak_mb"] = peak
        all_results[name] = {"metrics": m, "tests": rows}
        print(f"\n=== {name} === acc={m['accuracy']} unsupported={m['unsupported_rate']} "
              f"lat={m['mean_knowledge_latency_ms']}ms ram={work}MB")
        for r in rows:
            print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['id']}: unit={r['actual_unit']}")

    with open(os.path.join(RESULTS, "eval_all.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1)

    lines = ["# Stage 8 — Evaluation & Ablations (measured)", "",
             "| Variant | Accuracy | Knowledge acc | Unsupported rate | Mean lat (ms) | RAM (MB) | Train acc |",
             "|---|---|---|---|---|---|---|"]
    for name, d in all_results.items():
        m = d["metrics"]
        lines.append(f"| {name} | {m['accuracy']} | {m['knowledge_accuracy']} | "
                     f"{m['unsupported_rate']} | {m['mean_knowledge_latency_ms']} | "
                     f"{m['ram_working_mb']} | {m['train_acc']} |")
    with open(os.path.join(RESULTS, "stage8_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nWrote eval/results/eval_all.json and stage8_eval.md")


if __name__ == "__main__":
    main()
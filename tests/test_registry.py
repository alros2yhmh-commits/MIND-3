"""Stage 3 tests — Reference Token System. Run: python -m tests.test_registry"""
from __future__ import annotations

import time

from mind3.registry import SymbolRegistry, TraditionalVocab, UNK_ID

SAMPLES = [
    "ما اسمي؟",
    "What is my name?",
    "مستواي في Python 50%",
    "Hello, world! 123",
    "اشرح الحلقات في Python",
    "  spaces\tand\nnewlines  ",
    "عمري 15 وتاريخ ميلادي 2010-01-01",
]


def main() -> int:
    failures = 0
    reg = SymbolRegistry().fit(SAMPLES)

    # 1. Round-trip encode/decode on every sample (no unseen chars).
    for s in SAMPLES:
        ids = reg.encode(s)
        back = reg.decode(ids)
        ok = back == s
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] round-trip: {s!r}")

    # 2. No ID collisions: every symbol maps to a unique ID, bijective.
    ids = list(reg.sym2id.values())
    unique = len(ids) == len(set(ids))
    bijection = all(reg.id2sym[i] == s for s, i in reg.sym2id.items())
    print(f"[{'PASS' if unique and bijection else 'FAIL'}] no-collision bijection (vocab={reg.size})")
    failures += 0 if (unique and bijection) else 1

    # 3. Arabic + English + space + punctuation all present and distinct.
    probes = {"ع": None, "W": None, " ": None, "!": None, "؟": None, "5": None}
    for ch in probes:
        probes[ch] = reg.encode(ch)[0]
    distinct = len(set(probes.values())) == len(probes)
    print(f"[{'PASS' if distinct else 'FAIL'}] arabic/english/space/punct distinct IDs: {probes}")
    failures += 0 if distinct else 1

    # 4. Unseen char -> UNK, never crashes.
    unseen = reg.encode("\u2603")  # snowman not in corpus
    ok = unseen == [UNK_ID]
    print(f"[{'PASS' if ok else 'FAIL'}] unseen char -> UNK ({unseen})")
    failures += 0 if ok else 1

    # 5. Memory + speed vs traditional representation (honest measurement).
    trad = TraditionalVocab().fit(SAMPLES)
    corpus = SAMPLES * 200
    n = 2000

    t0 = time.perf_counter()
    for _ in range(n):
        for s in SAMPLES:
            reg.encode(s)
    ref_enc = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for _ in range(n):
        for s in SAMPLES:
            trad.encode(s)
    trad_enc = (time.perf_counter() - t0) / n

    ref_ids = reg.encode("".join(SAMPLES))
    trad_toks = trad.encode("".join(SAMPLES))
    ref_repr_bytes = sum(__import__("sys").getsizeof(i) for i in ref_ids)
    trad_repr_bytes = sum(__import__("sys").getsizeof(t) for t in trad_toks)

    print("\n--- Representation comparison (measured, not assumed) ---")
    print(f"registry mapping memory : {reg.memory_bytes()} bytes")
    print(f"traditional vocab memory: {trad.memory_bytes()} bytes")
    print(f"encoded-sample memory   : reference={ref_repr_bytes}B  traditional={trad_repr_bytes}B")
    print(f"encode time / batch     : reference={ref_enc*1e6:.1f}us  traditional={trad_enc*1e6:.1f}us")
    print(f"vocab size              : {reg.size} symbols")

    print(f"\nRESULT: {'ALL PASS' if failures == 0 else str(failures) + ' FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
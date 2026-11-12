"""H1 — Reference-Based Token Representation.

Explicit separation of four components:
  * Symbol Registry : symbols and their values (char <-> stable integer ID)
  * Token ID        : the integer the system actually processes
  * Embedding       : numeric representation looked up BY ID (see model.py)
  * Decoder         : maps produced IDs back to text

All processing paths operate on integer IDs. A TraditionalVocab is provided
only as a measurement baseline; the experiment does not assume IDs are better.
"""
from __future__ import annotations

import json
import sys

PAD = "<PAD>"
UNK = "<UNK>"
PAD_ID = 0
UNK_ID = 1


class SymbolRegistry:
    def __init__(self) -> None:
        self.id2sym: list[str] = [PAD, UNK]
        self.sym2id: dict[str, int] = {PAD: PAD_ID, UNK: UNK_ID}

    @property
    def size(self) -> int:
        return len(self.id2sym)

    def fit(self, texts) -> "SymbolRegistry":
        for text in texts:
            for ch in text:
                if ch not in self.sym2id:
                    self.sym2id[ch] = len(self.id2sym)
                    self.id2sym.append(ch)
        return self

    def encode(self, text: str) -> list[int]:
        unk = self.sym2id[UNK]
        return [self.sym2id.get(ch, unk) for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.id2sym[i] for i in ids if i != PAD_ID)

    def memory_bytes(self) -> int:
        total = sys.getsizeof(self.id2sym) + sys.getsizeof(self.sym2id)
        for sym, idx in self.sym2id.items():
            total += sys.getsizeof(sym) + sys.getsizeof(idx)
        for sym in self.id2sym:
            total += sys.getsizeof(sym)
        return total

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id2sym": self.id2sym}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SymbolRegistry":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        reg = cls()
        reg.id2sym = list(data["id2sym"])
        reg.sym2id = {s: i for i, s in enumerate(reg.id2sym)}
        return reg


class TraditionalVocab:
    """Baseline representation: the symbol strings themselves are carried
    through processing (no decoupled integer reference IDs)."""

    def __init__(self) -> None:
        self.tokens: set[str] = set()

    def fit(self, texts) -> "TraditionalVocab":
        for text in texts:
            self.tokens.update(text)
        return self

    def encode(self, text: str) -> list[str]:
        return [ch for ch in text]

    def decode(self, tokens) -> str:
        return "".join(tokens)

    def memory_bytes(self) -> int:
        total = sys.getsizeof(self.tokens)
        for tok in self.tokens:
            total += sys.getsizeof(tok)
        return total
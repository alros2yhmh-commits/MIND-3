"""Parse Markdown Knowledge Records into structured knowledge units.

Keeps three notions distinct (per the proposal):
  * confidence in the truth of a claim
  * reliability of the source
  * the user's estimated skill level
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class KnowledgeUnit:
    id: str
    claim: str = ""
    category: str = ""
    confidence: float = 1.0
    source: str = ""
    verification: str = ""
    answer_ar: str = ""
    answer_en: str = ""
    questions: list[str] = field(default_factory=list)
    conflict: str = ""
    note: str = ""


def parse_record(path: str) -> KnowledgeUnit:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    def get(name: str, default: str = "") -> str:
        return "\n".join(sections.get(name, [])).strip() or default

    questions = [q.strip() for q in sections.get("Questions", []) if q.strip()]
    stem = os.path.splitext(os.path.basename(path))[0]
    return KnowledgeUnit(
        id=get("ID", stem),
        claim=get("Claim"),
        category=get("Category"),
        confidence=float(get("Confidence", "1.0")),
        source=get("Source"),
        verification=get("Verification"),
        answer_ar=get("Answer AR"),
        answer_en=get("Answer EN"),
        questions=questions,
        conflict=get("Conflict"),
        note=get("Note"),
    )


def load_records(dirpath: str) -> list[KnowledgeUnit]:
    return [
        parse_record(os.path.join(dirpath, fn))
        for fn in sorted(os.listdir(dirpath))
        if fn.endswith(".md")
    ]


def load_negatives(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def unknown_unit() -> KnowledgeUnit:
    return KnowledgeUnit(
        id="unknown",
        claim="The requested information is not present in the trained knowledge.",
        category="Meta",
        confidence=1.0,
        source="System meta-unit",
        verification="Admit missing knowledge; offer explicit search",
        answer_ar="لا أملك هذه المعلومة ضمن معرفتي المدرّبة. يمكنني البحث عنها إذا طلبت ذلك صراحةً.",
        answer_en="I don't have this information in my trained knowledge. I can search for it if you explicitly ask.",
    )
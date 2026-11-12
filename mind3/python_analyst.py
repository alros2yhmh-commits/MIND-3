"""Evidence-grounded Python concept answering for the local MIND-3 UI.

This is deliberately a bounded concept resolver, not a general-purpose LLM.
It answers only when curated bilingual concepts match sufficiently.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z_]+|[\u0600-\u06FF]+", re.I)
STOP = {
    "what", "is", "the", "a", "an", "in", "python", "explain", "please",
    "tell", "me", "about", "how", "does", "do", "work", "can", "you",
    "اشرح", "ما", "هو", "هي", "في", "من", "عن", "يعني", "لي", "ماهو",
    "بايثون", "بايثون؟", "ممكن", "عايز", "عاوز", "ايه", "إيه",
}


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOP}


class PythonAnalyst:
    def __init__(self, concepts_path: str, router_path: str | None = None) -> None:
        self.concepts = []
        self.router = None
        if router_path and os.path.isfile(router_path):
            from .intent_router import IntentRouter
            self.router = IntentRouter()
            self.router.load(router_path)
        with open(concepts_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.concepts.append(json.loads(line))

    def _score(self, query: str, concept: dict) -> float:
        q = tokens(query)
        aliases = concept.get("aliases", {})
        candidates = aliases.get("en", []) + aliases.get("ar", [])
        candidates += [concept.get("concept", "")]
        best = 0.0
        for alias in candidates:
            a = tokens(alias)
            if not a:
                continue
            overlap = len(q & a)
            score = overlap / max(len(a), 1)
            if alias.lower() in query.lower():
                score = max(score, 1.0)
            best = max(best, score)
        return best

    def answer(self, message: str):
        query = message.strip()
        if not query:
            return None
        ranked = sorted(
            ((self._score(query, c), c) for c in self.concepts),
            key=lambda item: item[0], reverse=True,
        )
        if not ranked or ranked[0][0] < 0.5:
            return None

        router_result = self.router.predict(message) if self.router is not None else None
        learned_intent = (
            router_result["intent"]
            if router_result and router_result["router_score"] >= 0.60 else None
        )
        requested_ar = bool(re.search(r"بالعربية|بالعربي|in arabic|in Arabic", query, re.I))
        requested_ar = requested_ar or learned_intent == "translate_explain"
        query_ar = bool(re.search(r"[\u0600-\u06FF]", query))
        lang = "ar" if requested_ar or query_ar else "en"

        compare = bool(re.search(r"\b(compare|difference|versus|vs)\b|الفرق|قارن", query, re.I))
        compare = compare or learned_intent == "compare_concepts"
        selected = [c for score, c in ranked if score >= 0.5][:3] if compare else [ranked[0][1]]
        if compare and len(selected) < 2:
            return None

        if compare:
            if lang == "ar":
                reply = "مقارنة بناءً على المعرفة الموثقة:\n" + "\n".join(
                    f"- {c['concept']}: {c['claim_ar']}" for c in selected
                )
            else:
                reply = "Comparison from the curated knowledge records:\n" + "\n".join(
                    f"- {c['concept']}: {c['claim_en']}" for c in selected
                )
            operation = "compare_supported_claims"
        else:
            c = selected[0]
            claim = c["claim_ar"] if lang == "ar" else c["claim_en"]
            example = c.get("example", "")
            reply = claim + (("\n\nExample:\n" if lang == "en" else "\n\nمثال:\n") + example if example else "")
            operation = "retrieve_source_linked_concept"

        evidence = [
            {"claim_id": c["id"], "concept": c["concept"],
             "source": c["source"], "verification": c["verification"]}
            for c in selected
        ]
        return {
            "mode": "analytical",
            "lang": lang,
            "reply": reply,
            "sources": [],
            "available": True,
            "unit": "python_analytical",
            "diagnostics": {
                "chosen_unit": "python_analytical",
                "reason": f"{operation}; matched {len(selected)} curated concept record(s)",
                "events": ["No external search performed", "No unsupported claim generated"],
                "paths": [],
                "conflict": None,
                "source": "; ".join(e["source"] for e in evidence),
                "evidence": evidence,
                "matching_scores": [
                    {"concept": c["concept"], "score": round(score, 3)}
                    for score, c in ranked[:5]
                ],
                "learned_intent_router": router_result,
            },
        }

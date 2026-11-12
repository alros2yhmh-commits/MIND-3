"""Orchestration: load the trained model, route knowledge vs on-demand search,
and expose transition diagnostics. Search is imported lazily (Stage 6)."""
from __future__ import annotations

import json
import os
import re

from . import MODEL_NAME, __version__
from .knowledge import KnowledgeUnit
from .model import Mind3Model
from .registry import SymbolRegistry
from .python_analyst import PythonAnalyst

ARABIC_RE = re.compile(r"[؀-ۿ]")
SEARCH_AR = re.compile(r"^\s*(ابحث عن|ابحث لي|ابحث|بحث)\b")
SEARCH_EN = re.compile(r"^\s*(search for|search|look up|google)\b", re.IGNORECASE)


def detect_lang(text: str) -> str:
    ar = len(ARABIC_RE.findall(text))
    en = sum(c.isascii() and c.isalpha() for c in text)
    return "ar" if ar >= en else "en"


def is_search_request(text: str) -> bool:
    return bool(SEARCH_AR.match(text) or SEARCH_EN.match(text))


def strip_search_prefix(text: str) -> str:
    for pat in (SEARCH_AR, SEARCH_EN):
        m = pat.match(text)
        if m:
            return text[m.end():].strip(" :،?\t")
    return text


class Engine:
    def __init__(self, artifacts_dir: str) -> None:
        with open(os.path.join(artifacts_dir, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        with open(os.path.join(artifacts_dir, "units.json"), encoding="utf-8") as f:
            self.units = [KnowledgeUnit(**d) for d in json.load(f)]
        self.registry = SymbolRegistry.load(os.path.join(artifacts_dir, "registry.json"))
        flags = {k: self.config[k] for k in ("use_reference_ids", "use_transitions", "use_confidence")}
        self.model = Mind3Model(
            self.units, registry=self.registry, dim=self.config["dim"],
            seed=self.config["seed"], margin=self.config["margin"],
            beam_k=self.config["beam_k"],
            support_threshold=self.config.get("support_threshold", 0.5), **flags,
        )
        self.model.load_weights(os.path.join(artifacts_dir, "model.npz"))
        self.history: list[dict] = []
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(artifacts_dir)))
        concepts_path = os.path.join(project_root, "data", "curated", "python_concepts.jsonl")
        router_path = os.path.join(project_root, "artifacts", "analytical", "intent_router.npz")
        self.python_analyst = (
            PythonAnalyst(concepts_path, router_path=router_path)
            if os.path.isfile(concepts_path) else None
        )

    def answer(self, message: str) -> dict:
        lang = detect_lang(message)
        if is_search_request(message):
            from .search import search
            query = strip_search_prefix(message)
            result = search(query, lang)
            self.history.append({"role": "user", "text": message})
            self.history.append({"role": "assistant", "text": result["summary"]})
            return {"mode": "search", "lang": lang, "reply": result["summary"],
                    "sources": result["sources"], "available": result["available"],
                    "diagnostics": None}

        if self.python_analyst is not None:
            analytical = self.python_analyst.answer(message)
            if analytical is not None:
                self.history.append({"role": "user", "text": message})
                self.history.append({"role": "assistant", "text": analytical["reply"]})
                return analytical

        res = self.model.infer(message)
        chosen = res["chosen"]
        u = self.units[chosen.unit_index]
        reply = (u.answer_ar if lang == "ar" else u.answer_en) or u.answer_en or u.answer_ar
        diag = {
            "chosen_unit": u.id,
            "reason": res["reason"],
            "events": res["events"],
            "paths": [{"unit": p.unit_id, "model_prob": round(p.model_prob, 4),
                       "confidence": p.confidence, "verified": p.verified,
                       "final_score": round(p.final_score, 4)} for p in res["paths"]],
            "conflict": u.conflict or None,
            "source": u.source,
        }
        self.history.append({"role": "user", "text": message})
        self.history.append({"role": "assistant", "text": reply})
        return {"mode": "knowledge", "lang": lang, "reply": reply, "sources": [],
                "available": True, "diagnostics": diag, "unit": u.id}

    def clear(self) -> None:
        self.history = []

    def info(self) -> dict:
        return {"model_name": MODEL_NAME, "version": __version__,
                "variant": self.config["variant"], "units": len(self.units),
                "vocab": self.registry.size,
                "analytical_concepts": len(self.python_analyst.concepts) if self.python_analyst else 0}
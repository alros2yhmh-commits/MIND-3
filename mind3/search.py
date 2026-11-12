"""Stage 6 — On-demand search only.

Uses free, verified endpoints (no API key): Wikipedia Search API + DuckDuckGo
Instant Answer API, via the standard library (urllib). Never invents sources.
If no tool/results are available, it says so explicitly.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

UA = {"User-Agent": "MIND-3-experiment/0.1 (local research; contact: local)"}
TAG_RE = re.compile(r"<[^>]+>")
TIMEOUT = 8


def _get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _clean(text: str) -> str:
    return TAG_RE.sub("", text).replace("&quot;", '"').replace("&amp;", "&").strip()


def _wikipedia(query: str, lang: str) -> list[dict]:
    host = f"{lang}.wikipedia.org"
    url = (f"https://{host}/w/api.php?action=query&list=search&srsearch="
           f"{urllib.parse.quote(query)}&format=json&srlimit=3")
    data = _get_json(url)
    out = []
    for item in data.get("query", {}).get("search", []):
        out.append({
            "title": item.get("title", ""),
            "url": f"https://{host}/?curid={item.get('pageid', '')}",
            "snippet": _clean(item.get("snippet", "")),
            "provider": "Wikipedia",
        })
    return out


def _duckduckgo(query: str) -> list[dict]:
    url = (f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}"
           f"&format=json&no_html=1")
    data = _get_json(url)
    out = []
    abstract = data.get("Abstract") or data.get("AbstractText")
    if abstract:
        out.append({
            "title": data.get("Heading") or query,
            "url": data.get("AbstractURL") or f"https://duckduckgo.com/?q={urllib.parse.quote(query)}",
            "snippet": _clean(abstract),
            "provider": "DuckDuckGo",
        })
    return out


def search(query: str, lang: str = "en") -> dict:
    query = query.strip()
    sources: list[dict] = []
    errors: list[str] = []
    for fn, arg in ((_wikipedia, (query, lang)), (_duckduckgo, (query,))):
        try:
            sources.extend(fn(*arg))
        except Exception as exc:  # network/parse failure -> record, don't invent
            errors.append(f"{fn.__name__}: {exc}")

    available = len(sources) > 0
    if lang == "ar":
        if available:
            lines = [f"نتائج بحث حديثة عن «{query}» (ليست من معرفة النموذج المدرّبة):"]
            for i, s in enumerate(sources, 1):
                lines.append(f"{i}. {s['title']} — {s['snippet']}")
            summary = "\n".join(lines)
        else:
            summary = "تعذّر تنفيذ البحث: لا توجد أداة بحث متاحة الآن أو لا توجد نتائج."
    else:
        if available:
            lines = [f"Fresh search results for '{query}' (not from the model's trained knowledge):"]
            for i, s in enumerate(sources, 1):
                lines.append(f"{i}. {s['title']} — {s['snippet']}")
            summary = "\n".join(lines)
        else:
            summary = "Search could not be executed: no search tool available right now or no results."

    return {"summary": summary, "sources": sources, "available": available,
            "errors": errors, "query": query}
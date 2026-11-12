"""Download and prepare Hugging Face Python datasets reproducibly.

Install: python -m pip install datasets
Run:     python -m scripts.prepare_hf_data

The script never executes downloaded code. AST parsing checks syntax only.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
MAX_ROWS_PER_SOURCE = int(os.environ.get("MIND3_MAX_ROWS_PER_SOURCE", "20000"))
MAX_TEXT_CHARS = 12000
FENCE = chr(96) * 3


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def extract_code(text: str) -> str:
    blocks = re.findall(re.escape(FENCE) + r"(?:python)?\s*(.*?)" + re.escape(FENCE),
                        text, flags=re.I | re.S)
    return max(blocks, key=len) if blocks else text


def syntax_ok(text: str) -> bool:
    if not text or len(text) > MAX_TEXT_CHARS:
        return False
    candidate = extract_code(text)
    if not re.search(r"(^|\n)\s*(def |class |import |from |if |for |while |try:|with )",
                     candidate):
        return False
    try:
        ast.parse(candidate)
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalise_pair(prompt: Any, response: Any, source: str, row_id: str):
    p, r = clean(prompt), str(response or "").strip()
    if len(p) < 12 or len(r) < 12 or len(p) > MAX_TEXT_CHARS or len(r) > MAX_TEXT_CHARS:
        return None
    if not syntax_ok(r):
        return None
    return {
        "id": f"{source}:{row_id}", "source": source,
        "prompt": p, "response": r, "language": "en",
        "provenance": "community_or_generated_instruction",
        "verification": "syntax_checked_only",
        "sha256": digest(p + "\n" + r),
    }


def load_instructional_code_search_net(limit: int):
    repo_id = "Nan-Do/instructional_code-search-net-python"
    revision = HfApi().dataset_info(repo_id).sha
    ds = load_dataset(repo_id, split="train", revision=revision, streaming=True)
    rows = []
    for i, item in enumerate(ds):
        if len(rows) >= limit:
            break
        row = normalise_pair(item.get("INSTRUCTION"), item.get("RESPONSE"),
                             "Nan-Do/instructional_code-search-net-python", str(i))
        if row:
            rows.append(row)
    return rows, revision


def load_mbpp_splits():
    # This mirror is stored as Parquet, so it works with modern datasets
    # versions that no longer execute legacy dataset loading scripts.
    repo_id = "huggy-1/mbpp"
    config = "full"
    revision = HfApi().dataset_info(repo_id).sha

    def read_split(split_name: str):
        dataset = load_dataset(
            repo_id, name=config, split=split_name, revision=revision
        )
        rows = []
        seen_ids = set()
        for item in dataset:
            task_id = int(item["task_id"])
            if task_id in seen_ids:
                raise ValueError(f"Duplicate MBPP task_id {task_id} in {split_name}")
            seen_ids.add(task_id)
            prompt = str(item.get("text", "")).strip()
            code = str(item.get("code", "")).strip()
            if not prompt or not code:
                raise ValueError(
                    f"MBPP task {task_id} in {split_name} has an empty prompt/code"
                )
            rows.append({
                "id": f"mbpp:{task_id}", "source": repo_id,
                "prompt": prompt, "response": code,
                "tests": list(item.get("test_list", []) or []),
                "language": "en", "provenance": "crowdsourced_benchmark",
                "verification": "reference_solution_with_tests_not_independently_proven",
            })
        return rows

    # Use the dataset's published splits; do not infer split membership from IDs.
    train = read_split("train")
    valid = read_split("validation")
    test = read_split("test")
    prompt = read_split("prompt")

    split_ids = [
        {row["id"] for row in rows}
        for rows in (train, valid, test, prompt)
    ]
    for i, ids in enumerate(split_ids):
        for other in split_ids[i + 1:]:
            overlap = ids & other
            if overlap:
                raise ValueError(
                    "MBPP split leakage detected: " + ", ".join(sorted(overlap)[:5])
                )

    return train, valid, test, prompt, revision


def deduplicate(rows):
    seen, out = set(), []
    for row in rows:
        key = digest(clean(row["prompt"]).lower() + "\n" + clean(row["response"]).lower())
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def main():
    print("Loading Hugging Face Python instruction datasets...")
    rows = []
    rows, codesearchnet_revision = load_instructional_code_search_net(MAX_ROWS_PER_SOURCE)
    rows = deduplicate(rows)
    mbpp_train, mbpp_valid, mbpp_test, mbpp_prompt, mbpp_revision = load_mbpp_splits()

    write_jsonl(OUT / "python_train.jsonl", rows)
    write_jsonl(OUT / "mbpp_train_reference.jsonl", mbpp_train)
    write_jsonl(OUT / "mbpp_validation.jsonl", mbpp_valid)
    write_jsonl(OUT / "mbpp_test.jsonl", mbpp_test)
    write_jsonl(OUT / "mbpp_prompt_reference.jsonl", mbpp_prompt)

    report = {
        "instruction_rows": len(rows),
        "source_revisions": {
            "Nan-Do/instructional_code-search-net-python": codesearchnet_revision,
            "huggy-1/mbpp": mbpp_revision,
        },
        "mbpp_train_reference_rows": len(mbpp_train),
        "mbpp_validation_rows": len(mbpp_valid),
        "mbpp_test_rows": len(mbpp_test),
        "mbpp_prompt_reference_rows": len(mbpp_prompt),
        "intent_labels": {label: sum(r["intent"] == label for r in rows)
                          for label in sorted({r["intent"] for r in rows})},
        "notes": [
            "AST parsing checks syntax only; it does not prove semantic correctness.",
            "Downloaded source code was never executed.",
            "MBPP uses its published Parquet train/validation/test/prompt splits; split IDs are checked for overlap.",
            "MBPP test rows are held out and are not used to train the intent router.",
            "The prompt split is preserved separately and is not mixed into training.",
            "Arabic Python concept seeds are stored separately in data/curated/python_concepts.jsonl.",
            "CodeSearchNet prompts are retained as code-task corpus only; they do not receive intent labels and are excluded from intent-router supervision."
        ]
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

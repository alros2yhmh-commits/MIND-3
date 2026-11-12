# MIND-3 — Analytical Python Research Track

This branch adds a bounded evidence-aware Python knowledge path alongside the original experimental classifier. It is a research prototype, not a general LLM.

## Architecture

- **Curated knowledge**: bilingual Python concept claims in data/curated/python_concepts.jsonl, each linked to official Python docs.
- **Analytical engine**: explicit premises, declared rules, conflict detection, and abstention when required evidence is missing.
- **Learned intent router**: a compact neural classifier with **49,928 trainable parameters** (512 token buckets × 96 embedding dimensions + 96 × 8 output weights + 8 biases). It predicts a task family; it does not generate facts.
- **Hugging Face data pipeline**: downloads selected datasets locally, filters malformed rows, syntax-checks code with Python AST, deduplicates, and writes reproducible JSONL artifacts.
- **UI integration**: supported Python concepts are answered through the evidence-backed path and show their documentation sources in diagnostics. Other requests continue to the original MIND-3 path.

## Dataset sources and limitations

See data/hf_sources.json for licenses, intended roles, and quality notes.

The default pipeline uses:
- Nan-Do/instructional_code-search-net-python (Apache-2.0)
- Muennighoff/mbpp (CC-BY-4.0), split by task ID and held out for validation/test

Flytech/python-codes-25k was inspected but excluded from the default pipeline because visible preview rows contained malformed Python quoting/syntax and the card/viewer row counts were inconsistent. It remains documented as an excluded candidate in data/hf_sources.json.

Important:
- Dataset cards and licenses were inspected, but that does not make every row correct.
- CodeSearchNet-derived instruction pairs include template/NLP-generated text.
- Flytech is a cleaned dataset, not a formally verified textbook.
- AST parsing proves syntax validity only. The preparation script never executes downloaded code.
- MBPP test tasks (IDs 11–510) are not mixed into the instruction training set.
- Arabic Python facts are curated bilingual records linked to official Python documentation. General Arabic translation corpora are not treated as proof of Python facts.
- Raw corpora are not committed to GitHub. They are downloaded on the user's machine into data/processed/.

## Windows setup

Run from PowerShell:

```powershell
cd "C:\Users\Hamza\Desktop\MIND-3"

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-research.txt
```

If PowerShell blocks activation, use the interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
```

## Prepare datasets

```powershell
python -m scripts.prepare_hf_data
```

Optional cap for a smaller first run:

```powershell
$env:MIND3_MAX_ROWS_PER_SOURCE="5000"
python -m scripts.prepare_hf_data
```

The full default cap is 20,000 accepted rows per instructional source. Hugging Face downloads can be large; the CodeSearchNet-derived dataset is hundreds of megabytes. Keep generated data/processed/ local.

## Train the 49,928-parameter router

```powershell
python -m scripts.train_router
```

The router is trained using weak intent labels inferred from prompt wording, plus curated Arabic/English seed examples. Its training accuracy is not a measure of Python knowledge or reasoning ability. Inspect artifacts/analytical/intent_router_report.json.

## Tests

```powershell
python -m tests.test_analytical
python -m tests.test_python_analyst
python -m tests.test_model
python -m tests.test_engine
```

## Launch the local UI

```powershell
python server.py
```

Open http://127.0.0.1:8765.

The UI will display an evidence-backed analysis tag for supported curated Python concepts. The learned router's prediction appears in the diagnostics payload. The router is a task classifier, not a language generator; the analytical answer is grounded in curated claims and deliberately limited to concepts represented in the knowledge file.

## What this experiment can and cannot establish

It can test:
- whether a small learned router can distinguish broad task intents;
- whether bilingual curated claims can answer paraphrased concept questions;
- whether declared rules derive conclusions from explicit premises;
- whether missing/conflicting premises cause abstention.

It cannot yet establish:
- general Arabic-English understanding;
- general Python code generation;
- semantic correctness of every downloaded code example;
- a universal 100% answer-correctness guarantee.

A valid next experiment should compare the same held-out prompts against the original classifier and analytical path, report per-intent accuracy, unsupported-answer rate, conflict-detection rate, and router calibration.

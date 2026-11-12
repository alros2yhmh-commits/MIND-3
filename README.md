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
- huggy-1/mbpp (CC-BY-4.0), a Parquet mirror with published train/validation/test/prompt splits

Flytech/python-codes-25k was inspected but excluded from the default pipeline because visible preview rows contained malformed Python quoting/syntax and the card/viewer row counts were inconsistent. It remains documented as an excluded candidate in data/hf_sources.json.

Important:
- Dataset cards, licenses, and visible sample rows were inspected, but the full corpora were not manually audited row by row.
- CodeSearchNet-derived instruction pairs include template/NLP-generated text and are excluded from intent-router supervision.
- The Flytech candidate is excluded from the default pipeline after inspecting visible malformed examples.
- AST parsing proves syntax validity only. The preparation script never executes downloaded code.
- MBPP uses its published Parquet splits (374 train, 90 validation, 500 test, 10 prompt); the preparation script checks split IDs for overlap. Test rows remain held out, and prompt rows are stored separately.
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

## Intent data v2 and router training

The intent router is supervised only by the manually authored bilingual files `data/curated/intent_train.jsonl` and `data/curated/intent_eval.jsonl`. Each of the eight intents has equal representation in both splits, with Arabic and English examples. The evaluation prompts are distinct from training prompts and are never used for gradient updates.

The CodeSearchNet-derived instruction corpus remains available for code-task research, but it receives no heuristic intent labels and is excluded from router supervision. MBPP published splits remain separate; its test split is not used for router training.

Run:

```powershell
python -m scripts.train_router
```

The script validates labels, duplicate prompts, language coverage, exact class balance, and train/evaluation text separation before training. It reports overall and per-intent holdout accuracy plus a confusion matrix. The initial holdout is small and hand-authored: treat it as a diagnostic, not a statistically strong benchmark. The router's softmax score is uncalibrated and the classifier does not measure Python knowledge or reasoning.

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


## Intent data v3

The v3 intent-router experiment uses three newly authored, disjoint bilingual splits:

- `data/curated/intent_train_v3.jsonl`: 320 prompts (40 per intent; 20 Arabic and 20 English).
- `data/curated/intent_dev_v3.jsonl`: 80 prompts (10 per intent; 5 Arabic and 5 English).
- `data/curated/intent_test_v3.jsonl`: 64 prompts (8 per intent; 4 Arabic and 4 English).

Each row records its authoring method, reviewer, and review status. The separate 16-row boundary-review set remains a boundary suite and is not merged into these splits. The older v2 files remain unchanged; they are not copied wholesale because several labels do not satisfy the v3 taxonomy.

The v3 prompts are assistant-authored and taxonomy-reviewed, not independently human-annotated or sourced from an externally verified intent corpus. Treat this as a controlled first v3 experiment, not a claim of broad real-world coverage. The splits are balanced and have no exact normalized-text duplicates; semantic similarity and edge cases still warrant further review.

Validate the complete v3 dataset first, then train using only train and inspect dev metrics:

```powershell
python -m scripts.validate_intent_v3
python -m scripts.train_router_v3
```

The training script deliberately does not open the final test file. After training and freezing the model/settings, evaluate the final test once:

```powershell
python -m scripts.evaluate_router_v3
```

The final test must not be used to tune epochs, learning rate, taxonomy, or data. If test results trigger changes, create a new test set before another final evaluation. The original v2 training script and data remain available for comparison.

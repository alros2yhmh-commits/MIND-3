# MIND-3 Intent Taxonomy v3

Status: proposed labeling policy. This file documents annotation rules; it does not silently relabel v2 data.

## Label the requested operation, not isolated words

Read the whole request and identify the action the user expects MIND-3 to perform. Words such as "calculate", "difference", "output", "fix", or "explain" are not sufficient by themselves.

## Labels

1. `explain_concept`: Explain or define a Python concept, syntax feature, API behavior, or programming principle. The user wants understanding, not a pairwise comparison or a concrete execution result.
2. `write_function`: Create new Python code, a function, script, or implementation from a specification.
3. `debug_code`: Diagnose or repair existing code that is broken, incorrect, or raises an unintended error. A request to explain a correct snippet is not debugging.
4. `trace_code`: Simulate a supplied or described program's execution and report concrete runtime state: printed output, variable values, branches, iterations, or return value. Use this even when the requested result is numeric.
5. `compare_concepts`: Explicitly compare two or more concepts, approaches, APIs, or alternatives, focusing on differences, trade-offs, or similarities.
6. `translate_explain`: Translate or render supplied technical content into a requested language, optionally explaining it while preserving identifiers and code.
7. `calculate`: Perform a standalone mathematical or numerical calculation that is not asking to simulate a Python program. A numeric answer about a code snippet or its execution belongs to `trace_code`.
8. `general_python`: Broad Python guidance, learning advice, project planning, tooling, or topic recommendations that do not request one of the specific operations above.

## Decision order for overlapping requests

Apply the first clearly satisfied operation:
- Existing broken code + diagnose/fix -> `debug_code`.
- New implementation requested -> `write_function`.
- Supplied/described code + exact execution result/state/output -> `trace_code`.
- Explicit pairwise/multi-way comparison -> `compare_concepts`.
- Translation into a target language -> `translate_explain`.
- Standalone arithmetic/numerical task -> `calculate`.
- Concept definition or explanation -> `explain_concept`.
- Broad advice/planning/tooling -> `general_python`.

If a request genuinely asks for two independent operations, annotate the primary requested deliverable and record the secondary intent in a future multi-label field; do not force a single label based on one keyword. If the primary intent cannot be determined, mark it for human review instead of guessing.

## Boundary examples

- "What does this loop do?" (conceptual explanation) -> `explain_concept`.
- "What value does x have after this loop?" (runtime state) -> `trace_code`.
- "How many times does this supplied loop run?" -> `trace_code`.
- "What is 17 * 24?" -> `calculate`.
- "Fix the loop because it skips the last item." -> `debug_code`.
- "Write a loop that skips invalid items." -> `write_function`.
- "Explain the difference between list and tuple." -> `compare_concepts`, not `explain_concept`.
- "Explain what a tuple is." -> `explain_concept`.

## Dataset handling rules

- v2 training and holdout files remain immutable until reviewed against this policy.
- Boundary-review examples are not training data and must not be silently merged into training.
- Keep development/validation and final test sets distinct. Do not tune on the final test set.
- Track source, authoring method, language, reviewer, and review status for every future row.
- Balance by intent only after semantic validity is checked. Do not duplicate or paraphrase near-identical prompts just to hit a count.
- Report per-intent and per-language metrics. Overall accuracy alone can hide language-specific failures.

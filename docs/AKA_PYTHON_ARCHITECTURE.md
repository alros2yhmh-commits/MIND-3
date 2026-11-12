# MIND-3 / AKA — Analytical Knowledge Architecture

**Status:** architecture contract for the Python research track  
**Version:** AKA-PY-1  
**Training policy:** MIND-3 is trained from random initialization. No pretrained language model, external LLM, fine-tuning, or hidden hosted inference is part of this design.

## 1. Purpose

MIND-3 is a fixed neuro-symbolic analytical system, not an autonomous agent and not a wrapper around a generative model. Its objective is to learn representations and relations from Python data, store explicit knowledge with provenance, and answer *new combinations or deductions* when the required premises and rules are available—even when the final answer was not itself a training target.

Generalization means deriving an answer from learned/loaded facts and declared operations. It does **not** mean inventing unsupported facts or guaranteeing arbitrary free-form code generation.

## 2. Frozen data flow

`Input Tokens → Semantic & Intent Mapper → Knowledge Graph / Claims → Analytical Engine → Evidence & Confidence Evaluator (EAS) → Answer / Qualify / Conflict / Abstain`

The path is deterministic at the orchestration level. Learned components propose semantic mappings; the analytical and evidence layers decide whether a conclusion is justified.

### Stage A — Input Tokens

- Normalize text without discarding source language or code boundaries.
- Preserve identifiers, operators, punctuation, indentation, and code spans.
- Produce token/reference IDs using the project's reference-ID representation.

### Stage B — Semantic & Intent Mapper

- A MIND-3-owned neural component trained from random initialization.
- Maps input tokens to candidate concepts, entities, predicates, relation types, and task families.
- The existing v3 Intent Router is an auxiliary baseline only. It is **not** the MIND-3 semantic mapper and its accuracy is not a MIND-3 knowledge score.
- Parameter experiments are tracked separately at approximately 10K, 50K, and 100K trainable parameters. Record the exact count, seed, vocabulary, split hashes, and metrics for every run.
- No answer text is memorized as the definition of a fact. Training targets supervise mappings/relations; the graph stores the evidence.

### Stage C — Knowledge Graph / Claims

A claim has at least:
- `claim_id`
- `subject`
- `predicate`
- `value`
- `source`
- `source_reliability`
- `verification`
- `language`
- `status`

Relations are explicit directed edges with a relation type and provenance. Python AST extraction may create structural facts such as `defines`, `imports`, `calls`, `contains_construct`, and `mentions_concept`.

Do not upgrade syntax validity into semantic correctness:
- CodeSearchNet-derived pairs are `syntax_checked_only`.
- MBPP reference solutions are benchmark references; their tests are not executed by the preparation or graph-building pipeline.
- Curated concept records link to Python documentation, but a URL is not independent verification of every paraphrase.

The graph builder must preserve source IDs and verification labels. It must never execute downloaded code.

### Stage D — Analytical Engine

- Uses explicit premises and declared operators/rules.
- A rule has the form `R: (A1 ∧ A2 ∧ ... ∧ An) ⇒ B`.
- Supported operator families include relation traversal, comparison, set operations, arithmetic, and bounded logical composition.
- Every derived result retains its premises and rule ID.
- Missing premises, ambiguous entities, and contradictory values stop or qualify the derivation; they are not silently guessed.

### Stage E — Evidence & Confidence Evaluator (EAS)

EAS keeps **source reliability** separate from **inference validity**. The research score is:

`S(B) = w1·D + w2·P + w3·V − w4·X − w5·M`

where:
- `D`: direct evidence strength;
- `P`: required-premise coverage;
- `V`: verification quality;
- `X`: contradiction penalty;
- `M`: missing-premise penalty.

Weights and decision thresholds must be versioned and calibrated on development data, not guessed from final-test results. EAS decisions are:
- `ANSWER`: evidence and all required premises meet the frozen threshold;
- `QUALIFY`: evidence is partial or lower-confidence and the limitation can be stated;
- `CONFLICT`: incompatible supported claims exist;
- `ABSTAIN`: required evidence is missing or unsupported.

A high neural score alone must never force `ANSWER`.

## 3. Python data integration

The knowledge builder consumes only:
- `data/processed/python_train.jsonl` (CodeSearchNet-derived instruction/code pairs);
- `data/processed/mbpp_train_reference.jsonl`;
- `data/processed/mbpp_validation.jsonl` (development evaluation only);
- `data/curated/python_concepts.jsonl`.

It must not read `mbpp_test.jsonl` or `mbpp_prompt_reference.jsonl`. Those remain untouched and outside training and development.

The builder emits:
- `python_train_examples.jsonl`;
- `python_validation_examples.jsonl`;
- `knowledge_nodes.jsonl`;
- `knowledge_edges.jsonl`;
- `knowledge_manifest.json`.

AST-derived edges are structural observations, not proof that a code example is correct. Concept links based on literal aliases are lexical links, not deep semantic verification.

## 4. Training protocol

1. Build the graph and validate schema/provenance.
2. Freeze train/development inputs and hashes.
3. Train the MIND-3 semantic mapping component from random initialization.
4. Evaluate on MBPP validation and a separate unseen-composition suite.
5. Freeze parameters and thresholds before opening MBPP test.
6. Report per-family metrics, macro accuracy, abstention, conflict handling, and **False Acceptance Rate (FAR)**.
7. Run 20 inference problems whose answers must be derived from 10–15 supplied facts; do not train on those answers.

Required test families:
- direct fact;
- paraphrase;
- reordered facts;
- Arabic/English;
- compound/multi-hop;
- comparison;
- calculated result;
- missing premise;
- conflicting premise;
- out-of-knowledge request.

## 5. Acceptance criteria

- Every answer has a traceable evidence path or an explicit qualification.
- No unsupported claim is emitted as verified.
- Missing and conflicting evidence produce the correct EAS decision.
- The final test split remains unopened until the model and thresholds are frozen.
- Training accuracy alone is never treated as generalization evidence.

## 6. Current implementation boundary

The repository already contains partial primitives: reference-ID encoding, a transition system, an analytical claim/rule engine, curated Python concept records, and the v3 intent-router experiment. These are not yet the complete AKA pipeline.

The Python graph builder and first neural integration experiment are implementation steps toward this contract. They must be labelled as prototypes where their behavior is still retrieval/classification rather than novel symbolic derivation. In particular, a retrieved code example is a reference pattern, not a newly synthesized or verified solution.

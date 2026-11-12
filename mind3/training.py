"""H3 — Confidence-Weighted Markdown Training pipeline.

Markdown records become labeled examples; confidence weights the training loss.
The reference-ID encoder is position-aware, so its embedding gradients must be
mapped back through the inverse position rotation.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np

from . import EXPERIMENT, MODEL_NAME, __version__
from .knowledge import load_negatives, load_records, unknown_unit
from .model import Mind3Model
from .registry import SymbolRegistry
from .transitions import softmax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOW = os.path.join(ROOT, "knowledge")

VARIANTS = {
    "full": dict(use_reference_ids=True, use_transitions=True, use_confidence=True),
    "no_reference": dict(use_reference_ids=False, use_transitions=True, use_confidence=True),
    "no_transitions": dict(use_reference_ids=True, use_transitions=False, use_confidence=True),
    "no_confidence": dict(use_reference_ids=True, use_transitions=True, use_confidence=False),
    "baseline": dict(use_reference_ids=False, use_transitions=False, use_confidence=False),
}


def build_dataset():
    units = load_records(os.path.join(KNOW, "records")) + [unknown_unit()]
    negatives = load_negatives(os.path.join(KNOW, "negative_questions.txt"))
    idx = {u.id: i for i, u in enumerate(units)}
    texts, examples = [], []
    for u in units:
        for q in u.questions:
            examples.append((q, idx[u.id], float(u.confidence)))
            texts.append(q)
    for q in negatives:
        examples.append((q, idx["unknown"], 1.0))
        texts.append(q)
    registry = SymbolRegistry().fit(texts)
    return units, registry, examples, negatives


def train_model(model, examples, epochs=800, lr=0.05, log_every=100):
    texts = [e[0] for e in examples]
    ys = np.array([e[1] for e in examples])
    ws = np.array([e[2] for e in examples], dtype=float)
    ws = ws / ws.mean() if model.use_confidence else np.ones_like(ws)

    fixed_C = None if model.use_reference_ids else np.stack([model.features(t) for t in texts])
    params = [model.W, model.b] + ([model.E] if model.use_reference_ids else [])
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    log = []
    n = len(ys)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        C = np.stack([model.features(t) for t in texts]) if model.use_reference_ids else fixed_C
        probs = softmax(C @ model.W + model.b)
        loss = -np.mean(ws * np.log(probs[np.arange(n), ys] + 1e-12))
        dlog = probs.copy()
        dlog[np.arange(n), ys] -= 1.0
        dlog *= ws[:, None] / n
        grads = [C.T @ dlog, dlog.sum(axis=0)]
        if model.use_reference_ids:
            dC = dlog @ model.W.T
            gE = np.zeros_like(model.E)
            for i, text in enumerate(texts):
                ids = model.registry.encode(text) or [0]
                scale = 1.0 / len(ids)
                for position, token_id in enumerate(ids):
                    shift = position % model.dim
                    # features() rolls the embedding forward; backpropagate with
                    # the inverse roll to reach the original embedding coordinates.
                    np.add.at(gE, token_id, np.roll(dC[i], -shift) * scale)
            grads.append(gE)
        for j, (p, g) in enumerate(zip(params, grads)):
            m[j] = b1 * m[j] + (1 - b1) * g
            v[j] = b2 * v[j] + (1 - b2) * (g * g)
            mh = m[j] / (1 - b1 ** epoch)
            vh = v[j] / (1 - b2 ** epoch)
            p -= lr * mh / (np.sqrt(vh) + eps)
        if epoch == 1 or epoch % log_every == 0:
            log.append({"epoch": epoch, "loss": round(float(loss), 5),
                        "train_acc": round(float((probs.argmax(1) == ys).mean()), 4)})
    duration = time.perf_counter() - t0
    C = np.stack([model.features(t) for t in texts]) if model.use_reference_ids else fixed_C
    final_probs = softmax(C @ model.W + model.b)
    final_acc = float((final_probs.argmax(1) == ys).mean())
    return {"log": log, "final_train_acc": final_acc, "duration_s": round(duration, 3),
            "epochs": epochs, "examples": n}


def train_variant(name, out_dir, epochs=800, lr=0.05, dim=48, seed=0):
    flags = VARIANTS[name]
    units, registry, examples, negatives = build_dataset()
    model = Mind3Model(units, registry=registry, dim=dim, seed=seed, **flags)
    stats = train_model(model, examples, epochs=epochs, lr=lr)
    model.compute_prototypes(examples)
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.npz"))
    registry.save(os.path.join(out_dir, "registry.json"))
    with open(os.path.join(out_dir, "units.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(u) for u in units], f, ensure_ascii=False, indent=1)
    config = {"model_name": MODEL_NAME, "version": __version__, "experiment": EXPERIMENT,
              "variant": name, "dim": dim, "seed": seed, "margin": 0.30, "beam_k": 4,
              "support_threshold": 0.5, "negative_count": len(negatives),
              "encoder": "position-rotated-reference-embeddings-v2", **flags}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    return config, stats


def main():
    cfg, stats = train_variant("full", os.path.join(ROOT, "artifacts", "full"))
    print(json.dumps({"variant": "full", "final_train_acc": stats["final_train_acc"],
                      "duration_s": stats["duration_s"], "examples": stats["examples"],
                      "loss_curve": stats["log"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

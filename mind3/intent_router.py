"""A compact learned intent router with a declared parameter budget.

This is not a generative language model. It predicts which analytical operation
should handle a query; facts and conclusions remain in the verified knowledge
store and deterministic analytical engine.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

INTENTS = (
    "explain_concept", "write_function", "debug_code", "trace_code",
    "compare_concepts", "translate_explain", "calculate", "general_python",
)
TOKEN_RE = re.compile(r"[A-Za-z_]+|[\u0600-\u06FF]+|\d+|[^\w\s]", re.UNICODE)


def token_bucket(token: str, buckets: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


class IntentRouter:
    def __init__(self, buckets: int = 512, dim: int = 96, seed: int = 17) -> None:
        self.buckets = buckets
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.E = self.rng.normal(0.0, 0.025, (buckets, dim))
        self.W = self.rng.normal(0.0, 0.025, (dim, len(INTENTS)))
        self.b = np.zeros(len(INTENTS), dtype=np.float64)

    @property
    def parameter_count(self) -> int:
        return int(self.E.size + self.W.size + self.b.size)

    def encode(self, text: str) -> list[int]:
        tokens = TOKEN_RE.findall(text.lower())
        return [token_bucket(t, self.buckets) for t in tokens] or [0]

    def features(self, text: str) -> np.ndarray:
        ids = self.encode(text)
        return self.E[ids].mean(axis=0)

    def predict(self, text: str) -> dict:
        x = self.features(text)
        logits = x @ self.W + self.b
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        idx = int(probs.argmax())
        return {
            "intent": INTENTS[idx],
            "probabilities": {name: float(probs[i]) for i, name in enumerate(INTENTS)},
            "router_score": float(probs[idx]),
            "parameter_count": self.parameter_count,
        }

    def train(self, examples: list[tuple[str, str]], epochs: int = 24,
              learning_rate: float = 0.08, seed: int = 23) -> dict:
        rng = np.random.default_rng(seed)
        rows = [(text, INTENTS.index(label)) for text, label in examples
                if label in INTENTS and text.strip()]
        if not rows:
            raise ValueError("No valid intent examples")
        history = []
        for epoch in range(1, epochs + 1):
            rng.shuffle(rows)
            total_loss = 0.0
            correct = 0
            for text, target in rows:
                ids = self.encode(text)
                x = self.E[ids].mean(axis=0)
                logits = x @ self.W + self.b
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                total_loss -= float(np.log(probs[target] + 1e-12))
                correct += int(int(probs.argmax()) == target)
                grad = probs
                grad[target] -= 1.0
                grad_W = np.outer(x, grad)
                grad_b = grad
                grad_x = self.W @ grad
                grad_E = grad_x / len(ids)
                self.W -= learning_rate * grad_W
                self.b -= learning_rate * grad_b
                for token_id in ids:
                    self.E[token_id] -= learning_rate * grad_E
            if epoch == 1 or epoch % 4 == 0 or epoch == epochs:
                history.append({
                    "epoch": epoch,
                    "loss": total_loss / len(rows),
                    "accuracy": correct / len(rows),
                })
        return {"examples": len(rows), "epochs": epochs, "history": history,
                "parameter_count": self.parameter_count}

    def save(self, path: str) -> None:
        np.savez(path, E=self.E, W=self.W, b=self.b,
                 buckets=np.array([self.buckets]), dim=np.array([self.dim]))

    def load(self, path: str) -> None:
        data = np.load(path)
        self.E, self.W, self.b = data["E"], data["W"], data["b"]
        self.buckets, self.dim = int(data["buckets"][0]), int(data["dim"][0])

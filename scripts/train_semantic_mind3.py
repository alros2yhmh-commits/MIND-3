"""Train AKA-PY-2 DistMult graph embeddings from random initialization.

This is a link-prediction experiment over curated facts, not a language model.
Held-out dev triples are used only for ranking evaluation.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
DEFAULT_OUT = ROOT / "artifacts" / "aka_py2"


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(E, R, h, r, t):
    return float(np.sum(E[h] * R[r] * E[t]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--negatives", type=int, default=4)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.dim < 4 or args.epochs < 1 or args.negatives < 1:
        raise ValueError("dim>=4, epochs>=1 and negatives>=1 are required")
    rows = read_jsonl(DATA / "semantic_triples_v2.jsonl")
    train = [x for x in rows if x["split"] == "train"]
    dev = [x for x in rows if x["split"] == "dev"]
    if not train or not dev:
        raise ValueError("Both train and dev triples are required")
    entities = sorted({x["subject"] for x in rows} | {x["object"] for x in rows})
    relations = sorted({x["relation"] for x in rows})
    ei, ri = {x:i for i,x in enumerate(entities)}, {x:i for i,x in enumerate(relations)}
    train_ids = [(ei[x["subject"]], ri[x["relation"]], ei[x["object"]]) for x in train]
    dev_ids = [(ei[x["subject"]], ri[x["relation"]], ei[x["object"]]) for x in dev]
    rng = np.random.default_rng(args.seed)
    E = rng.normal(0, 0.08, (len(entities), args.dim))
    R = rng.normal(0, 0.08, (len(relations), args.dim))
    known = set(train_ids)
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_ids))
        total_loss, wins = 0.0, 0
        for ix in order:
            h, r, t = train_ids[ix]
            for _ in range(args.negatives):
                corrupt_tail = bool(rng.integers(0, 2))
                if corrupt_tail:
                    nt = int(rng.integers(0, len(entities)))
                    tries = 0
                    while (h, r, nt) in known and tries < 20:
                        nt = int(rng.integers(0, len(entities))); tries += 1
                    nh = h
                else:
                    nh = int(rng.integers(0, len(entities))); nt = t
                    tries = 0
                    while (nh, r, t) in known and tries < 20:
                        nh = int(rng.integers(0, len(entities))); tries += 1
                pos = score(E,R,h,r,t); neg = score(E,R,nh,r,nt)
                loss = max(0.0, 1.0 - pos + neg)
                total_loss += loss
                if loss > 0:
                    wins += 1
                    gp = E[h] * R[r] * E[t]
                    gn_h = E[nh] * R[r] * E[nt]
                    grad_h = R[r] * E[t]
                    grad_t = E[h] * R[r]
                    grad_r = E[h] * E[t]
                    grad_nh = R[r] * E[nt]
                    grad_nt = E[nh] * R[r]
                    E[h] += args.lr * (grad_h - grad_nh)
                    E[t] += args.lr * grad_t
                    E[nh] -= args.lr * grad_nh
                    E[nt] -= args.lr * grad_nt
                    R[r] += args.lr * (grad_r - E[nh] * E[nt])
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            history.append({"epoch":epoch,"mean_pairwise_loss":round(total_loss/(len(train_ids)*args.negatives),6),
                            "active_negative_pairs":wins})
    ranks = []
    for h,r,t in dev_ids:
        candidates = [score(E,R,h,r,c) for c in range(len(entities))]
        rank = 1 + sum(s > candidates[t] for s in candidates)
        ranks.append(rank)
    metrics = {
        "mrr": float(np.mean([1.0/x for x in ranks])),
        "hits_at_1": float(np.mean([x <= 1 for x in ranks])),
        "hits_at_3": float(np.mean([x <= 3 for x in ranks])),
        "hits_at_10": float(np.mean([x <= 10 for x in ranks])),
        "dev_ranks": ranks,
        "dev_count": len(dev_ids)
    }
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output} is not empty; choose another output or --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "semantic_model.npz", entity_embeddings=E, relation_embeddings=R)
    config = {"architecture":"AKA-PY-2","model":"DistMult","initialization":"random",
              "pretrained_weights_loaded":False,"dim":args.dim,"epochs":args.epochs,
              "learning_rate":args.lr,"negatives_per_positive":args.negatives,
              "entities":len(entities),"relations":len(relations),"train_triples":len(train),
              "dev_triples":len(dev),"parameter_count":int(E.size+R.size),"seed":args.seed,
              "training_objective":"pairwise margin ranking"}
    (args.output/"entities.json").write_text(json.dumps(entities,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (args.output/"relations.json").write_text(json.dumps(relations,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report={"config":config,"history":history,"validation":metrics,
            "duration_s":round(time.perf_counter()-started,3),
            "validation_interpretation":"Held-out graph tail-ranking only; not natural-language QA or code generation."}
    (args.output/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

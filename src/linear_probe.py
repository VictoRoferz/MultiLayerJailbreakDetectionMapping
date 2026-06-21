"""
Linear probe — single-number signal test for Module 2 output.

For each layer in --layers, fits a logistic regression on train activations
and evaluates on val + test. Reports accuracy, AUROC, F1 per split per layer.
Gate for Phase 4: val accuracy >= 0.80 at the best layer means labels are
carrying enough geometric signal for downstream CVAE training.

Usage:
    python src/linear_probe.py \
        --artifacts-dir artifacts \
        --layers 5 10 15 20 25 \
        --results-path results/linear_probe.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load_split(layer_dir: Path, split: str):
    path = layer_dir / f"{split}_activations.pt"
    if not path.exists():
        return None, None
    d = torch.load(path, weights_only=False)
    X = d["activations"].cpu().numpy()
    y = d["labels"].cpu().numpy().astype(int)
    # Drop unjudged samples (label == -1)
    mask = y != -1
    return X[mask], y[mask]


def probe_one_layer(layer_dir: Path) -> dict:
    X_tr, y_tr = load_split(layer_dir, "train")
    X_va, y_va = load_split(layer_dir, "val")
    X_te, y_te = load_split(layer_dir, "test")
    if X_tr is None or len(np.unique(y_tr)) < 2:
        return {"error": "train split missing or single-class"}

    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="liblinear",
    )
    clf.fit(X_tr, y_tr)

    def _metrics(X, y):
        if X is None or y is None or len(y) == 0:
            return None
        prob = clf.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)
        try:
            auroc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else None
        except ValueError:
            auroc = None
        return {
            "n": int(len(y)),
            "acc": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auroc": float(auroc) if auroc is not None else None,
            "pos_rate": float((y == 1).mean()),
        }

    return {
        "n_train": int(len(y_tr)),
        "train_pos_rate": float((y_tr == 1).mean()),
        "train": _metrics(X_tr, y_tr),
        "val":   _metrics(X_va, y_va),
        "test":  _metrics(X_te, y_te),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--layers", type=int, nargs="+", default=[5, 10, 15, 20, 25])
    p.add_argument("--results-path", type=Path, default=Path("results/linear_probe.json"))
    args = p.parse_args()

    results: dict = {}
    print(f"{'layer':>6s} {'n_tr':>6s} {'tr_acc':>7s} {'val_acc':>8s} "
          f"{'val_auroc':>10s} {'val_f1':>7s} {'te_acc':>7s}")
    print("-" * 60)
    for L in args.layers:
        layer_dir = args.artifacts_dir / f"layer_{L}"
        r = probe_one_layer(layer_dir)
        results[f"layer_{L}"] = r
        if "error" in r:
            print(f"{L:>6d}  {r['error']}")
            continue
        tr = r["train"] or {}
        va = r["val"] or {}
        te = r["test"] or {}
        print(f"{L:>6d} {r['n_train']:>6d} "
              f"{tr.get('acc', 0):>7.3f} "
              f"{va.get('acc', 0):>8.3f} "
              f"{(va.get('auroc') or 0):>10.3f} "
              f"{va.get('f1', 0):>7.3f} "
              f"{te.get('acc', 0):>7.3f}")

    # Pick best layer by val accuracy
    scored = [(k, v["val"]["acc"]) for k, v in results.items()
              if "error" not in v and v.get("val")]
    if scored:
        best = max(scored, key=lambda x: x[1])
        print("-" * 60)
        print(f"best layer by val acc: {best[0]} = {best[1]:.3f}  "
              f"[gate for Phase 4: >= 0.80]")
        results["_best_layer"] = {"layer": best[0], "val_acc": best[1]}

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults written to {args.results_path}")


if __name__ == "__main__":
    main()

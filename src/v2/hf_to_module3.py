#!/usr/bin/env python3
"""
hf_to_module3.py — Materialize the HF v2 dataset into the v1-format files that
module3_perturbation_generator.py expects.

module3 trains a conditional generator  G(z, f_L(x)) -> delta_f  so that
injecting  f_L(x) + delta_f  into the target model at layer L flips a *benign*
prompt into jailbreak behavior. It reads three local artifacts per layer:

    artifacts/layer_{L}/train_activations.pt   {"activations": [N,d], "labels": [N]}
    artifacts/layer_{L}/harmful_activations.pt  Tensor[M, d]   (the steering target)
    artifacts/test_passages.pt                  list[str]      (benign val prompts)

CRITICAL design choice (see CLAUDE.md "Findings"):
    The steering TARGET must be the *jailbroken-vs-refused* direction, NOT the
    benign-vs-harmful-topic direction. benign-vs-harmful-topic is trivially
    separable (AUROC ~1.0) and only encodes topic, so a generator trained on it
    learns "look like a harmful question" — which the model just refuses.

    So:  benign_acts   = rows with category == "benign"
         harmful_acts  = rows with label == 1   (CONFIRMED jailbreaks, any family)
    This is the refusal-suppression direction (AUROC ~0.72) we actually want.

Usage (from repo root):
    python -m src.v2.hf_to_module3 --target gemma --layers 20
    python -m src.v2.hf_to_module3 --target gemma --layers 5 10 15 20 25
    # or point straight at a repo:
    python -m src.v2.hf_to_module3 --repo victorroferz/gemma-2b-jailbreak-behavior-dataset-v2 \
        --layers 20 --hidden-dim 2304
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.v2.config import get_config, list_targets


def _acts_for_layer(split, layer: int) -> torch.Tensor:
    """Stack the activation_layer_{L} column of a HF split into a float32 tensor."""
    col = f"activation_layer_{layer}"
    if col not in split.column_names:
        raise KeyError(
            f"Column '{col}' not in dataset. Available: "
            f"{[c for c in split.column_names if c.startswith('activation_layer_')]}"
        )
    arr = np.asarray(split[col], dtype=np.float32)   # [N, d]
    return torch.from_numpy(arr)


def main():
    ap = argparse.ArgumentParser(
        description="Convert HF v2 dataset -> module3 v1-format files."
    )
    ap.add_argument("--target", choices=list_targets(), default=None,
                    help="Named target in config.py (sets repo, layers, dim).")
    ap.add_argument("--repo", type=str, default=None,
                    help="HF repo id (overrides --target's hf_repo).")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Layers to materialize. Default: target's layers.")
    ap.add_argument("--out-root", type=str, default="artifacts",
                    help="Where to write layer_{L}/ dirs (default: artifacts).")
    ap.add_argument("--n-val-passages", type=int, default=200,
                    help="How many benign test-split prompts to save for ASR validation.")
    ap.add_argument("--min-jailbroken", type=int, default=30,
                    help="Warn if fewer than this many label==1 rows are found.")
    args = ap.parse_args()

    from datasets import load_dataset, concatenate_datasets

    if args.target is None and args.repo is None:
        args.target = "gemma"
    if args.target is not None:
        cfg = get_config(args.target)
        repo = args.repo or cfg["hf_repo"]
        layers = args.layers or list(cfg["layers"])
    else:
        repo = args.repo
        layers = args.layers
        if layers is None:
            ap.error("--layers is required when using --repo without --target.")

    out_root = Path(args.out_root)
    print("=" * 64)
    print("  HF v2  ->  module3 inputs")
    print(f"  repo:   {repo}")
    print(f"  layers: {layers}")
    print(f"  out:    {out_root}/layer_{{L}}/")
    print("=" * 64)

    dd = load_dataset(repo)
    print(f"[-] Splits: { {k: len(v) for k, v in dd.items()} }")

    # The generator trains on the TRAIN split (no leakage). We pull benign +
    # jailbroken activation pools from train, and benign prompt strings from test.
    train = dd["train"]
    test = dd.get("test", dd["train"])

    cats = np.asarray(train["category"])
    labels = np.asarray(train["label"])
    benign_mask = cats == "benign"
    jb_mask = labels == 1            # confirmed jailbreaks (any attack family)
    refused_mask = (~benign_mask) & (labels == 0)

    n_benign = int(benign_mask.sum())
    n_jb = int(jb_mask.sum())
    n_ref = int(refused_mask.sum())
    print(f"[-] TRAIN pools: benign={n_benign}  jailbroken(label==1)={n_jb}  "
          f"refused={n_ref}")
    if n_jb < args.min_jailbroken:
        print(f"    [WARN] Only {n_jb} jailbroken rows — the steering target will "
              f"be weak. Consider merging in v1 jailbreak rows or lowering judge tau.")

    benign_idx = np.where(benign_mask)[0].tolist()
    jb_idx = np.where(jb_mask)[0].tolist()

    for L in layers:
        layer_dir = out_root / f"layer_{L}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        all_acts = _acts_for_layer(train, L)              # [N, d]
        benign_acts = all_acts[benign_idx]                # [n_benign, d]
        harmful_acts = all_acts[jb_idx]                   # [n_jb, d]  (= jailbroken)

        # train_activations.pt: module3's load_benign_activations() filters to
        # labels == 0.0, so we tag benign rows 0 and (unused-here) others 1.
        train_labels = torch.zeros(benign_acts.shape[0], dtype=torch.float32)
        torch.save(
            {"activations": benign_acts, "labels": train_labels},
            layer_dir / "train_activations.pt",
        )
        torch.save(harmful_acts, layer_dir / "harmful_activations.pt")
        print(f"  layer {L:>2}: benign {tuple(benign_acts.shape)} -> "
              f"train_activations.pt | jailbroken {tuple(harmful_acts.shape)} "
              f"-> harmful_activations.pt")

    # Benign prompt strings for the ASR validation loop (delta injected into these).
    test_cats = np.asarray(test["category"])
    test_benign_idx = np.where(test_cats == "benign")[0][: args.n_val_passages]
    passages = [test["prompt"][int(i)] for i in test_benign_idx]
    torch.save(passages, out_root / "test_passages.pt")
    print(f"[-] Saved {len(passages)} benign validation prompts -> "
          f"{out_root}/test_passages.pt")

    print("\nDone. Now train the generator, e.g.:")
    lyr = layers[0]
    print(f"    python src/module3_perturbation_generator.py --layer {lyr} "
          f"--phase all --epsilon 0.15 --rl-steps 3000")


if __name__ == "__main__":
    main()

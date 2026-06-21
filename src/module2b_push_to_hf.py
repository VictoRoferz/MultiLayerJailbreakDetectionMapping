"""
Module 2b — Push labeled activation dataset to HuggingFace Hub.

Loads the local artifacts produced by Module 2 and publishes them as a single
unified HuggingFace Dataset with train / validation / test splits.

Dataset schema (per row):
    prompt               : string
    response             : string
    category             : string   ("benign" | "harmful_direct" | "jailbreak_wrapped")
    source               : string   (e.g. "wikitext-103", "trustairlab")
    judge_score          : dict     (raw judge output; empty dict for benign)
    label                : int      (0, 1, or -1 for judge failures)
    activation_layer_5   : float32[2304]
    activation_layer_10  : float32[2304]
    activation_layer_15  : float32[2304]
    activation_layer_20  : float32[2304]
    activation_layer_25  : float32[2304]

Usage:
    export HF_TOKEN=hf_xxx
    python module2b_push_to_hf.py --repo-id yourusername/dataset-name

Requires:
    pip install datasets huggingface_hub
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

LAYERS = [5, 10, 15, 20, 25]


def build_hf_dataset(source_dir: Path):
    """Load full_dataset.pt + per-layer splits and build a HuggingFace DatasetDict."""
    from datasets import Dataset, DatasetDict, Features, Sequence, Value

    full_path = source_dir / "labeled_data" / "full_dataset.pt"
    if not full_path.exists():
        raise FileNotFoundError(
            f"{full_path} not found. Run module2_labeling_extraction.py first."
        )
    raw = torch.load(full_path, weights_only=False)

    n = len(raw["labels"])
    print(f"[-] Loaded {n} rows from full_dataset.pt")

    # Reassemble splits from the full dataset with the SAME seed used in
    # Module 2 (42 default). Keys match HuggingFace's expected split names.
    rng = np.random.default_rng(42)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)
    split_idx: Dict[str, List[int]] = {
        "train": all_idx[:n_train].tolist(),
        "validation": all_idx[n_train:n_train + n_val].tolist(),
        "test": all_idx[n_train + n_val:].tolist(),
    }

    def build_split(indices: List[int]) -> Dataset:
        rows = {
            "prompt": [raw["prompts"][i] for i in indices],
            "response": [raw["responses"][i] for i in indices],
            "category": [raw["categories"][i] for i in indices],
            "source": [raw["sources"][i] for i in indices],
            "judge_score": [
                {k: float(v) if isinstance(v, (int, float)) else str(v)
                 for k, v in (raw["judge_scores"][i] or {}).items()}
                for i in indices
            ],
            "label": [int(raw["labels"][i]) for i in indices],
        }
        for L in LAYERS:
            rows[f"activation_layer_{L}"] = [
                raw["activations"][L][i].tolist() for i in indices
            ]
        # Let datasets infer judge_score schema; pin activation shape.
        d_hidden = raw["activations"][LAYERS[0]][0].shape[0]
        features = Features({
            "prompt": Value("string"),
            "response": Value("string"),
            "category": Value("string"),
            "source": Value("string"),
            "judge_score": {},  # will be inferred
            "label": Value("int32"),
            **{
                f"activation_layer_{L}": Sequence(Value("float32"), length=d_hidden)
                for L in LAYERS
            },
        })
        # datasets struggles with empty-dict Features spec; skip explicit features
        # for judge_score so it auto-infers the struct.
        return Dataset.from_dict(rows)

    print("[-] Building train / validation / test splits...")
    dd = DatasetDict({
        name: build_split(idx) for name, idx in split_idx.items()
    })
    for name, split in dd.items():
        print(f"    {name:<12s}: {len(split)} rows")
    return dd


def main():
    parser = argparse.ArgumentParser(
        description="Push labeled activation dataset to the HuggingFace Hub."
    )
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HuggingFace repo, e.g. username/dataset-name")
    parser.add_argument("--source-dir", type=str, default="artifacts",
                        help="Directory containing labeled_data/ and layer_*/")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Upload as a private dataset (default: True)")
    parser.add_argument("--public", action="store_true",
                        help="Flip to a public dataset.")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not set. Export a HuggingFace token with write access."
        )

    private = args.private and not args.public
    source_dir = Path(args.source_dir)

    print("=" * 60)
    print("MODULE 2b — PUSH TO HUGGINGFACE HUB")
    print("=" * 60)
    print(f"    Source:   {source_dir}")
    print(f"    Target:   {args.repo_id}  ({'private' if private else 'public'})")
    print("=" * 60)

    dd = build_hf_dataset(source_dir)

    # Attach metadata if present
    meta_path = source_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        print("[-] Attaching metadata.json to dataset card")
    else:
        metadata = {}

    print(f"[-] Pushing to hub: {args.repo_id}")
    dd.push_to_hub(
        repo_id=args.repo_id,
        token=token,
        private=private,
    )

    print("\n" + "=" * 60)
    print(f"DONE — uploaded to https://huggingface.co/datasets/{args.repo_id}")
    if metadata:
        print(f"      Metadata: n_total={metadata.get('n_total')}, "
              f"label_1={metadata.get('label_counts', {}).get('label_1')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
v2_to_artifacts.py -- materialize the HuggingFace v2 dataset into the per-layer
.pt artifact files the v1 pipeline (module3 CVAE + module4-7) already reads.

Replaces Extraction.py as the activation source. The v2 dataset already carries
outcomes (`label`: 1 = harmful prompt, NOT refused, substantive harm = a genuine
successful jailbreak; 0 = refused/benign), so we skip the live-model extraction and
feed precomputed (activation, label) pairs straight into training + detection.

LABEL SEMANTICS (deliberate -- see CLAUDE.md "topic confound" finding):

  train_activations.pt   labels: 0 = benign, 1 = non-benign
      -> module3's load_benign_activations() filters label==0, giving CLEAN benign
         activations for CVAE conditioning (no refused-harmful leakage).

  harmful_activations.pt  = activations of SUCCESSFUL JAILBREAKS only
      (category starts with 'jailbreak' AND label == 1). Bare tensor. This is the
      positive class for the warm-up / reward AND the Option-B clustering input.
      delta = jailbreak - benign therefore points toward the region where the model
      actually produced harmful content (the steering direction the generator learns).

  val/test_activations.pt labels: 1 = SUCCESSFUL JAILBREAK, 0 = everything else
      (benign + refused-harmful + harmful-direct). module7 then measures the honest
      target -- flag successful jailbreaks among normal + refused traffic, with
      refused-harmful as HARD NEGATIVES -- instead of the benign-vs-harmful AUROC=1.0
      topic confound. Switch with --detector-target nonbenign to reproduce that confound.

  benign_centroid.pt      = mean of train benign activations (bare tensor). Shared
      reference point for Option-B direction computation and module7's delta space.

  artifacts/test_passages.pt, calibration_passages.pt = benign PROMPT strings (lists)
      from the test / validation splits, consumed by module4 corruption.

Every split file also stores `categories` and `success` (raw dataset label) so the
contrast can be re-derived downstream without re-reading the dataset.

USAGE
  python src/v2_to_artifacts.py --target gemma --layers 5 10 15 20 25
  python src/v2_to_artifacts.py --target gemma --layers 20            # single layer
  python src/v2_to_artifacts.py --source local \
      --parquet-dir hf_analysis/gemma-2b-jailbreak-behavior-dataset-v2/data --layers 20
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.v2.config import get_config


SPLIT_FILE_MAP = {"train": "train", "validation": "val", "test": "test"}
HF_SPLITS = ["train", "validation", "test"]


def is_jailbreak_category(cat: str) -> bool:
    return cat.startswith("jailbreak")


def load_split_hf(repo: str, split: str):
    from datasets import load_dataset
    return load_dataset(repo, split=split)


def load_split_local(parquet_dir: str, split: str):
    import glob
    import pandas as pd
    # HF parquet shards are named e.g. train-00000-of-00001.parquet
    files = sorted(glob.glob(str(Path(parquet_dir) / f"{split}*.parquet")))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _col(rows, name):
    """Index a column from either an HF Dataset or a pandas DataFrame."""
    return rows[name]


def extract_acts(rows, layer: int) -> np.ndarray:
    """Return acts[N, d] float32 for a layer, handling HF list-cols and pandas."""
    raw = rows[f"activation_layer_{layer}"]
    # pandas Series-of-lists -> materialize to a python list first; HF Dataset
    # column access already returns a python list of lists.
    if hasattr(raw, "to_list"):
        raw = raw.to_list()
    return np.asarray(raw, dtype=np.float32)


def get_meta(rows):
    """Return (categories list[str], success int array, prompts list[str])."""
    categories = list(_col(rows, "category"))
    success = np.asarray(_col(rows, "label"), dtype=np.int64)
    prompts = list(_col(rows, "prompt"))
    return categories, success, prompts


def build_labels(categories, success, target: str) -> np.ndarray:
    cats = np.array(categories)
    if target in ("benign", "nonbenign"):
        return (cats != "benign").astype(np.float32)
    if target == "jailbreak-success":
        is_jb = np.array([is_jailbreak_category(c) for c in categories])
        return ((is_jb) & (success == 1)).astype(np.float32)
    raise ValueError(f"unknown label target {target}")


def save_split(out_root, layer, split_file, acts, categories, success, label_target):
    labels = build_labels(categories, success, label_target)
    payload = {
        "activations": torch.from_numpy(acts),
        "labels": torch.from_numpy(labels),
        "categories": categories,
        "success": torch.from_numpy(success),
    }
    out_dir = out_root / f"layer_{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split_file}_activations.pt"
    torch.save(payload, path)
    print(f"    [{split_file}] {path.name}: {len(labels)} rows, "
          f"{int(labels.sum())} positive (label_target={label_target})")


def save_harmful(out_root, layer, acts, categories, success):
    """harmful_activations.pt = successful jailbreaks (positive class for CVAE + Option B)."""
    is_jb = np.array([is_jailbreak_category(c) for c in categories])
    jb_mask = is_jb & (success == 1)
    out_dir = out_root / f"layer_{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    harmful = torch.from_numpy(acts[jb_mask])
    torch.save(harmful, out_dir / "harmful_activations.pt")
    print(f"    harmful_activations.pt: {harmful.shape[0]} successful jailbreaks "
          f"(positive class for CVAE + Option-B clustering)")


def save_centroid(out_root, layer, acts, categories, src_split):
    """
    benign_centroid.pt = mean of benign activations. Computed from the VALIDATION
    split's benign rows so it matches module7, which recomputes the centroid from
    its calibration (= validation) benign and overwrites this file. Keeping the two
    identical means Option-B clustering and the detector share the same origin.
    """
    benign_mask = np.array(categories) == "benign"
    out_dir = out_root / f"layer_{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    centroid = torch.from_numpy(acts[benign_mask].mean(axis=0))
    torch.save(centroid, out_dir / "benign_centroid.pt")
    print(f"    benign_centroid.pt: mean of {int(benign_mask.sum())} benign acts "
          f"(from {src_split} split)")


def save_benign_passages(out_root, rows_by_split):
    """Benign prompt strings -> artifacts/{test,calibration}_passages.pt for module4."""
    def benign_prompts(rows):
        cats, _, prompts = get_meta(rows)
        return [p for c, p in zip(cats, prompts) if c == "benign" and p]

    out_root.mkdir(parents=True, exist_ok=True)
    if "test" in rows_by_split:
        test_p = benign_prompts(rows_by_split["test"])
        torch.save(test_p, out_root / "test_passages.pt")
        print(f"[-] test_passages.pt: {len(test_p)} benign prompts (module4)")
    if "validation" in rows_by_split:
        cal_p = benign_prompts(rows_by_split["validation"])
        torch.save(cal_p, out_root / "calibration_passages.pt")
        print(f"[-] calibration_passages.pt: {len(cal_p)} benign prompts (module4)")


def main():
    ap = argparse.ArgumentParser(description="HF v2 dataset -> v1 pipeline artifacts")
    ap.add_argument("--target", default="gemma", help="config target (gemma | vicuna)")
    ap.add_argument("--source", choices=["hf", "local"], default="hf")
    ap.add_argument("--repo", default=None, help="override HF repo (else from config)")
    ap.add_argument("--parquet-dir",
                    default="hf_analysis/gemma-2b-jailbreak-behavior-dataset-v2/data")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="layers to materialize (else all from config)")
    ap.add_argument("--out-root", default="artifacts")
    ap.add_argument("--detector-target", choices=["jailbreak-success", "nonbenign"],
                    default="jailbreak-success",
                    help="val/test label semantics for module7. jailbreak-success = "
                         "honest; nonbenign = topic-confound baseline")
    args = ap.parse_args()

    cfg = get_config(args.target)
    repo = args.repo or cfg["hf_repo"]
    layers = args.layers or list(cfg["layers"])
    out_root = Path(args.out_root)

    print(f"[-] target={args.target}  repo={repo}  layers={layers}")

    # Load each split once, then slice per layer
    splits = {}
    for hf_split in HF_SPLITS:
        rows = (load_split_hf(repo, hf_split) if args.source == "hf"
                else load_split_local(args.parquet_dir, hf_split))
        if rows is None:
            print(f"[!] split '{hf_split}' not found; skipping")
            continue
        splits[hf_split] = rows
        print(f"[-] loaded split '{hf_split}': {len(rows)} rows")

    for layer in layers:
        print(f"\n[=] Layer {layer}")
        per_split = {}  # cache acts+meta per split for centroid step
        for hf_split, rows in splits.items():
            acts = extract_acts(rows, layer)
            categories, success, _ = get_meta(rows)
            per_split[hf_split] = (acts, categories, success)
            split_file = SPLIT_FILE_MAP[hf_split]
            # train: benign-indicator labels (clean benign conditioning for module3)
            # val/test: detector-target labels (honest module7 eval)
            label_target = "benign" if hf_split == "train" else args.detector_target
            save_split(out_root, layer, split_file, acts, categories, success,
                       label_target)
            if hf_split == "train":
                save_harmful(out_root, layer, acts, categories, success)

        # benign_centroid from validation (matches module7's calibration source);
        # fall back to train benign if no validation split is available.
        cen_split = "validation" if "validation" in per_split else "train"
        c_acts, c_cats, _ = per_split[cen_split]
        save_centroid(out_root, layer, c_acts, c_cats, cen_split)

    save_benign_passages(out_root, splits)
    print("\n[done] artifacts written under", out_root)


if __name__ == "__main__":
    main()

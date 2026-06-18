"""
plot_separated_space.py -- draw the activation space with MDS, colored by group.

Embeds Gemma activations into 2-D with MDS (distance-preserving, the method from
arXiv 2412.17034 which avoids PCA's linear assumption) and colors points by group
(benign / harmful-direct / jailbreak) to show where jailbreaks sit relative to the
rest. Data comes from the HuggingFace dataset (default) or local artifacts/.

NOTE: the paper used the last-token vector of the INPUT prompt; our activations are
the mean of the last 5 RESPONSE tokens (src/Extraction.py), so the geometry may
differ. MDS is unsupervised (labels only pick colors), so the split just controls
how many points are drawn -- no train/test leakage concern.

USAGE
  python plot_separated_space.py --layer 15
  python plot_separated_space.py --layer 15 --split val --mds-sample 1000
  python plot_separated_space.py --source local --layer 15
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_HF_REPO = "victorroferz/gemma-2b-jailbreak-behavior-dataset-v2"
HF_SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}

GROUP_ORDER = ["benign", "harmful-direct", "jailbreak", "other"]
GROUP_COLORS = {"benign": "steelblue", "harmful-direct": "orange",
                "jailbreak": "crimson", "other": "gray"}


def category_to_group(cat: str) -> str:
    # map the dataset's category string to one of the plot groups
    if cat == "benign":
        return "benign"
    if cat == "harmful_direct":
        return "harmful-direct"
    if cat.startswith("jailbreak"):
        return "jailbreak"
    return "other"


def load_layer_hf(hf_repo, layer, split):
    # pull one layer's activations + category from the hf split, grouped for coloring
    from datasets import load_dataset
    ds = load_dataset(hf_repo, split=HF_SPLIT_MAP[split])
    acts = np.array(ds[f"activation_layer_{layer}"], dtype=np.float32)
    groups = np.array([category_to_group(c) for c in ds["category"]])
    return acts, groups


def load_layer_local(artifacts_dir, layer, split, pos_label, neg_label):
    # local .pt has no category, so fall back to 2 groups derived from the label
    path = Path(artifacts_dir) / f"layer_{layer}" / f"{split}_activations.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    print("  [warn] local mode has no category column -> only benign/harmful groups "
          "(use --source hf for the 3-group jailbreak view)")
    data = torch.load(path, weights_only=False)
    acts = data["activations"].to(torch.float32).numpy()
    labels = data["labels"].to(torch.long).numpy()
    keep = np.isin(labels, [neg_label, pos_label])
    acts, labels = acts[keep], labels[keep]
    groups = np.where(labels == pos_label, "harmful-direct", "benign")
    return acts, groups


def stratified_sample(groups, cap, seed):
    # take an even share per group so rare groups (jailbreak) still show up
    rng = np.random.default_rng(seed)
    present = [g for g in GROUP_ORDER if (groups == g).any()]
    per_group = max(1, cap // len(present))
    idx = []
    for g in present:
        g_idx = np.where(groups == g)[0]
        if len(g_idx) > per_group:
            g_idx = rng.choice(g_idx, per_group, replace=False)
        idx.append(g_idx)
    return np.sort(np.concatenate(idx))


def mds_2d(acts, seed):
    # standardize first so massive-activation dims don't dominate the distances
    X = StandardScaler().fit_transform(acts)
    mds = MDS(n_components=2, n_init=1, max_iter=300,
              normalized_stress="auto", random_state=seed)
    return mds.fit_transform(X)


def main():
    ap = argparse.ArgumentParser(description="MDS activation-space plot by group.")
    ap.add_argument("--source", choices=["hf", "local"], default="hf")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--out-dir", default="figures/mds")
    ap.add_argument("--mds-sample", type=int, default=800,
                    help="cap on points embedded (MDS is O(N^2)); sampled evenly per group")
    ap.add_argument("--benign-label", type=int, default=0)
    ap.add_argument("--harmful-label", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-outliers", action="store_true",
                    help="drop >3-sigma activation-norm points (BOS / massive-activation)")
    args = ap.parse_args()

    if args.source == "hf":
        acts, groups = load_layer_hf(args.hf_repo, args.layer, args.split)
    else:
        acts, groups = load_layer_local(args.artifacts_dir, args.layer, args.split,
                                        args.harmful_label, args.benign_label)
    print(f"loaded layer {args.layer} ({args.split}): {len(acts)} points")

    if args.drop_outliers:
        norms = np.linalg.norm(acts, axis=1)
        z = (norms - norms.mean()) / (norms.std() + 1e-8)
        mask = np.abs(z) < 3.0
        print(f"  dropped {int((~mask).sum())} norm-outlier points")
        acts, groups = acts[mask], groups[mask]

    if len(acts) > args.mds_sample:
        sel = stratified_sample(groups, args.mds_sample, args.seed)
        acts, groups = acts[sel], groups[sel]
        print(f"  subsampled to {len(acts)} points for MDS")

    coords = mds_2d(acts, args.seed)

    out = Path(args.out_dir) / f"layer_{args.layer}_mds.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for g in GROUP_ORDER:
        m = groups == g
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.55,
                   c=GROUP_COLORS[g], label=f"{g} (n={int(m.sum())})")
    ax.set_title(f"Layer {args.layer}: MDS activation space ({args.split})")
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

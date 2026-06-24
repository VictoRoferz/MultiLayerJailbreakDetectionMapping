"""Experiment 1 — Are real jailbreak attack families distinct subspaces?

For each model/layer, build a per-attack-family jailbreak direction
    d_f = mean(successful jailbreaks of family f) - mean(refused-harmful)
and ask whether the families point in *different* directions / span *different*
subspaces, or collapse onto one shared "jailbreak axis".

Rigor: cross-family cosine is meaningless without a noise floor. So for every
family we also compute a within-family split-half cosine (bootstrapped) — the
*ceiling* of how aligned two estimates of the SAME direction are given finite
samples. Families are genuinely distinct only if
    cross-family cosine  <<  within-family split-half cosine.

Two views:
  (A) rank-1 directions  -> pairwise cosine matrix + within-family ceiling
  (B) top-r subspaces    -> mean principal-angle cosine (subspace overlap)

CPU-only; reads the local v2 parquet. Run from repo root:
    python hf_analysis/family_subspaces.py            # both models
    python hf_analysis/family_subspaces.py gemma 20   # one model/layer
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

RNG = np.random.default_rng(42)

REPOS = {
    "gemma": ("hf_analysis/gemma-2b-jailbreak-behavior-dataset-v2", [5, 10, 15, 20, 25]),
    "vicuna": ("hf_analysis/vicuna-7b-jailbreak-behavior-dataset-v2", [4, 10, 15, 20, 25, 30]),
}

# Colab / no-local-data path: set these to your Hub dataset repo ids, e.g.
#   export HF_GEMMA_REPO="<user>/gemma-2b-jailbreak-behavior-dataset-v2"
#   export HF_VICUNA_REPO="<user>/vicuna-7b-jailbreak-behavior-dataset-v2"
HF_IDS = {
    "gemma": os.environ.get("HF_GEMMA_REPO"),
    "vicuna": os.environ.get("HF_VICUNA_REPO"),
}
FAMILIES = ["jailbreak_gcg_universal", "jailbreak_artprompt",
            "jailbreak_gcg_individual", "harmful_direct"]
MIN_SUCCESS = 25          # skip families with fewer successful rows
R = 5                     # subspace rank for view (B)
N_BOOT = 50               # split-half bootstraps for the ceiling


def load_full(base, hf_id=None):
    """Local parquet if present; otherwise pull the dataset from the HF Hub."""
    frames = []
    for split in ("train", "validation", "test"):
        for f in sorted(glob.glob(f"{base}/data/{split}-*.parquet")):
            frames.append(pd.read_parquet(f))
    if frames:
        return pd.concat(frames, ignore_index=True)
    if hf_id:
        from datasets import load_dataset
        print(f"  [hub] no local parquet — loading {hf_id} from HF Hub ...")
        dd = load_dataset(hf_id)
        return pd.concat([dd[s].to_pandas() for s in dd], ignore_index=True)
    raise FileNotFoundError(
        f"No parquet under {base}/data/ and no HF repo id given. "
        f"Set HF_GEMMA_REPO / HF_VICUNA_REPO env vars for Colab.")


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def split_half_cosine(X, ref_mean):
    """Bootstrapped cosine between two half-sample DiM directions of one family."""
    n = len(X)
    cs = []
    for _ in range(N_BOOT):
        perm = RNG.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        da = unit(X[a].mean(0) - ref_mean)
        db = unit(X[b].mean(0) - ref_mean)
        cs.append(float(da @ db))
    return float(np.mean(cs))


def subspace_basis(X, r):
    """Top-r right-singular vectors of centered X (columns = orthonormal basis)."""
    p = PCA(n_components=min(r, X.shape[0] - 1, X.shape[1]), random_state=42)
    p.fit(X)
    return p.components_.T  # (dim, r)


def subspace_overlap(Ba, Bb):
    """Mean cosine of principal angles between two orthonormal bases in [0,1]."""
    s = np.linalg.svd(Ba.T @ Bb, compute_uv=False)
    return float(np.clip(s, 0, 1).mean())


def analyze(model, base, layers):
    print(f"\n{'='*72}\nMODEL: {model}")
    full = load_full(base, HF_IDS.get(model))
    refused_mask = (full["category"] != "benign") & (full["label"] == 0)
    print(f"  refused-harmful pool: n={int(refused_mask.sum())}")
    present = {}
    for f in FAMILIES:
        n = int(((full["category"] == f) & (full["label"] == 1)).sum())
        present[f] = n
        print(f"  {f:28s} successful jailbreaks n={n}"
              + ("  [SKIP <%d]" % MIN_SUCCESS if n < MIN_SUCCESS else ""))
    fams = [f for f in FAMILIES if present[f] >= MIN_SUCCESS]

    out = {}
    for L in layers:
        col = f"activation_layer_{L}"
        if col not in full.columns:
            continue
        X = np.stack(full[col].to_numpy()).astype(np.float32)
        ref_mean = X[refused_mask.to_numpy()].mean(0)

        dirs, bases, ceil = {}, {}, {}
        for f in fams:
            m = ((full["category"] == f) & (full["label"] == 1)).to_numpy()
            Xf = X[m]
            dirs[f] = unit(Xf.mean(0) - ref_mean)
            bases[f] = subspace_basis(Xf - ref_mean, R)
            ceil[f] = split_half_cosine(Xf, ref_mean)

        # (A) cross-family rank-1 cosine matrix
        cos = {a: {b: round(float(dirs[a] @ dirs[b]), 3) for b in fams} for a in fams}
        # (B) cross-family subspace overlap matrix
        ov = {a: {b: round(subspace_overlap(bases[a], bases[b]), 3) for b in fams}
              for a in fams}

        pairs = [(a, b) for i, a in enumerate(fams) for b in fams[i + 1:]]
        mean_cross_cos = round(float(np.mean([cos[a][b] for a, b in pairs])), 3) if pairs else None
        mean_ceiling = round(float(np.mean(list(ceil.values()))), 3)

        out[str(L)] = dict(
            families=fams,
            within_family_ceiling={f: round(ceil[f], 3) for f in fams},
            mean_within_family_ceiling=mean_ceiling,
            cross_family_cosine=cos,
            mean_cross_family_cosine=mean_cross_cos,
            cross_family_subspace_overlap=ov,
        )
        print(f"\n  layer {L}:  mean cross-family cosine={mean_cross_cos}  "
              f"| mean within-family ceiling={mean_ceiling}")
        print(f"    {'':22s}" + "".join(f"{f.split('_')[-1][:8]:>10s}" for f in fams))
        for a in fams:
            print(f"    {a.split('_')[-1][:20]:22s}"
                  + "".join(f"{cos[a][b]:>10.3f}" for b in fams))
    return out


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        wanted = {args[0]: (REPOS[args[0]][0], [int(args[1])])}
    else:
        wanted = {m: REPOS[m] for m in (args or REPOS)}
    results = {m: analyze(m, base, layers) for m, (base, layers) in wanted.items()}
    with open("hf_analysis/family_subspaces_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved hf_analysis/family_subspaces_results.json")


if __name__ == "__main__":
    main()

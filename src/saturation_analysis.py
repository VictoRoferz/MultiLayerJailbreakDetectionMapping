#!/usr/bin/env python3
"""
saturation_analysis.py — Table 3 + Figure 5 of the mapping paper.

The paper's "coverage-driven" claim: as we process more benign passages through
corruption -> judge -> clustering, we stop discovering NEW jailbreak clusters
(the successful-perturbation space saturates). This script measures that.

Method:
    1. Load module5's judged_results.pt (successful delta_f + per-perturbation
       passage_idx in successful_metadata).
    2. Order passages by passage_idx. At each milestone m (a growing fraction of
       the unique passages seen), take all successful delta_f from passages with
       passage_idx <= cutoff, cluster them (PCA -> K-means sweep, K* = argmax
       silhouette — same recipe as module6), and record K* (the number of
       distinct jailbreak clusters discovered so far) + n_successful so far.
    3. Output saturation_results.json + figures/saturation_curve.png. A curve
       that flattens = coverage saturates.

Usage (from repo root, after corruption+judge for the layer):
    python src/saturation_analysis.py --layer 20
    python src/saturation_analysis.py --layer 20 --n-milestones 10 --n-pca 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _kstar(data: np.ndarray, n_pca: int, k_max: int = 20):
    """PCA -> K-means sweep; return (k_star, best_silhouette). Mirrors module6."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(data)
    if n < 4:
        return 1, float("nan")
    comp = int(min(n_pca, data.shape[1], n - 1))
    reduced = PCA(n_components=comp, random_state=42).fit_transform(data)

    best_k, best_sil = 2, -1.0
    for k in range(2, min(k_max, n - 1) + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(reduced)
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(reduced, labels))
        if sil > best_sil:
            best_sil, best_k = sil, k
    return best_k, best_sil


def main():
    ap = argparse.ArgumentParser(description="Saturation / coverage analysis.")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--artifacts-root", default="artifacts")
    ap.add_argument("--n-milestones", type=int, default=10)
    ap.add_argument("--n-pca", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    art_root = Path(args.artifacts_root)
    layer_dir = art_root / f"layer_{args.layer}"
    judged = layer_dir / "judged_results.pt"
    if not judged.exists():
        raise FileNotFoundError(
            f"{judged} not found — run module4 corruption + module5 judge first.")

    data = torch.load(judged, weights_only=False)
    deltas = data.get("successful_delta_f", [])
    meta = data.get("successful_metadata", [])
    if not deltas:
        raise ValueError("No successful_delta_f in judged_results.pt (ASR was 0).")

    delta_np = torch.stack([d if isinstance(d, torch.Tensor) else torch.tensor(d)
                            for d in deltas]).to(torch.float32).numpy()
    passage_idx = np.array([m.get("passage_idx", i) for i, m in enumerate(meta)])

    unique_passages = np.unique(passage_idx)
    n_unique = len(unique_passages)
    print(f"[-] {len(delta_np)} successful delta_f from {n_unique} unique passages")

    # Milestones: growing fractions of the unique passages seen so far.
    fracs = np.linspace(1.0 / args.n_milestones, 1.0, args.n_milestones)
    curve = []
    for frac in fracs:
        n_keep = max(1, int(round(frac * n_unique)))
        cutoff = unique_passages[n_keep - 1]
        mask = passage_idx <= cutoff
        subset = delta_np[mask]
        k_star, sil = _kstar(subset, args.n_pca)
        curve.append({
            "passages_processed": int(n_keep),
            "fraction": float(frac),
            "n_successful": int(mask.sum()),
            "k_star": int(k_star),
            "silhouette": None if np.isnan(sil) else float(sil),
        })
        print(f"  passages={n_keep:4d}  n_success={int(mask.sum()):4d}  "
              f"K*={k_star}  sil={sil:.4f}")

    out_dir = Path(args.out) if args.out else art_root / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"layer": args.layer, "n_unique_passages": int(n_unique),
               "n_total_successful": int(len(delta_np)), "curve": curve}
    with open(out_dir / "saturation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[-] wrote {out_dir / 'saturation_results.json'}")

    # Figure 5: cumulative distinct clusters (K*) vs passages processed.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        xs = [c["passages_processed"] for c in curve]
        ys = [c["k_star"] for c in curve]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, "o-", color="#C44E52")
        ax.set_xlabel("Benign passages processed")
        ax.set_ylabel("Distinct jailbreak clusters (K*)")
        ax.set_title(f"Coverage saturation — layer {args.layer}")
        fig.tight_layout()
        fig.savefig(fig_dir / "saturation_curve.png", dpi=150)
        plt.close(fig)
        print(f"[-] wrote {fig_dir / 'saturation_curve.png'}")
    except Exception as e:
        print(f"  [skip] figure: matplotlib unavailable ({e})")


if __name__ == "__main__":
    main()

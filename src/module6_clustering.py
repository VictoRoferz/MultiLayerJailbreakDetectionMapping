#!/usr/bin/env python3
"""
module6_clustering.py

Module 6 — Clustering & Subspace Mapping (Paper Section 3.7)

Given the set of successful delta_f vectors from Module 5 (judge):
    1. PCA to 50 dimensions (verify >90% variance retained)
    2. K-means for K in [1, 20]: elbow criterion + silhouette score → K*
    3. DBSCAN cross-validation (auto-tuned eps)
    4. UMAP 2D visualization colored by cluster label
    5. Compute cluster centers mu_k, intra/inter cluster distances

Outputs fill:
    - Paper Table 2: K*, Silhouette, Intra, Inter per layer
    - Paper Figure 2: UMAP of delta_f clusters
    - Paper Figure 3: Elbow plot

Baselines:
    - Random clustering: randomly assign delta_f to K* clusters

Usage:
    python src/module6_clustering.py --layer 20 --n-pca-dims 50
    python src/module6_clustering.py --layer all
    python src/module6_clustering.py --layer 20 --k-range 1 20
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from module3_perturbation_generator import DEFAULT_LAYERS

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Optional dependencies                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[!] scikit-learn required. Install: pip install scikit-learn")
    import sys; sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 1: Data Loading                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_successful_perturbations(layer_idx: int) -> dict:
    """
    Load successful delta_f vectors from Module 5 output.

    Returns dict with:
        delta_f: [N_success, D] tensor
        f_L: [N_success, D] tensor
        metadata: list of dicts
    """
    path = Path("artifacts") / f"layer_{layer_idx}" / "judged_results.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No judged results at {path}. "
            f"Run module5_judge.py --layer {layer_idx} first."
        )

    data = torch.load(path, weights_only=False)

    delta_f_list = data.get("successful_delta_f", [])
    f_L_list = data.get("successful_f_L", [])

    if not delta_f_list:
        raise ValueError(
            f"No successful perturbations found for layer {layer_idx}. "
            f"This means no corruption outputs passed the judge threshold."
        )

    delta_f = torch.stack(delta_f_list)
    f_L = torch.stack(f_L_list) if f_L_list else None

    return {
        "delta_f": delta_f,
        "f_L": f_L,
        "metadata": data.get("successful_metadata", []),
        "n_successful": len(delta_f_list),
    }


def ensure_figures_dir(layer_idx: int) -> Path:
    fig_dir = Path("artifacts") / f"layer_{layer_idx}" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 2: PCA Dimensionality Reduction                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def fit_pca(delta_f_np: np.ndarray, n_components: int = 50) -> tuple:
    """
    Fit PCA on delta_f vectors.

    Returns (pca_model, transformed_data, cumulative_variance).
    Warns if cumulative variance < 90%.
    """
    n_comp = min(n_components, delta_f_np.shape[0], delta_f_np.shape[1])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(delta_f_np)

    cumulative_var = np.sum(pca.explained_variance_ratio_)
    print(f"  PCA: {delta_f_np.shape[1]}D → {n_comp}D")
    print(f"  Cumulative variance retained: {cumulative_var:.4f} ({cumulative_var*100:.1f}%)")

    if cumulative_var < 0.90:
        print(f"  [WARN] Variance < 90%. Paper requires >90%. "
              f"Consider increasing --n-pca-dims.")

    return pca, transformed, cumulative_var


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 3: K-Means Clustering                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_kmeans_sweep(data: np.ndarray, k_min: int = 1, k_max: int = 21) -> dict:
    """
    Run K-means for K in [k_min, k_max) and compute metrics.

    Returns dict with:
        inertias: list of inertia values
        silhouette_scores: list of silhouette scores (NaN for K=1)
        k_star: optimal K by silhouette
        models: dict of {K: fitted KMeans model}
    """
    print(f"\n  K-means sweep: K = {k_min} to {k_max-1}")

    inertias = []
    sil_scores = []
    models = {}

    for k in range(k_min, k_max):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(data)
        inertias.append(float(km.inertia_))
        models[k] = km

        if k > 1 and len(set(labels)) > 1:
            sil = float(silhouette_score(data, labels))
        else:
            sil = float("nan")
        sil_scores.append(sil)

        print(f"    K={k:2d}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}"
              if not np.isnan(sil) else f"    K={k:2d}: inertia={km.inertia_:.1f}, silhouette=N/A")

    # Find K* (argmax silhouette, excluding K=1)
    valid_sil = [(k_min + i, s) for i, s in enumerate(sil_scores) if not np.isnan(s)]
    if valid_sil:
        k_star = max(valid_sil, key=lambda x: x[1])[0]
    else:
        k_star = 2  # fallback

    print(f"\n  K* (best silhouette) = {k_star}")

    return {
        "inertias": inertias,
        "silhouette_scores": sil_scores,
        "k_star": k_star,
        "k_range": list(range(k_min, k_max)),
        "models": models,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 4: DBSCAN Cross-Validation                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_dbscan(data: np.ndarray, min_samples: int = 5) -> dict:
    """
    Run DBSCAN with auto-tuned eps from k-distance graph.

    The eps is chosen at the "elbow" of the sorted k-nearest-neighbor
    distances (k = min_samples).

    Returns dict with labels, n_clusters, noise_fraction.
    """
    print(f"\n  DBSCAN (auto-eps, min_samples={min_samples})")

    # Compute k-distance graph
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    k_distances = np.sort(distances[:, -1])  # distances to k-th neighbor

    # Find elbow: second derivative
    if len(k_distances) > 10:
        d1 = np.diff(k_distances)
        d2 = np.diff(d1)
        elbow_idx = np.argmax(d2) + 2  # offset from diff
        eps = k_distances[min(elbow_idx, len(k_distances) - 1)]
    else:
        eps = float(np.median(k_distances))

    # Ensure reasonable eps
    eps = max(eps, 0.01)
    print(f"    Auto-tuned eps = {eps:.4f}")

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()

    print(f"    Clusters found: {n_clusters}")
    print(f"    Noise fraction: {noise_frac:.4f} ({noise_frac*100:.1f}%)")

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "noise_fraction": float(noise_frac),
        "eps": float(eps),
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 5: Cluster Metrics                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def compute_cluster_metrics(data: np.ndarray, labels: np.ndarray,
                            centers: np.ndarray) -> dict:
    """
    Compute intra-cluster and inter-cluster distances (cosine).
    These fill Paper Table 2.
    """
    unique_labels = [l for l in np.unique(labels) if l >= 0]

    # Intra-cluster: mean cosine distance within each cluster
    intra_distances = []
    for label in unique_labels:
        mask = labels == label
        if mask.sum() < 2:
            continue
        cluster_data = data[mask]
        cos_dist = cosine_distances(cluster_data)
        # Mean of upper triangle
        triu_idx = np.triu_indices(len(cluster_data), k=1)
        intra_distances.append(cos_dist[triu_idx].mean())

    mean_intra = float(np.mean(intra_distances)) if intra_distances else 0.0

    # Inter-cluster: mean cosine distance between cluster centers
    if len(centers) > 1:
        center_cos_dist = cosine_distances(centers)
        triu_idx = np.triu_indices(len(centers), k=1)
        mean_inter = float(center_cos_dist[triu_idx].mean())
    else:
        mean_inter = 0.0

    print(f"\n  Cluster metrics:")
    print(f"    Mean intra-cluster cosine distance: {mean_intra:.4f}")
    print(f"    Mean inter-cluster cosine distance: {mean_inter:.4f}")

    return {
        "mean_intra_cosine": mean_intra,
        "mean_inter_cosine": mean_inter,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 6: Visualization                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_elbow(k_range: list, inertias: list, sil_scores: list,
               k_star: int, save_path: Path):
    """Plot elbow + silhouette curves (Paper Figure 3)."""
    if not HAS_PLOTTING:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow plot
    ax1.plot(k_range, inertias, "o-", color="steelblue", linewidth=2)
    ax1.axvline(k_star, color="red", linestyle="--", alpha=0.7, label=f"K*={k_star}")
    ax1.set_xlabel("K (number of clusters)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("K-means Inertia (Elbow Plot)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Silhouette plot
    valid_k = [k for k, s in zip(k_range, sil_scores) if not np.isnan(s)]
    valid_s = [s for s in sil_scores if not np.isnan(s)]
    ax2.plot(valid_k, valid_s, "o-", color="darkorange", linewidth=2)
    if k_star in valid_k:
        idx = valid_k.index(k_star)
        ax2.scatter([k_star], [valid_s[idx]], color="red", s=100, zorder=5,
                    label=f"K*={k_star} (sil={valid_s[idx]:.3f})")
    ax2.set_xlabel("K (number of clusters)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs K")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_clusters_2d(data_2d: np.ndarray, labels: np.ndarray,
                     title: str, save_path: Path, method: str = "UMAP"):
    """Plot 2D cluster visualization (Paper Figure 2)."""
    if not HAS_PLOTTING:
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", max(len(unique_labels), 1))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                       c="gray", alpha=0.2, s=8, label="Noise")
        else:
            ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                       c=[cmap(i)], alpha=0.5, s=15, label=f"Cluster {label}")

    ax.set_xlabel(f"{method} dim 1")
    ax.set_ylabel(f"{method} dim 2")
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 7: Main Clustering Pipeline                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_clustering(
    layer_idx: int,
    n_pca_dims: int = 50,
    k_min: int = 1,
    k_max: int = 21,
    dbscan_min_samples: int = 5,
    save_plots: bool = True,
) -> dict:
    """
    Full Module 6 pipeline.

    Steps:
        1. Load successful delta_f from Module 5
        2. PCA to n_pca_dims (verify >90% variance)
        3. K-means sweep → K*
        4. DBSCAN cross-validation
        5. UMAP/PCA 2D visualization
        6. Compute cluster metrics

    Returns dict with all results for Paper Table 2.
    """
    print(f"\n{'='*60}")
    print(f"  Module 6: Clustering — Layer {layer_idx}")
    print(f"{'='*60}")

    # Load data
    perturbation_data = load_successful_perturbations(layer_idx)
    delta_f = perturbation_data["delta_f"].numpy()
    print(f"  Loaded {len(delta_f)} successful perturbation vectors")
    print(f"  Dimensionality: {delta_f.shape[1]}")

    # ── Step 1: PCA ──────────────────────────────────────────────────────
    pca_model, pca_data, cumulative_var = fit_pca(delta_f, n_pca_dims)

    # ── Step 2: K-means ──────────────────────────────────────────────────
    max_k = min(k_max, len(pca_data))  # can't have more clusters than samples
    kmeans_results = run_kmeans_sweep(pca_data, k_min, max_k)
    k_star = kmeans_results["k_star"]

    # Get best model's labels and centers
    best_km = kmeans_results["models"][k_star]
    km_labels = best_km.labels_
    km_centers = best_km.cluster_centers_

    # ── Step 3: DBSCAN ───────────────────────────────────────────────────
    dbscan_results = run_dbscan(pca_data, min_samples=dbscan_min_samples)

    # ── Step 4: Cluster metrics ──────────────────────────────────────────
    metrics = compute_cluster_metrics(pca_data, km_labels, km_centers)

    # Silhouette of best K
    best_sil = float(silhouette_score(pca_data, km_labels)) if k_star > 1 else 0.0

    # ── Baseline: Random clustering ──────────────────────────────────────
    random_labels = np.random.randint(0, k_star, size=len(pca_data))
    random_sil = float(silhouette_score(pca_data, random_labels)) if k_star > 1 else 0.0
    print(f"\n  Baseline (random K*={k_star} clustering): silhouette={random_sil:.4f}")
    print(f"  K-means advantage: {best_sil - random_sil:+.4f}")

    # ── Step 5: 2D Visualization ─────────────────────────────────────────
    if save_plots:
        fig_dir = ensure_figures_dir(layer_idx)

        # Elbow plot (Paper Figure 3)
        plot_elbow(
            kmeans_results["k_range"], kmeans_results["inertias"],
            kmeans_results["silhouette_scores"], k_star,
            fig_dir / "elbow_plot.png"
        )

        # 2D cluster visualization (Paper Figure 2)
        if HAS_UMAP and len(pca_data) > 15:
            print("\n  Computing UMAP embedding...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            data_2d = reducer.fit_transform(pca_data)
            vis_method = "UMAP"
        else:
            if not HAS_UMAP:
                print("  [NOTE] umap-learn not installed, using PCA-2D instead")
            pca_2d = PCA(n_components=2)
            data_2d = pca_2d.fit_transform(pca_data)
            vis_method = "PCA"

        plot_clusters_2d(
            data_2d, km_labels,
            f"Successful Perturbation Clusters — Layer {layer_idx}\n"
            f"K*={k_star}, silhouette={best_sil:.3f}",
            fig_dir / "umap_clusters.png",
            method=vis_method,
        )

        # Also plot DBSCAN clusters
        plot_clusters_2d(
            data_2d, dbscan_results["labels"],
            f"DBSCAN Clusters — Layer {layer_idx}\n"
            f"clusters={dbscan_results['n_clusters']}, "
            f"noise={dbscan_results['noise_fraction']:.1%}",
            fig_dir / "dbscan_clusters.png",
            method=vis_method,
        )

    # ── Save artifacts ────────────────────────────────────────────────────
    out_dir = Path("artifacts") / f"layer_{layer_idx}"

    # Save PCA model + cluster centers (for Module 7 detector)
    torch.save({
        "pca_components": torch.tensor(pca_model.components_, dtype=torch.float32),
        "pca_mean": torch.tensor(pca_model.mean_, dtype=torch.float32),
        "pca_explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
        "n_components": pca_model.n_components_,
    }, out_dir / "pca_model.pt")

    torch.save({
        "centers": torch.tensor(km_centers, dtype=torch.float32),
        "k_star": k_star,
        "labels": torch.tensor(km_labels, dtype=torch.long),
    }, out_dir / "cluster_centers.pt")

    # Save full results as JSON
    results = {
        "layer": layer_idx,
        "n_successful_perturbations": len(delta_f),
        "pca": {
            "n_components": int(pca_model.n_components_),
            "cumulative_variance": float(cumulative_var),
        },
        "kmeans": {
            "k_star": k_star,
            "silhouette_score": best_sil,
            "inertias": kmeans_results["inertias"],
            "silhouette_scores": [
                float(s) if not np.isnan(s) else None
                for s in kmeans_results["silhouette_scores"]
            ],
        },
        "dbscan": {
            "n_clusters": dbscan_results["n_clusters"],
            "noise_fraction": dbscan_results["noise_fraction"],
            "eps": dbscan_results["eps"],
        },
        "cluster_metrics": metrics,
        "baseline_random_silhouette": random_sil,
        "kmeans_advantage": float(best_sil - random_sil),
    }

    with open(out_dir / "cluster_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved cluster_results.json")
    print(f"  Saved pca_model.pt + cluster_centers.pt")

    # ── Summary (Paper Table 2 row) ──────────────────────────────────────
    print(f"\n  --- Paper Table 2 Row ---")
    print(f"  Layer  K*  Sil.   Intra    Inter")
    print(f"  {layer_idx:<6} {k_star:<3} {best_sil:.3f}  "
          f"{metrics['mean_intra_cosine']:.3f}    {metrics['mean_inter_cosine']:.3f}")

    return results


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 8: Main + CLI                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 6: Clustering & Subspace Mapping (Paper Section 3.7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/module6_clustering.py --layer 20 --n-pca-dims 50
  python src/module6_clustering.py --layer all
  python src/module6_clustering.py --layer 20 --k-min 1 --k-max 20
        """
    )

    parser.add_argument("--layer", type=str, default="20")
    parser.add_argument("--n-pca-dims", type=int, default=50)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=21)
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    if args.layer.lower() == "all":
        all_results = {}
        for layer_idx in DEFAULT_LAYERS:
            print(f"\n{'#'*60}")
            print(f"  LAYER {layer_idx}")
            print(f"{'#'*60}")
            try:
                results = run_clustering(
                    layer_idx, args.n_pca_dims, args.k_min, args.k_max,
                    args.dbscan_min_samples, save_plots=not args.no_plots
                )
                all_results[layer_idx] = results
            except (FileNotFoundError, ValueError) as e:
                print(f"  [SKIP] {e}")

        # Print combined Table 2
        if all_results:
            print(f"\n{'='*60}")
            print(f"  Paper Table 2: Cluster Analysis Per Layer")
            print(f"{'='*60}")
            print(f"  {'Layer':<8} {'K*':<4} {'Sil.':<8} {'Intra':<8} {'Inter':<8}")
            print(f"  {'-'*36}")
            for layer_idx, r in sorted(all_results.items()):
                print(f"  {layer_idx:<8} {r['kmeans']['k_star']:<4} "
                      f"{r['kmeans']['silhouette_score']:<8.3f} "
                      f"{r['cluster_metrics']['mean_intra_cosine']:<8.3f} "
                      f"{r['cluster_metrics']['mean_inter_cosine']:<8.3f}")
    else:
        run_clustering(
            int(args.layer), args.n_pca_dims, args.k_min, args.k_max,
            args.dbscan_min_samples, save_plots=not args.no_plots
        )

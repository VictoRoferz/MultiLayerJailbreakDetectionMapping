#!/usr/bin/env python3
"""
Stage 1 PCA diagnostic — class separability of Gemma activations.

Purpose: before training any generator, confirm that jailbreak vs non-jailbreak
activations (as labeled by Module 2) are geometrically distinguishable at the
target layers. If PCA shows no separation at any layer, the framework cannot
work and Module 2 needs debugging first.

Per layer L (default {10, 15, 20, 25}):
  1. 2D PCA scatter, colored by label.
  2. Silhouette score in PCA-50 space (label-separability in [-1, 1]).
  3. Class-conditional mean distance ||E[f|c=0] - E[f|c=1]|| / E[||f||].
  4. Number of PCs to reach 50/90/95% variance.

Cross-layer:
  5. Bar chart of silhouette by layer (paper Fig 4 analogue).
  6. Explained-variance curves, one line per layer.

Outputs:
  figures/pca/layer_{L}_scatter.png
  figures/pca/explained_variance.png
  figures/pca/silhouette_by_layer.png
  results/pca_diagnostic.json

Usage:
  python src/pca_diagnostic.py
  python src/pca_diagnostic.py --layers 10 15 20 25
  python src/pca_diagnostic.py --split val --artifacts-dir artifacts
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LAYERS = [10, 15, 20, 25]
PCA_FULL_COMPONENTS = 50
SCATTER_MAX_PER_CLASS = 3000
VARIANCE_TARGETS = [0.50, 0.90, 0.95]


# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------

def load_layer(artifacts_dir: Path, layer: int, split: str):
    """Load one layer's activations + labels. Drops label=-1 (judge-failed)."""
    path = artifacts_dir / f"layer_{layer}" / f"{split}_activations.pt"
    if not path.exists():
        print(f"  [SKIP] {path} missing - did Module 2 run on layer {layer}?")
        return None

    data = torch.load(path, weights_only=False)
    acts = data["activations"].to(torch.float32).numpy()
    labels = data["labels"].to(torch.long).numpy()

    keep = labels != -1
    dropped = int((~keep).sum())
    if dropped:
        print(f"  [INFO] Layer {layer}: dropped {dropped} samples with label=-1")
    acts, labels = acts[keep], labels[keep]

    return {"activations": acts, "labels": labels, "path": str(path)}


# ------------------------------------------------------------
# Per-layer metrics
# ------------------------------------------------------------

def analyze_layer(acts: np.ndarray, labels: np.ndarray, layer: int) -> dict:
    """Fit PCA, compute silhouette + class-mean distance + variance stats."""
    n_total, dim = acts.shape
    classes, counts = np.unique(labels, return_counts=True)
    class_counts = {int(c): int(n) for c, n in zip(classes, counts)}

    n_components = min(PCA_FULL_COMPONENTS, n_total - 1, dim)
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    proj = pca.fit_transform(acts)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    variance_targets = {}
    for t in VARIANCE_TARGETS:
        key = f"pcs_for_{int(t * 100)}pct"
        if cumulative[-1] >= t:
            variance_targets[key] = int(np.searchsorted(cumulative, t) + 1)
        else:
            variance_targets[key] = None

    if len(classes) >= 2 and all(c >= 2 for c in counts):
        sil = float(silhouette_score(proj, labels))
    else:
        sil = None

    mean_per_class = {int(c): acts[labels == c].mean(axis=0) for c in classes}
    if {0, 1}.issubset(mean_per_class.keys()):
        mean_diff = float(np.linalg.norm(mean_per_class[1] - mean_per_class[0]))
        mean_act_norm = float(np.linalg.norm(acts, axis=1).mean())
        normalized_mean_distance = mean_diff / max(mean_act_norm, 1e-8)
    else:
        normalized_mean_distance = None

    return {
        "layer": layer,
        "n_total": int(n_total),
        "dim": int(dim),
        "class_counts": class_counts,
        "pca_components": int(n_components),
        "explained_variance": explained.tolist(),
        "cumulative_variance": cumulative.tolist(),
        **variance_targets,
        "silhouette_pca50": sil,
        "normalized_mean_distance": normalized_mean_distance,
        "proj_2d": proj[:, :2],
    }


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_layer_scatter(metrics: dict, labels: np.ndarray, out_path: Path) -> None:
    proj_2d = metrics["proj_2d"]
    layer = metrics["layer"]

    benign = proj_2d[labels == 0]
    jailbreak = proj_2d[labels == 1]

    rng = np.random.default_rng(42)
    if len(benign) > SCATTER_MAX_PER_CLASS:
        benign = benign[rng.choice(len(benign), SCATTER_MAX_PER_CLASS, replace=False)]
    if len(jailbreak) > SCATTER_MAX_PER_CLASS:
        jailbreak = jailbreak[rng.choice(len(jailbreak), SCATTER_MAX_PER_CLASS, replace=False)]

    var1 = metrics["explained_variance"][0] * 100
    var2 = metrics["explained_variance"][1] * 100 if len(metrics["explained_variance"]) > 1 else 0.0
    sil = metrics["silhouette_pca50"]
    dmean = metrics["normalized_mean_distance"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(benign[:, 0], benign[:, 1],
               s=10, alpha=0.4, c="steelblue",
               label=f"label=0 benign/refused (n={metrics['class_counts'].get(0, 0)})")
    ax.scatter(jailbreak[:, 0], jailbreak[:, 1],
               s=10, alpha=0.5, c="crimson",
               label=f"label=1 jailbreak (n={metrics['class_counts'].get(1, 0)})")

    ax.set_xlabel(f"PC1 ({var1:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% var)")

    title = f"Layer {layer} - PCA class separability"
    subtitle_parts = []
    if sil is not None:
        subtitle_parts.append(f"silhouette(PCA50)={sil:.3f}")
    if dmean is not None:
        subtitle_parts.append(f"||dmu||/||f||={dmean:.3f}")
    if subtitle_parts:
        title += "\n" + "  |  ".join(subtitle_parts)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_explained_variance(all_metrics: list, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in all_metrics:
        cum = np.array(m["cumulative_variance"])
        xs = np.arange(1, len(cum) + 1)
        ax.plot(xs, cum, label=f"layer {m['layer']}")

    ax.axhline(0.90, linestyle="--", color="gray", alpha=0.5, label="90% target")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Explained variance by PCs, per layer")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_by_layer(all_metrics: list, out_path: Path) -> None:
    layers = [m["layer"] for m in all_metrics]
    sils = [m["silhouette_pca50"] if m["silhouette_pca50"] is not None else 0.0
            for m in all_metrics]
    dists = [m["normalized_mean_distance"] if m["normalized_mean_distance"] is not None else 0.0
             for m in all_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar([str(l) for l in layers], sils, color="steelblue")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Silhouette score (PCA50)")
    ax1.set_title("Class separability by layer")
    ax1.axhline(0.0, color="gray", linewidth=0.8)
    ax1.grid(alpha=0.3, axis="y")
    for bar, s in zip(bars, sils):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{s:.3f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar([str(l) for l in layers], dists, color="crimson")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("||E[f|c=1] - E[f|c=0]|| / E[||f||]")
    ax2.set_title("Normalized class-mean distance by layer")
    ax2.grid(alpha=0.3, axis="y")
    for bar, d in zip(bars2, dists):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{d:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1 PCA diagnostic")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                        help=f"Layers to analyze (default: {DEFAULT_LAYERS})")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--figures-dir", type=str, default="figures/pca")
    parser.add_argument("--results-path", type=str, default="results/pca_diagnostic.json")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Stage 1 PCA diagnostic - split={args.split}, layers={args.layers}")

    all_metrics = []
    for layer in args.layers:
        print(f"\n-- Layer {layer} --")
        data = load_layer(artifacts_dir, layer, args.split)
        if data is None:
            continue

        metrics = analyze_layer(data["activations"], data["labels"], layer)
        metrics["source_path"] = data["path"]

        scatter_path = figures_dir / f"layer_{layer}_scatter.png"
        plot_layer_scatter(metrics, data["labels"], scatter_path)
        print(f"  scatter  -> {scatter_path}")

        all_metrics.append(metrics)

        sil = metrics["silhouette_pca50"]
        dmean = metrics["normalized_mean_distance"]
        pcs90 = metrics.get("pcs_for_90pct")
        if sil is not None:
            print(f"  silhouette(PCA50) = {sil:.4f}")
        else:
            print("  silhouette(PCA50) = n/a (need both classes with >=2 samples)")
        if dmean is not None:
            print(f"  ||dmu||/||f||     = {dmean:.4f}")
        else:
            print("  ||dmu||/||f||     = n/a (need both classes)")
        print(f"  PCs for 90% var   = {pcs90}")

    if not all_metrics:
        print("\nNo layers successfully analyzed. Exiting.")
        return

    ev_path = figures_dir / "explained_variance.png"
    plot_explained_variance(all_metrics, ev_path)
    print(f"\nExplained variance -> {ev_path}")

    sil_path = figures_dir / "silhouette_by_layer.png"
    plot_silhouette_by_layer(all_metrics, sil_path)
    print(f"Silhouette by layer -> {sil_path}")

    best_sil = max(
        (m for m in all_metrics if m["silhouette_pca50"] is not None),
        key=lambda m: m["silhouette_pca50"],
        default=None,
    )
    best_dist = max(
        (m for m in all_metrics if m["normalized_mean_distance"] is not None),
        key=lambda m: m["normalized_mean_distance"],
        default=None,
    )

    serializable = []
    for m in all_metrics:
        m_out = {k: v for k, v in m.items() if k != "proj_2d"}
        serializable.append(m_out)

    output = {
        "split": args.split,
        "layers_analyzed": [m["layer"] for m in all_metrics],
        "best_layer_by_silhouette": best_sil["layer"] if best_sil else None,
        "best_silhouette": best_sil["silhouette_pca50"] if best_sil else None,
        "best_layer_by_class_mean_distance": best_dist["layer"] if best_dist else None,
        "best_class_mean_distance": best_dist["normalized_mean_distance"] if best_dist else None,
        "per_layer": serializable,
    }

    with results_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Metrics JSON     -> {results_path}")

    print("\nSummary:")
    if output["best_silhouette"] is not None:
        print(f"  Best layer by silhouette:          "
              f"{output['best_layer_by_silhouette']} (sil={output['best_silhouette']:.4f})")
    if output["best_class_mean_distance"] is not None:
        print(f"  Best layer by class-mean distance: "
              f"{output['best_layer_by_class_mean_distance']} "
              f"(d={output['best_class_mean_distance']:.4f})")

    if output["best_silhouette"] is not None and output["best_silhouette"] < 0.05:
        print("\n[WARN] All silhouette scores are very low (<0.05).")
        print("       Module 2 labels may be too noisy or class-mean signal too weak.")
        print("       Inspect artifacts/labeled_data/full_dataset.pt before training Module 3.")


if __name__ == "__main__":
    main()

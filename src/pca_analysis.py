#!/usr/bin/env python3
"""
pca_analysis.py

PCA-based analysis for evaluating generator quality and activation-space structure.

Three analyses:
    1. Raw activation space — PCA-2D scatter of benign vs harmful (baseline)
    2. Perturbation effect  — Does the generator push benign toward harmful?
    3. Cross-layer comparison — Which layers show the cleanest structure?

Each analysis includes a baseline comparison (random projection / random perturbation)
to demonstrate that learned structure is meaningful.

Usage:
    python src/pca_analysis.py --layer 20 --analysis raw
    python src/pca_analysis.py --layer 20 --analysis perturbation --architecture mlp --epsilon 0.15
    python src/pca_analysis.py --layer all --analysis cross-layer
    python src/pca_analysis.py --layer 20 --analysis all --architecture cvae
"""

import argparse
import json
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Imports from existing modules                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

from module3_perturbation_generator import (
    PerturbationGenerator,
    CVAEPerturbationGenerator,
    FlowMatchingDenoiser,
    RewardModel,
    apply_norm_constraint,
    get_device,
    DEFAULT_LAYERS,
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Optional dependencies — degrade gracefully                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.random_projection import GaussianRandomProjection
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[!] scikit-learn not installed. Install with: pip install scikit-learn")
    print("    PCA analysis requires scikit-learn. Exiting.")
    sys.exit(1)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 1: Data Loading                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_activations(layer_idx: int) -> dict:
    """
    Load activations.pt for a given layer.

    Returns dict with keys:
        benign:  [N_benign, 2304] tensor
        harmful: [N_harmful, 2304] tensor
        all:     [N_total, 2304] tensor
        labels:  [N_total] tensor (0.0=benign, 1.0=harmful)
    """
    # Try split files first (train/val/test), fall back to single file
    base = Path("artifacts") / f"layer_{layer_idx}"

    # Check for split files
    if (base / "train_activations.pt").exists():
        splits = {}
        for split_name in ["train", "val", "test"]:
            path = base / f"{split_name}_activations.pt"
            if path.exists():
                splits[split_name] = torch.load(path, weights_only=False)
        # Combine all splits for visualization
        all_acts = torch.cat([s["activations"] for s in splits.values()])
        all_labels = torch.cat([s["labels"] for s in splits.values()])
    elif (base / "activations.pt").exists():
        data = torch.load(base / "activations.pt", weights_only=False)
        all_acts = data["activations"]
        all_labels = data["labels"]
    else:
        raise FileNotFoundError(
            f"No activations found at {base}/. "
            f"Run data.py or Extraction.py first."
        )

    benign_mask = all_labels == 0.0
    harmful_mask = all_labels == 1.0

    return {
        "benign": all_acts[benign_mask],
        "harmful": all_acts[harmful_mask],
        "all": all_acts,
        "labels": all_labels,
        "layer": layer_idx,
    }


def load_generator(layer_idx: int, architecture: str = "mlp",
                   device: torch.device = torch.device("cpu")):
    """
    Load a trained generator from artifacts.

    Returns (generator, checkpoint_dict) or raises FileNotFoundError.
    """
    if architecture == "cvae":
        base = Path("artifacts") / f"layer_{layer_idx}" / "cvae"
    else:
        base = Path("artifacts") / f"layer_{layer_idx}"

    ckpt_path = base / "generator.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No generator found at {ckpt_path}. "
            f"Run module3_perturbation_generator.py --layer {layer_idx} "
            f"--architecture {architecture} first."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    activation_dim = ckpt["activation_dim"]
    z_dim = ckpt["z_dim"]

    if architecture == "cvae":
        gen = CVAEPerturbationGenerator(activation_dim, z_dim=z_dim)
    else:
        gen = PerturbationGenerator(activation_dim, z_dim=z_dim)

    gen.load_state_dict(ckpt["model_state_dict"])
    gen.to(device)
    gen.eval()
    return gen, ckpt


def load_fw_generators(layer_idx: int, architecture: str = "mlp",
                       device: torch.device = torch.device("cpu")) -> list:
    """Load all Frank-Wolfe generator checkpoints. Returns list of generators."""
    if architecture == "cvae":
        base = Path("artifacts") / f"layer_{layer_idx}" / "cvae"
    else:
        base = Path("artifacts") / f"layer_{layer_idx}"

    generators = []
    for i in range(1, 10):  # up to 9 FW iterations
        path = base / f"generator_fw_{i}.pt"
        if not path.exists():
            break
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if architecture == "cvae":
            gen = CVAEPerturbationGenerator(ckpt["activation_dim"], z_dim=ckpt["z_dim"])
        else:
            gen = PerturbationGenerator(ckpt["activation_dim"], z_dim=ckpt["z_dim"])
        gen.load_state_dict(ckpt["model_state_dict"])
        gen.to(device)
        gen.eval()
        generators.append(gen)

    return generators


def generate_perturbations(generator, benign_acts: torch.Tensor,
                           device: torch.device, n_samples: int = 300,
                           epsilon: float = 0.15):
    """
    Generate delta_f perturbations using a trained generator.

    Returns:
        delta_f:    [n_samples, D] perturbation vectors
        perturbed:  [n_samples, D] perturbed activations (benign + delta_f)
        benign_sub: [n_samples, D] the benign activations used
    """
    n = min(n_samples, len(benign_acts))
    idx = torch.randperm(len(benign_acts))[:n]
    benign_sub = benign_acts[idx].to(device)

    z = torch.randn(n, generator.z_dim, device=device)
    with torch.no_grad():
        delta_f = generator(z, benign_sub)
        delta_f = apply_norm_constraint(delta_f, benign_sub, epsilon)

    perturbed = benign_sub + delta_f
    return delta_f.cpu(), perturbed.cpu(), benign_sub.cpu()


def ensure_figures_dir(layer_idx: int) -> Path:
    """Create and return the figures directory for a layer."""
    fig_dir = Path("artifacts") / f"layer_{layer_idx}" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def ensure_global_figures_dir() -> Path:
    """Create and return the top-level figures directory."""
    fig_dir = Path("artifacts") / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 2: Analysis 1 — Raw Activation Space                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def analysis_raw_activation_space(layer_idx: int, save_plots: bool = True) -> dict:
    """
    PCA-2D scatter of benign vs harmful activations.

    This is the BASELINE analysis — shows whether the activation space has
    natural structure BEFORE any perturbation. If benign and harmful already
    separate clearly, the reward model's job is easy.

    Also includes a random-projection baseline to show PCA captures more
    structure than chance.

    Returns dict of metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Analysis 1: Raw Activation Space — Layer {layer_idx}")
    print(f"{'='*60}")

    data = load_activations(layer_idx)
    benign = data["benign"].numpy()
    harmful = data["harmful"].numpy()
    all_acts = data["all"].numpy()
    labels = data["labels"].numpy()

    print(f"  Benign samples:  {len(benign)}")
    print(f"  Harmful samples: {len(harmful)}")

    # ── PCA Analysis ──────────────────────────────────────────────────────
    n_components_full = min(50, len(all_acts), all_acts.shape[1])
    pca_full = PCA(n_components=n_components_full)
    pca_full.fit(all_acts)

    # Explained variance
    var_pc1 = pca_full.explained_variance_ratio_[0]
    var_pc2 = pca_full.explained_variance_ratio_[1]
    var_cumulative_50 = np.sum(pca_full.explained_variance_ratio_[:n_components_full])

    print(f"\n  Explained variance:")
    print(f"    PC1:           {var_pc1:.4f} ({var_pc1*100:.1f}%)")
    print(f"    PC2:           {var_pc2:.4f} ({var_pc2*100:.1f}%)")
    print(f"    Top {n_components_full} PCs:    {var_cumulative_50:.4f} ({var_cumulative_50*100:.1f}%)")

    # Project to 2D
    pca_2d = PCA(n_components=2)
    proj_2d = pca_2d.fit_transform(all_acts)
    benign_2d = proj_2d[labels == 0.0]
    harmful_2d = proj_2d[labels == 1.0]

    # Silhouette score (in 50-PC space for robustness)
    proj_50 = pca_full.transform(all_acts)
    sil_score = silhouette_score(proj_50, labels) if len(np.unique(labels)) > 1 else 0.0
    print(f"\n  Silhouette score (50-PC): {sil_score:.4f}")

    # ── Random Projection Baseline ────────────────────────────────────────
    rp = GaussianRandomProjection(n_components=2, random_state=42)
    proj_random = rp.fit_transform(all_acts)
    benign_random = proj_random[labels == 0.0]
    harmful_random = proj_random[labels == 1.0]

    # Silhouette on random 50-dim projection
    rp_50 = GaussianRandomProjection(n_components=min(50, all_acts.shape[1]), random_state=42)
    proj_random_50 = rp_50.fit_transform(all_acts)
    sil_random = silhouette_score(proj_random_50, labels) if len(np.unique(labels)) > 1 else 0.0
    print(f"  Silhouette score (random 50-dim): {sil_random:.4f}")
    print(f"  PCA advantage: {sil_score - sil_random:+.4f}")

    # ── Plotting ──────────────────────────────────────────────────────────
    if save_plots and HAS_PLOTTING:
        fig_dir = ensure_figures_dir(layer_idx)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: PCA projection
        ax = axes[0]
        ax.scatter(benign_2d[:, 0], benign_2d[:, 1],
                   c="steelblue", alpha=0.4, s=15, label=f"Benign (n={len(benign_2d)})")
        ax.scatter(harmful_2d[:, 0], harmful_2d[:, 1],
                   c="crimson", alpha=0.4, s=15, label=f"Harmful (n={len(harmful_2d)})")
        ax.set_xlabel(f"PC1 ({var_pc1*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var_pc2*100:.1f}% var)")
        ax.set_title(f"PCA — Layer {layer_idx}  (sil={sil_score:.3f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Right: Random projection baseline
        ax = axes[1]
        ax.scatter(benign_random[:, 0], benign_random[:, 1],
                   c="steelblue", alpha=0.4, s=15, label="Benign")
        ax.scatter(harmful_random[:, 0], harmful_random[:, 1],
                   c="crimson", alpha=0.4, s=15, label="Harmful")
        ax.set_xlabel("Random dim 1")
        ax.set_ylabel("Random dim 2")
        ax.set_title(f"Random Projection (baseline, sil={sil_random:.3f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Analysis 1: Raw Activation Space — Layer {layer_idx}", fontsize=13)
        plt.tight_layout()
        out_path = fig_dir / "raw_activation_pca.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved figure: {out_path}")

    # ── Results ───────────────────────────────────────────────────────────
    results = {
        "layer": layer_idx,
        "n_benign": len(benign),
        "n_harmful": len(harmful),
        "explained_variance_pc1": float(var_pc1),
        "explained_variance_pc2": float(var_pc2),
        "explained_variance_cumulative_50pc": float(var_cumulative_50),
        "silhouette_score_pca50": float(sil_score),
        "silhouette_score_random50": float(sil_random),
        "pca_advantage": float(sil_score - sil_random),
    }
    return results


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 3: Analysis 2 — Perturbation Effect                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def analysis_perturbation_effect(
    layer_idx: int,
    architecture: str = "mlp",
    epsilon: float = 0.15,
    n_samples: int = 300,
    save_plots: bool = True,
) -> dict:
    """
    Visualize how the trained generator moves benign activations.

    Three groups in PCA-2D:
        - Benign (blue): original clean activations
        - Harmful (red): real harmful activations (ground truth target)
        - Perturbed (orange): benign + delta_f from generator

    Also includes a RANDOM PERTURBATION BASELINE: random vectors with the
    same norm budget (epsilon * ||f_L||). This shows the trained generator
    does better than random noise at pushing toward harmful.

    Returns dict of metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Analysis 2: Perturbation Effect — Layer {layer_idx} ({architecture})")
    print(f"{'='*60}")

    device = get_device()
    data = load_activations(layer_idx)

    # Load generator
    gen, _ = load_generator(layer_idx, architecture, device)
    print(f"  Generator loaded: {architecture} (z_dim={gen.z_dim})")

    # Generate perturbations
    delta_f, perturbed, benign_sub = generate_perturbations(
        gen, data["benign"], device, n_samples, epsilon
    )

    # Random perturbation baseline (same norm budget)
    benign_for_random = benign_sub.clone()
    random_delta = torch.randn_like(benign_for_random)
    # Scale to same norm: epsilon * ||f_L||
    f_norms = benign_for_random.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    random_delta = random_delta / random_delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    random_delta = random_delta * epsilon * f_norms
    random_perturbed = benign_for_random + random_delta

    # ── Metrics ───────────────────────────────────────────────────────────
    harmful_np = data["harmful"].numpy()
    benign_np = benign_sub.numpy()
    perturbed_np = perturbed.numpy()
    random_perturbed_np = random_perturbed.numpy()

    # Centroids
    benign_centroid = benign_np.mean(axis=0)
    harmful_centroid = harmful_np.mean(axis=0)
    direction = harmful_centroid - benign_centroid
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

    # Directional alignment: how much does delta_f point toward harmful?
    delta_np = delta_f.numpy()
    delta_norms = np.linalg.norm(delta_np, axis=1, keepdims=True) + 1e-8
    delta_normalized = delta_np / delta_norms
    alignment = (delta_normalized @ direction_norm).mean()

    random_delta_np = random_delta.numpy()
    random_norms = np.linalg.norm(random_delta_np, axis=1, keepdims=True) + 1e-8
    random_normalized = random_delta_np / random_norms
    random_alignment = (random_normalized @ direction_norm).mean()

    print(f"\n  Directional alignment (cosine with harmful direction):")
    print(f"    Trained generator: {alignment:.4f}")
    print(f"    Random baseline:   {random_alignment:.4f}")

    # Fraction of perturbed closer to harmful centroid
    dist_to_harmful_orig = np.linalg.norm(benign_np - harmful_centroid, axis=1)
    dist_to_harmful_pert = np.linalg.norm(perturbed_np - harmful_centroid, axis=1)
    dist_to_harmful_rand = np.linalg.norm(random_perturbed_np - harmful_centroid, axis=1)

    frac_closer_trained = (dist_to_harmful_pert < dist_to_harmful_orig).mean()
    frac_closer_random = (dist_to_harmful_rand < dist_to_harmful_orig).mean()

    print(f"\n  Fraction moved closer to harmful centroid:")
    print(f"    Trained generator: {frac_closer_trained:.4f} ({frac_closer_trained*100:.1f}%)")
    print(f"    Random baseline:   {frac_closer_random:.4f} ({frac_closer_random*100:.1f}%)")

    # Mean perturbation norm relative to activation norm
    mean_delta_ratio = (delta_f.norm(dim=-1) / benign_sub.norm(dim=-1).clamp(min=1e-8)).mean().item()
    print(f"\n  Mean ||delta_f|| / ||f_L||: {mean_delta_ratio:.4f} (target: {epsilon})")

    # ── PCA + Plotting ────────────────────────────────────────────────────
    if save_plots and HAS_PLOTTING:
        fig_dir = ensure_figures_dir(layer_idx)

        # Fit PCA on all data jointly
        all_data = np.vstack([benign_np, harmful_np, perturbed_np, random_perturbed_np])
        pca = PCA(n_components=2)
        pca.fit(all_data)

        benign_2d = pca.transform(benign_np)
        harmful_2d = pca.transform(harmful_np)
        perturbed_2d = pca.transform(perturbed_np)
        random_2d = pca.transform(random_perturbed_np)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Trained generator
        ax = axes[0]
        ax.scatter(benign_2d[:, 0], benign_2d[:, 1],
                   c="steelblue", alpha=0.3, s=12, label="Benign", zorder=2)
        ax.scatter(harmful_2d[:, 0], harmful_2d[:, 1],
                   c="crimson", alpha=0.3, s=12, label="Harmful", zorder=2)
        ax.scatter(perturbed_2d[:, 0], perturbed_2d[:, 1],
                   c="darkorange", alpha=0.4, s=12, label="Perturbed", zorder=3)

        # Draw arrows for a subset
        n_arrows = min(40, len(benign_2d))
        arrow_idx = np.random.choice(len(benign_2d), n_arrows, replace=False)
        for i in arrow_idx:
            ax.annotate("",
                        xy=perturbed_2d[i], xytext=benign_2d[i],
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3, lw=0.5))

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"Trained Generator ({architecture})\n"
                     f"align={alignment:.3f}, closer={frac_closer_trained:.1%}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Right: Random baseline
        ax = axes[1]
        ax.scatter(benign_2d[:, 0], benign_2d[:, 1],
                   c="steelblue", alpha=0.3, s=12, label="Benign", zorder=2)
        ax.scatter(harmful_2d[:, 0], harmful_2d[:, 1],
                   c="crimson", alpha=0.3, s=12, label="Harmful", zorder=2)
        ax.scatter(random_2d[:, 0], random_2d[:, 1],
                   c="mediumpurple", alpha=0.4, s=12, label="Random perturbed", zorder=3)

        for i in arrow_idx:
            ax.annotate("",
                        xy=random_2d[i], xytext=benign_2d[i],
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3, lw=0.5))

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"Random Perturbation (baseline)\n"
                     f"align={random_alignment:.3f}, closer={frac_closer_random:.1%}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Analysis 2: Perturbation Effect — Layer {layer_idx}", fontsize=13)
        plt.tight_layout()
        out_path = fig_dir / f"perturbation_effect_{architecture}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved figure: {out_path}")

    # ── Results ───────────────────────────────────────────────────────────
    results = {
        "layer": layer_idx,
        "architecture": architecture,
        "epsilon": epsilon,
        "n_samples": len(delta_f),
        "directional_alignment_trained": float(alignment),
        "directional_alignment_random": float(random_alignment),
        "frac_closer_to_harmful_trained": float(frac_closer_trained),
        "frac_closer_to_harmful_random": float(frac_closer_random),
        "mean_delta_norm_ratio": float(mean_delta_ratio),
    }
    return results


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 4: Analysis 3 — Cross-Layer Comparison                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def analysis_cross_layer(
    layers: list = None,
    save_plots: bool = True,
) -> dict:
    """
    Compare activation-space structure across layers.

    For each layer:
        - PCA-2D scatter (benign vs harmful)
        - Silhouette score

    Produces:
        - 2x2 subplot grid of PCA scatters
        - Bar chart of silhouette scores across layers

    This fills Paper Figure 4 (silhouette per layer).

    Returns dict of per-layer metrics.
    """
    if layers is None:
        layers = list(DEFAULT_LAYERS)

    print(f"\n{'='*60}")
    print(f"  Analysis 3: Cross-Layer Comparison — Layers {layers}")
    print(f"{'='*60}")

    per_layer = {}
    projections = {}

    for layer_idx in layers:
        try:
            data = load_activations(layer_idx)
        except FileNotFoundError as e:
            print(f"  [SKIP] Layer {layer_idx}: {e}")
            continue

        all_acts = data["all"].numpy()
        labels = data["labels"].numpy()

        # PCA
        n_comp = min(50, len(all_acts), all_acts.shape[1])
        pca = PCA(n_components=n_comp)
        proj = pca.fit_transform(all_acts)

        # 2D for plotting
        pca_2d = PCA(n_components=2)
        proj_2d = pca_2d.fit_transform(all_acts)

        # Silhouette
        sil = silhouette_score(proj, labels) if len(np.unique(labels)) > 1 else 0.0

        var_pc1 = pca_2d.explained_variance_ratio_[0]
        var_pc2 = pca_2d.explained_variance_ratio_[1]
        var_cum = np.sum(pca.explained_variance_ratio_)

        per_layer[layer_idx] = {
            "silhouette_score": float(sil),
            "explained_variance_pc1": float(var_pc1),
            "explained_variance_pc2": float(var_pc2),
            "explained_variance_cumulative": float(var_cum),
            "n_benign": int((labels == 0.0).sum()),
            "n_harmful": int((labels == 1.0).sum()),
        }

        projections[layer_idx] = {
            "proj_2d": proj_2d,
            "labels": labels,
            "var_pc1": var_pc1,
            "var_pc2": var_pc2,
            "sil": sil,
        }

        print(f"  Layer {layer_idx}: sil={sil:.4f}, "
              f"var(PC1+PC2)={var_pc1+var_pc2:.4f}, "
              f"var(top-{n_comp})={var_cum:.4f}")

    if not per_layer:
        print("  No layers with data found.")
        return {}

    # Find best layer
    best_layer = max(per_layer, key=lambda l: per_layer[l]["silhouette_score"])
    print(f"\n  Best layer by silhouette: {best_layer} "
          f"(sil={per_layer[best_layer]['silhouette_score']:.4f})")

    # ── Plotting ──────────────────────────────────────────────────────────
    if save_plots and HAS_PLOTTING and projections:
        fig_dir = ensure_global_figures_dir()

        # Figure 1: 2x2 PCA scatter grid
        n_plots = len(projections)
        n_cols = 2
        n_rows = (n_plots + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for ax_idx, (layer_idx, pdata) in enumerate(sorted(projections.items())):
            row, col = divmod(ax_idx, n_cols)
            ax = axes[row, col]

            proj_2d = pdata["proj_2d"]
            labels = pdata["labels"]

            benign_2d = proj_2d[labels == 0.0]
            harmful_2d = proj_2d[labels == 1.0]

            ax.scatter(benign_2d[:, 0], benign_2d[:, 1],
                       c="steelblue", alpha=0.4, s=10, label="Benign")
            ax.scatter(harmful_2d[:, 0], harmful_2d[:, 1],
                       c="crimson", alpha=0.4, s=10, label="Harmful")
            ax.set_xlabel(f"PC1 ({pdata['var_pc1']*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pdata['var_pc2']*100:.1f}%)")
            ax.set_title(f"Layer {layer_idx} (sil={pdata['sil']:.3f})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for ax_idx in range(len(projections), n_rows * n_cols):
            row, col = divmod(ax_idx, n_cols)
            axes[row, col].set_visible(False)

        fig.suptitle("Cross-Layer PCA Comparison: Benign vs Harmful", fontsize=13)
        plt.tight_layout()
        out_path = fig_dir / "cross_layer_pca.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved figure: {out_path}")

        # Figure 2: Silhouette bar chart (Paper Figure 4)
        fig, ax = plt.subplots(figsize=(8, 5))
        sorted_layers = sorted(per_layer.keys())
        sil_scores = [per_layer[l]["silhouette_score"] for l in sorted_layers]
        colors = ["gold" if l == best_layer else "steelblue" for l in sorted_layers]

        bars = ax.bar([str(l) for l in sorted_layers], sil_scores, color=colors, edgecolor="black")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Benign/Harmful Separation by Layer\n"
                     "(higher = better separated clusters)")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars, sil_scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        out_path = fig_dir / "cross_layer_silhouette.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved figure: {out_path}")

    results = {
        "layers": per_layer,
        "best_layer": best_layer,
    }
    return results


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 5: Results Saving                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def save_results(results: dict, layer_idx: int, filename: str = "pca_results.json"):
    """Save analysis results to JSON."""
    out_dir = Path("artifacts") / f"layer_{layer_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # Make JSON-serializable
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Saved results: {out_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 6: Main + CLI                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

ANALYSIS_CHOICES = ["all", "raw", "perturbation", "cross-layer"]


def main(args):
    """Dispatch to requested analyses."""
    analyses = [args.analysis] if args.analysis != "all" else ["raw", "perturbation", "cross-layer"]
    save_plots = not args.no_plots
    all_results = {}

    for analysis in analyses:
        if analysis == "raw":
            if args.layer_idx == "all":
                for layer in DEFAULT_LAYERS:
                    results = analysis_raw_activation_space(layer, save_plots=save_plots)
                    all_results[f"raw_layer_{layer}"] = results
                    save_results(results, layer)
            else:
                results = analysis_raw_activation_space(args.layer_idx, save_plots=save_plots)
                all_results["raw"] = results
                save_results(results, args.layer_idx)

        elif analysis == "perturbation":
            if args.layer_idx == "all":
                for layer in DEFAULT_LAYERS:
                    try:
                        results = analysis_perturbation_effect(
                            layer, args.architecture, args.epsilon,
                            args.n_samples, save_plots=save_plots
                        )
                        all_results[f"perturbation_layer_{layer}"] = results
                        save_results(results, layer)
                    except FileNotFoundError as e:
                        print(f"  [SKIP] Layer {layer}: {e}")
            else:
                results = analysis_perturbation_effect(
                    args.layer_idx, args.architecture, args.epsilon,
                    args.n_samples, save_plots=save_plots
                )
                all_results["perturbation"] = results
                save_results(results, args.layer_idx)

        elif analysis == "cross-layer":
            layers = list(DEFAULT_LAYERS) if args.layer_idx == "all" else [args.layer_idx]
            if len(layers) == 1:
                print("  [NOTE] Cross-layer requires multiple layers. Using all default layers.")
                layers = list(DEFAULT_LAYERS)
            results = analysis_cross_layer(layers, save_plots=save_plots)
            all_results["cross_layer"] = results
            # Save to global results
            out_dir = Path("artifacts")
            out_dir.mkdir(exist_ok=True)
            with open(out_dir / "cross_layer_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Saved: artifacts/cross_layer_results.json")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PCA Analysis Complete")
    print(f"{'='*60}")
    for key, res in all_results.items():
        if isinstance(res, dict) and "silhouette_score_pca50" in res:
            print(f"  {key}: sil={res['silhouette_score_pca50']:.4f}")
        elif isinstance(res, dict) and "directional_alignment_trained" in res:
            print(f"  {key}: align={res['directional_alignment_trained']:.4f}, "
                  f"closer={res['frac_closer_to_harmful_trained']:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCA Analysis: Evaluate generator quality and activation-space structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pca_analysis.py --layer 20 --analysis raw
  python src/pca_analysis.py --layer 20 --analysis perturbation --architecture mlp
  python src/pca_analysis.py --layer all --analysis cross-layer
  python src/pca_analysis.py --layer 20 --analysis all --architecture cvae --epsilon 0.1
        """
    )

    parser.add_argument("--layer", type=str, default="20",
                        help="Layer index (e.g. 20) or 'all' for layers 10,15,20,25")
    parser.add_argument("--analysis", type=str, default="all",
                        choices=ANALYSIS_CHOICES,
                        help="Which analysis to run: raw, perturbation, cross-layer, or all")
    parser.add_argument("--architecture", type=str, default="mlp",
                        choices=["mlp", "cvae"],
                        help="Generator architecture for perturbation analysis")
    parser.add_argument("--epsilon", type=float, default=0.15,
                        help="Norm constraint ratio: ||delta_f|| <= eps * ||f_L||")
    parser.add_argument("--n-samples", type=int, default=300,
                        help="Number of perturbation samples to generate")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable plot generation (print metrics only)")

    args = parser.parse_args()

    # Parse --layer
    if args.layer.lower() == "all":
        args.layer_idx = "all"
    else:
        args.layer_idx = int(args.layer)

    main(args)

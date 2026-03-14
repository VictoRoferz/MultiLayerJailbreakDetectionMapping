#!/usr/bin/env python3
"""
module7_detector.py

Module 7 — Detector Construction (Paper Section 3.8)

Builds a prompt-agnostic jailbreak detector from the jailbreak activation
subspace mapped by Module 6.

Detection rule:
    For input x, extract f_L(x), project into PCA subspace, compute:
        d(x) = min_k cosdist(PCA(f_L(x)), mu_k)
    Flag as jailbreak if d(x) < theta.

Threshold theta is tuned on held-out benign data for FPR < 2%.

Baselines (Paper Table 4):
    - String-match detector: check output for refusal phrases
    - Random-direction detector: use random vectors as "cluster centers"

Usage:
    python src/module7_detector.py --layer 20 --fpr-target 0.02
    python src/module7_detector.py --layer 20 --evaluate-baselines
    python src/module7_detector.py --layer all
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from module3_perturbation_generator import (
    DEFAULT_LAYERS,
    REFUSAL_PHRASES,
)

try:
    from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
    from sklearn.metrics.pairwise import cosine_distances
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 1: Detector Class                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SubspaceDetector:
    """
    Prompt-agnostic jailbreak detector based on activation subspace membership.

    Detection: d(x) = min_k cosdist(PCA(f_L(x)), mu_k)
    Flag as jailbreak if d(x) < theta.
    """

    def __init__(self, pca_components: np.ndarray, pca_mean: np.ndarray,
                 cluster_centers: np.ndarray, threshold: float = 0.5):
        self.pca_components = pca_components  # [n_pca, D]
        self.pca_mean = pca_mean              # [D]
        self.centers = cluster_centers         # [K*, n_pca]
        self.threshold = threshold

    def pca_transform(self, activations: np.ndarray) -> np.ndarray:
        """Project activations into PCA space."""
        centered = activations - self.pca_mean
        return centered @ self.pca_components.T

    def score(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute detection scores (lower = more jailbreak-like).
        Returns array of min cosine distances to cluster centers.
        """
        projected = self.pca_transform(activations)
        cos_dist = cosine_distances(projected, self.centers)  # [N, K*]
        return cos_dist.min(axis=1)  # [N]

    def predict(self, activations: np.ndarray) -> np.ndarray:
        """Predict jailbreak (1) or benign (0)."""
        scores = self.score(activations)
        return (scores < self.threshold).astype(int)

    @classmethod
    def from_artifacts(cls, layer_idx: int, threshold: float = 0.5):
        """Load detector from Module 6 artifacts."""
        base = Path("artifacts") / f"layer_{layer_idx}"

        pca_data = torch.load(base / "pca_model.pt", weights_only=False)
        cluster_data = torch.load(base / "cluster_centers.pt", weights_only=False)

        return cls(
            pca_components=pca_data["pca_components"].numpy(),
            pca_mean=pca_data["pca_mean"].numpy(),
            cluster_centers=cluster_data["centers"].numpy(),
            threshold=threshold,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 2: Threshold Tuning                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def tune_threshold(detector: SubspaceDetector, benign_activations: np.ndarray,
                   fpr_target: float = 0.02) -> float:
    """
    Tune detection threshold theta for target FPR on benign data.

    Since benign inputs should NOT be flagged, we want:
        FPR = fraction of benign flagged as jailbreak < fpr_target

    We find the threshold such that at most fpr_target fraction of benign
    scores fall below it.
    """
    scores = detector.score(benign_activations)

    # theta = percentile corresponding to fpr_target
    # Lower scores = more jailbreak-like, so we want the fpr_target-th percentile
    theta = float(np.percentile(scores, fpr_target * 100))

    actual_fpr = (scores < theta).mean()
    print(f"  Threshold tuning:")
    print(f"    Target FPR: {fpr_target:.4f}")
    print(f"    Theta:      {theta:.4f}")
    print(f"    Actual FPR: {actual_fpr:.4f}")

    return theta


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 3: Evaluation                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def evaluate_detector(detector: SubspaceDetector,
                      benign_acts: np.ndarray,
                      harmful_acts: np.ndarray) -> dict:
    """
    Evaluate detector on benign + harmful activations.

    Returns dict with TPR, FPR, AUC, F1.
    """
    benign_scores = detector.score(benign_acts)
    harmful_scores = detector.score(harmful_acts)

    benign_preds = (benign_scores < detector.threshold).astype(int)
    harmful_preds = (harmful_scores < detector.threshold).astype(int)

    fpr = benign_preds.mean()
    tpr = harmful_preds.mean()

    # For AUC: combine all scores and labels
    all_scores = np.concatenate([benign_scores, harmful_scores])
    all_labels = np.concatenate([
        np.zeros(len(benign_scores)),
        np.ones(len(harmful_scores)),
    ])
    # Flip scores since lower = more jailbreak-like
    try:
        auc = float(roc_auc_score(all_labels, -all_scores))
    except ValueError:
        auc = 0.5

    # F1
    all_preds = np.concatenate([benign_preds, harmful_preds])
    f1 = float(f1_score(all_labels, all_preds, zero_division=0))

    return {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "auc": auc,
        "f1": f1,
        "threshold": detector.threshold,
        "n_benign": len(benign_acts),
        "n_harmful": len(harmful_acts),
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 4: Baselines                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def random_direction_baseline(benign_acts: np.ndarray, harmful_acts: np.ndarray,
                              n_centers: int, n_pca_dims: int,
                              fpr_target: float = 0.02) -> dict:
    """
    Baseline: use random vectors as "cluster centers" instead of learned ones.
    Shows that the learned subspace captures real structure.
    """
    D = benign_acts.shape[1]

    # Random PCA (just take first n_pca_dims)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_pca_dims, D))
    all_acts = np.vstack([benign_acts, harmful_acts])
    pca.fit(all_acts)

    # Random centers in PCA space
    pca_data = pca.transform(all_acts)
    random_centers = np.random.randn(n_centers, pca.n_components_)
    # Scale to match real data magnitude
    random_centers *= np.std(pca_data, axis=0, keepdims=True)

    detector = SubspaceDetector(
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        cluster_centers=random_centers,
    )

    # Tune threshold
    benign_projected = detector.score(benign_acts)
    theta = float(np.percentile(benign_projected, fpr_target * 100))
    detector.threshold = theta

    return evaluate_detector(detector, benign_acts, harmful_acts)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 5: Main Pipeline                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_activations_for_eval(layer_idx: int) -> dict:
    """Load benign and harmful activations for detector evaluation."""
    base = Path("artifacts") / f"layer_{layer_idx}"

    # Try split files first
    if (base / "test_activations.pt").exists():
        data = torch.load(base / "test_activations.pt", weights_only=False)
        all_acts = data["activations"].numpy()
        all_labels = data["labels"].numpy()
    elif (base / "activations.pt").exists():
        data = torch.load(base / "activations.pt", weights_only=False)
        all_acts = data["activations"].numpy()
        all_labels = data["labels"].numpy()
    else:
        raise FileNotFoundError(f"No activations at {base}")

    benign = all_acts[all_labels == 0.0]
    harmful = all_acts[all_labels == 1.0]

    # Split benign into calibration (first 50%) and test (last 50%)
    n_cal = len(benign) // 2
    calibration_benign = benign[:n_cal]
    test_benign = benign[n_cal:]

    return {
        "calibration_benign": calibration_benign,
        "test_benign": test_benign,
        "harmful": harmful,
    }


def run_detector(
    layer_idx: int,
    fpr_target: float = 0.02,
    evaluate_baselines: bool = False,
    save_plots: bool = True,
) -> dict:
    """
    Build and evaluate the jailbreak detector.

    Steps:
        1. Load PCA model + cluster centers from Module 6
        2. Build SubspaceDetector
        3. Tune threshold on calibration benign data
        4. Evaluate on test benign + harmful
        5. Optionally evaluate baselines

    Returns dict for Paper Table 4.
    """
    print(f"\n{'='*60}")
    print(f"  Module 7: Detector — Layer {layer_idx}")
    print(f"{'='*60}")

    # Load detector
    detector = SubspaceDetector.from_artifacts(layer_idx)
    print(f"  Loaded detector: {detector.centers.shape[0]} cluster centers, "
          f"{detector.pca_components.shape[0]} PCA dims")

    # Load evaluation data
    eval_data = load_activations_for_eval(layer_idx)
    print(f"  Calibration benign: {len(eval_data['calibration_benign'])}")
    print(f"  Test benign:        {len(eval_data['test_benign'])}")
    print(f"  Harmful:            {len(eval_data['harmful'])}")

    # Tune threshold
    theta = tune_threshold(detector, eval_data["calibration_benign"], fpr_target)
    detector.threshold = theta

    # Evaluate
    print(f"\n  Evaluating detector...")
    our_results = evaluate_detector(
        detector, eval_data["test_benign"], eval_data["harmful"]
    )
    our_results["method"] = "Ours (subspace)"

    print(f"\n  Our detector:")
    print(f"    TPR:  {our_results['tpr']:.4f} ({our_results['tpr']*100:.1f}%)")
    print(f"    FPR:  {our_results['fpr']:.4f} ({our_results['fpr']*100:.1f}%)")
    print(f"    AUC:  {our_results['auc']:.4f}")
    print(f"    F1:   {our_results['f1']:.4f}")

    all_results = {"ours": our_results}

    # ── Baselines ─────────────────────────────────────────────────────────
    if evaluate_baselines:
        print(f"\n  --- Baselines ---")

        # Random direction baseline
        random_results = random_direction_baseline(
            eval_data["test_benign"], eval_data["harmful"],
            n_centers=detector.centers.shape[0],
            n_pca_dims=detector.pca_components.shape[0],
            fpr_target=fpr_target,
        )
        random_results["method"] = "Random directions"
        all_results["random"] = random_results

        print(f"\n  Random direction detector:")
        print(f"    TPR:  {random_results['tpr']:.4f}")
        print(f"    FPR:  {random_results['fpr']:.4f}")
        print(f"    AUC:  {random_results['auc']:.4f}")
        print(f"    F1:   {random_results['f1']:.4f}")

    # ── Plot: Score distributions ─────────────────────────────────────────
    if save_plots and HAS_PLOTTING:
        fig_dir = Path("artifacts") / f"layer_{layer_idx}" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        benign_scores = detector.score(eval_data["test_benign"])
        harmful_scores = detector.score(eval_data["harmful"])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(benign_scores, bins=50, alpha=0.6, color="steelblue",
                label=f"Benign (n={len(benign_scores)})", density=True)
        ax.hist(harmful_scores, bins=50, alpha=0.6, color="crimson",
                label=f"Harmful (n={len(harmful_scores)})", density=True)
        ax.axvline(theta, color="black", linestyle="--", linewidth=2,
                   label=f"Threshold theta={theta:.3f}")
        ax.set_xlabel("Min cosine distance to cluster center")
        ax.set_ylabel("Density")
        ax.set_title(f"Detector Score Distribution — Layer {layer_idx}\n"
                     f"TPR={our_results['tpr']:.3f}, FPR={our_results['fpr']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = fig_dir / "detector_scores.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: {out_path}")

    # ── Save results ──────────────────────────────────────────────────────
    out_dir = Path("artifacts") / f"layer_{layer_idx}"

    # Save deployable detector config
    torch.save({
        "pca_components": torch.tensor(detector.pca_components, dtype=torch.float32),
        "pca_mean": torch.tensor(detector.pca_mean, dtype=torch.float32),
        "cluster_centers": torch.tensor(detector.centers, dtype=torch.float32),
        "threshold": theta,
        "layer": layer_idx,
    }, out_dir / "detector_config.pt")

    # Save metrics JSON
    metrics = {
        "layer": layer_idx,
        "fpr_target": fpr_target,
        "threshold": theta,
        "results": {k: {kk: vv for kk, vv in v.items()}
                    for k, v in all_results.items()},
    }
    with open(out_dir / "detector_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved detector_config.pt + detector_results.json")

    # ── Paper Table 4 row ─────────────────────────────────────────────────
    print(f"\n  --- Paper Table 4 Row ---")
    print(f"  {'Method':<25} {'TPR':>6} {'FPR':>6} {'AUC':>6} {'F1':>6}")
    print(f"  {'-'*49}")
    for name, r in all_results.items():
        print(f"  {r['method']:<25} {r['tpr']:>6.3f} {r['fpr']:>6.3f} "
              f"{r['auc']:>6.3f} {r['f1']:>6.3f}")

    return metrics


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 6: Main + CLI                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 7: Detector Construction (Paper Section 3.8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/module7_detector.py --layer 20 --fpr-target 0.02
  python src/module7_detector.py --layer 20 --evaluate-baselines
  python src/module7_detector.py --layer all
        """
    )

    parser.add_argument("--layer", type=str, default="20")
    parser.add_argument("--fpr-target", type=float, default=0.02,
                        help="Target false positive rate for threshold tuning")
    parser.add_argument("--evaluate-baselines", action="store_true",
                        help="Also evaluate baseline detectors")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    if args.layer.lower() == "all":
        all_results = {}
        for layer_idx in DEFAULT_LAYERS:
            print(f"\n{'#'*60}")
            print(f"  LAYER {layer_idx}")
            print(f"{'#'*60}")
            try:
                results = run_detector(
                    layer_idx, args.fpr_target,
                    args.evaluate_baselines, save_plots=not args.no_plots
                )
                all_results[layer_idx] = results
            except (FileNotFoundError, ValueError) as e:
                print(f"  [SKIP] {e}")

        # Print combined Table 4
        if all_results:
            print(f"\n{'='*60}")
            print(f"  Paper Table 4: Detector Performance Per Layer")
            print(f"{'='*60}")
            print(f"  {'Layer':<8} {'TPR':>6} {'FPR':>6} {'AUC':>6} {'F1':>6}")
            print(f"  {'-'*32}")
            for layer_idx, r in sorted(all_results.items()):
                ours = r["results"]["ours"]
                print(f"  {layer_idx:<8} {ours['tpr']:>6.3f} {ours['fpr']:>6.3f} "
                      f"{ours['auc']:>6.3f} {ours['f1']:>6.3f}")
    else:
        run_detector(
            int(args.layer), args.fpr_target,
            args.evaluate_baselines, save_plots=not args.no_plots
        )

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

    The detector works in the perturbation (delta_f) space:
    1. Compute the "perturbation" of the input relative to the benign centroid:
       delta = f_L(x) - benign_centroid
    2. Project delta into PCA space (fit on successful delta_f from Module 6)
    3. Score: d(x) = min_k cosdist(PCA(delta), mu_k)
    4. Flag as jailbreak if d(x) < theta

    This aligns train (Module 6 clusters delta_f) with inference (we compute
    an analogous delta from the input activation).
    """

    def __init__(self, pca_components: np.ndarray, pca_mean: np.ndarray,
                 cluster_centers: np.ndarray, threshold: float = 0.5,
                 benign_centroid: np.ndarray = None):
        self.pca_components = pca_components  # [n_pca, D]
        self.pca_mean = pca_mean              # [D]
        self.centers = cluster_centers         # [K*, n_pca]
        self.threshold = threshold
        self.benign_centroid = benign_centroid  # [D] — mean of benign activations

    def pca_transform(self, data: np.ndarray) -> np.ndarray:
        """Project data into PCA space."""
        centered = data - self.pca_mean
        return centered @ self.pca_components.T

    def _to_delta_space(self, activations: np.ndarray) -> np.ndarray:
        """Convert raw activations to delta space (relative to benign centroid)."""
        if self.benign_centroid is not None:
            return activations - self.benign_centroid
        return activations

    def score(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute detection scores (lower = more jailbreak-like).
        Returns array of min cosine distances to cluster centers.
        """
        deltas = self._to_delta_space(activations)
        projected = self.pca_transform(deltas)
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

        # Load benign centroid if available
        benign_centroid = None
        centroid_path = base / "benign_centroid.pt"
        if centroid_path.exists():
            benign_centroid = torch.load(centroid_path, weights_only=False).numpy()

        return cls(
            pca_components=pca_data["pca_components"].numpy(),
            pca_mean=pca_data["pca_mean"].numpy(),
            cluster_centers=cluster_data["centers"].numpy(),
            threshold=threshold,
            benign_centroid=benign_centroid,
        )


class MultiLayerDetector:
    """
    Combines detectors from multiple layers with a learned combiner.

    Why not just take the min score across layers?
        4 independent detectors each at FPR=2% gives:
            FPR_combined = 1 - (1-0.02)^4 ≈ 7.8%
        A learned combiner (logistic regression on 4 distances) can
        achieve the target FPR=2% for the combined detector.
    """

    def __init__(self, layer_detectors: dict):
        """
        Args:
            layer_detectors: dict mapping layer_idx -> SubspaceDetector
        """
        self.detectors = layer_detectors
        self.layers = sorted(layer_detectors.keys())
        self.combiner = None
        self.threshold = 0.5

    def _get_features(self, activations_per_layer: dict) -> np.ndarray:
        """Get score features from each layer detector."""
        features = []
        for layer in self.layers:
            scores = self.detectors[layer].score(activations_per_layer[layer])
            features.append(scores.reshape(-1, 1))
        return np.hstack(features)  # [N, n_layers]

    def fit_combiner(self, calibration_data: dict, fpr_target: float = 0.02):
        """
        Train a logistic regression combiner on calibration data.

        Args:
            calibration_data: dict with 'benign' and 'harmful' keys,
                each mapping to dict of layer_idx -> activations
            fpr_target: target combined FPR
        """
        from sklearn.linear_model import LogisticRegression

        benign_features = self._get_features(calibration_data["benign"])
        harmful_features = self._get_features(calibration_data["harmful"])

        X = np.vstack([benign_features, harmful_features])
        y = np.concatenate([
            np.zeros(len(benign_features)),
            np.ones(len(harmful_features)),
        ])

        self.combiner = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )
        self.combiner.fit(X, y)

        # Tune threshold on benign calibration data for target FPR
        benign_probs = self.combiner.predict_proba(benign_features)[:, 1]
        self.threshold = float(np.percentile(benign_probs, (1 - fpr_target) * 100))

        actual_fpr = (benign_probs >= self.threshold).mean()
        print(f"  Multi-layer combiner:")
        print(f"    Layers: {self.layers}")
        print(f"    Threshold: {self.threshold:.4f}")
        print(f"    Calibration FPR: {actual_fpr:.4f}")
        print(f"    Layer weights: {dict(zip(self.layers, self.combiner.coef_[0]))}")

    def predict_proba(self, activations_per_layer: dict) -> np.ndarray:
        """Returns jailbreak probability for each sample."""
        features = self._get_features(activations_per_layer)
        return self.combiner.predict_proba(features)[:, 1]

    def predict(self, activations_per_layer: dict) -> np.ndarray:
        """Returns True for detected jailbreaks."""
        probs = self.predict_proba(activations_per_layer)
        return probs >= self.threshold


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
    """
    Load activations for detector evaluation.

    Uses val split for calibration (threshold tuning) and test split for
    evaluation to prevent data leakage.
    """
    base = Path("artifacts") / f"layer_{layer_idx}"

    # Load val split for calibration
    val_path = base / "val_activations.pt"
    test_path = base / "test_activations.pt"

    if val_path.exists() and test_path.exists():
        print("  Using val split for calibration, test split for evaluation")
        val_data = torch.load(val_path, weights_only=False)
        val_acts = val_data["activations"].numpy()
        val_labels = val_data["labels"].numpy()
        calibration_benign = val_acts[val_labels == 0.0]

        test_data = torch.load(test_path, weights_only=False)
        test_acts = test_data["activations"].numpy()
        test_labels = test_data["labels"].numpy()
        test_benign = test_acts[test_labels == 0.0]
        harmful = test_acts[test_labels == 1.0]
    elif test_path.exists():
        # Fallback: only test split available, split benign for calibration
        print("  [WARN] No val split found, splitting test benign for calibration")
        data = torch.load(test_path, weights_only=False)
        all_acts = data["activations"].numpy()
        all_labels = data["labels"].numpy()
        benign = all_acts[all_labels == 0.0]
        harmful = all_acts[all_labels == 1.0]
        n_cal = len(benign) // 2
        calibration_benign = benign[:n_cal]
        test_benign = benign[n_cal:]
    else:
        raise FileNotFoundError(
            f"No split activation files at {base}. "
            f"Run Extraction.py first to create val/test splits."
        )

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

    # Compute benign centroid from calibration data ONLY (no test data — avoids leakage)
    all_benign = eval_data["calibration_benign"]
    benign_centroid = all_benign.mean(axis=0)
    detector.benign_centroid = benign_centroid

    out_dir = Path("artifacts") / f"layer_{layer_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(benign_centroid, dtype=torch.float32),
               out_dir / "benign_centroid.pt")
    print(f"  Computed benign centroid from {len(all_benign)} samples")

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
        "benign_centroid": torch.tensor(benign_centroid, dtype=torch.float32),
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

            # Multi-layer ensemble with learned combiner
            if len(all_results) >= 2:
                print(f"\n  --- Multi-Layer Ensemble Detector ---")
                try:
                    layer_detectors = {}
                    calibration_data = {"benign": {}, "harmful": {}}
                    test_data = {"benign": {}, "harmful": {}}

                    for layer_idx in all_results:
                        det = SubspaceDetector.from_artifacts(layer_idx)
                        eval_data = load_activations_for_eval(layer_idx)
                        # Fix centroid from calibration only
                        det.benign_centroid = eval_data["calibration_benign"].mean(axis=0)
                        layer_detectors[layer_idx] = det
                        calibration_data["benign"][layer_idx] = eval_data["calibration_benign"]
                        calibration_data["harmful"][layer_idx] = eval_data["harmful"][:len(eval_data["calibration_benign"])]
                        test_data["benign"][layer_idx] = eval_data["test_benign"]
                        test_data["harmful"][layer_idx] = eval_data["harmful"]

                    multi_det = MultiLayerDetector(layer_detectors)
                    multi_det.fit_combiner(calibration_data, fpr_target=args.fpr_target)

                    # Evaluate on test data
                    test_benign_preds = multi_det.predict(test_data["benign"])
                    test_harmful_preds = multi_det.predict(test_data["harmful"])
                    fpr = test_benign_preds.mean()
                    tpr = test_harmful_preds.mean()
                    print(f"  Multi-layer TPR: {tpr:.3f}, FPR: {fpr:.3f}")
                except Exception as e:
                    print(f"  [SKIP] Multi-layer ensemble failed: {e}")
    else:
        run_detector(
            int(args.layer), args.fpr_target,
            args.evaluate_baselines, save_plots=not args.no_plots
        )

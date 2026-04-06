#!/usr/bin/env python3
"""
pipeline.py

Full pipeline orchestrator for the paper:
"Mapping the Jailbreak Activation Space: A Coverage-Driven Framework
 for Prompt-Agnostic Jailbreak Detection"

Runs modules in sequence, skipping completed steps when artifacts exist.

Pipeline:
    PCA Analysis → Module 4 (Corruption) → Module 5 (Judge) →
    Module 6 (Clustering) → Module 7 (Detector)

Usage:
    # Full pipeline for one layer
    python src/pipeline.py --layer 20 --architecture cvae --epsilon 0.1 --K 500

    # Full pipeline for all layers
    python src/pipeline.py --layer all

    # Run specific modules
    python src/pipeline.py --layer 20 --modules pca
    python src/pipeline.py --layer 20 --modules corruption,judge
    python src/pipeline.py --layer 20 --modules clustering,detector
    python src/pipeline.py --layer 20 --modules all

    # With all options
    python src/pipeline.py --layer 20 --architecture cvae --epsilon 0.1 --K 500 \\
        --judge-method heuristic --fpr-target 0.02 --evaluate-baselines
"""

import argparse
import json
import time
from pathlib import Path

from module3_perturbation_generator import DEFAULT_LAYERS

# Module imports (lazy — only imported when needed)
MODULE_NAMES = ["pca", "generator", "corruption", "judge", "clustering", "detector"]


def check_artifacts(layer_idx: int, module: str) -> bool:
    """Check if a module's output artifacts already exist."""
    base = Path("artifacts") / f"layer_{layer_idx}"
    arch_dir = base / "cvae"  # default architecture subdir
    checks = {
        "pca": base / "pca_results.json",
        "generator": arch_dir / "generator.pt",
        "corruption": base / "corruption_results.pt",
        "judge": base / "judged_results.pt",
        "clustering": base / "cluster_centers.pt",
        "detector": base / "detector_config.pt",
    }
    path = checks.get(module)
    return path is not None and path.exists()


def run_pipeline(
    layer_idx: int,
    modules: list,
    architecture: str = "cvae",
    epsilon: float = 0.1,
    K: int = 500,
    max_passages: int = None,
    judge_method: str = "heuristic",
    judge_threshold: float = 7.0,
    api_key: str = None,
    n_pca_dims: int = 50,
    fpr_target: float = 0.02,
    evaluate_baselines: bool = False,
    force: bool = False,
):
    """
    Run pipeline modules in sequence for a single layer.

    Args:
        layer_idx: target layer
        modules: list of module names to run
        force: if True, re-run even if artifacts exist
    """
    print(f"\n{'#'*60}")
    print(f"  PIPELINE — Layer {layer_idx}")
    print(f"  Modules: {', '.join(modules)}")
    print(f"{'#'*60}")

    start_time = time.time()

    # ── PCA Analysis ──────────────────────────────────────────────────────
    if "pca" in modules:
        if not force and check_artifacts(layer_idx, "pca"):
            print(f"\n  [SKIP] PCA analysis — artifacts exist (use --force to re-run)")
        else:
            from pca_analysis import (
                analysis_raw_activation_space,
                analysis_perturbation_effect,
                save_results,
            )
            print(f"\n  Running PCA analysis...")
            try:
                results = analysis_raw_activation_space(layer_idx)
                save_results(results, layer_idx)
            except FileNotFoundError as e:
                print(f"  [SKIP] Raw analysis: {e}")

            try:
                results = analysis_perturbation_effect(
                    layer_idx, architecture, epsilon
                )
                save_results(results, layer_idx, "perturbation_results.json")
            except FileNotFoundError as e:
                print(f"  [SKIP] Perturbation analysis: {e}")

    # ── Module 3: Generator Training ─────────────────────────────────────
    if "generator" in modules:
        if not force and check_artifacts(layer_idx, "generator"):
            print(f"\n  [SKIP] Generator — artifacts exist")
        else:
            from module3_perturbation_generator import main as gen_main
            print(f"\n  Running generator training (all phases)...")
            import types
            gen_args = types.SimpleNamespace(
                layer_idx=layer_idx,
                architecture=architecture,
                phase="all",
                epsilon=epsilon,
                z_dim=32 if architecture == "cvae" else 64,
                n_rl_steps=5000,
                n_fw_iterations=3,
            )
            try:
                gen_main(gen_args)
            except (FileNotFoundError, ValueError) as e:
                print(f"  [ERROR] Generator training failed: {e}")

    # ── Module 4: Corruption ──────────────────────────────────────────────
    if "corruption" in modules:
        if not force and check_artifacts(layer_idx, "corruption"):
            print(f"\n  [SKIP] Corruption — artifacts exist")
        else:
            from module4_corruption import run_corruption_loop
            print(f"\n  Running corruption loop...")
            run_corruption_loop(
                layer_idx=layer_idx,
                architecture=architecture,
                K=K,
                epsilon=epsilon,
                max_passages=max_passages,
            )

    # ── Module 5: Judge ───────────────────────────────────────────────────
    if "judge" in modules:
        if not force and check_artifacts(layer_idx, "judge"):
            print(f"\n  [SKIP] Judge — artifacts exist")
        else:
            from module5_judge import run_judge
            print(f"\n  Running judge...")
            run_judge(
                layer_idx=layer_idx,
                method=judge_method,
                threshold=judge_threshold,
                api_key=api_key,
            )

    # ── Module 6: Clustering ─────────────────────────────────────────────
    if "clustering" in modules:
        if not force and check_artifacts(layer_idx, "clustering"):
            print(f"\n  [SKIP] Clustering — artifacts exist")
        else:
            from module6_clustering import run_clustering
            print(f"\n  Running clustering...")
            try:
                run_clustering(
                    layer_idx=layer_idx,
                    n_pca_dims=n_pca_dims,
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"  [ERROR] Clustering failed: {e}")

    # ── Module 7: Detector ────────────────────────────────────────────────
    if "detector" in modules:
        if not force and check_artifacts(layer_idx, "detector"):
            print(f"\n  [SKIP] Detector — artifacts exist")
        else:
            from module7_detector import run_detector
            print(f"\n  Running detector...")
            try:
                run_detector(
                    layer_idx=layer_idx,
                    fpr_target=fpr_target,
                    evaluate_baselines=evaluate_baselines,
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"  [ERROR] Detector failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n  Layer {layer_idx} complete ({elapsed:.1f}s)")


def print_summary(layers: list):
    """Print summary of all results across layers."""
    print(f"\n{'='*70}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*70}")

    # Collect results
    for layer_idx in layers:
        base = Path("artifacts") / f"layer_{layer_idx}"
        print(f"\n  Layer {layer_idx}:")

        # PCA results
        pca_path = base / "pca_results.json"
        if pca_path.exists():
            with open(pca_path) as f:
                pca = json.load(f)
            sil = pca.get("silhouette_score_pca50", "N/A")
            print(f"    PCA silhouette:    {sil}")

        # Judge metrics
        judge_path = base / "judge_metrics.json"
        if judge_path.exists():
            with open(judge_path) as f:
                judge = json.load(f)
            n_success = judge.get("n_successful_delta_f", "N/A")
            for method, data in judge.items():
                if isinstance(data, dict) and "asr" in data:
                    print(f"    Judge ({method}): ASR={data['asr']:.4f}")
            print(f"    Successful delta_f: {n_success}")

        # Cluster results
        cluster_path = base / "cluster_results.json"
        if cluster_path.exists():
            with open(cluster_path) as f:
                cluster = json.load(f)
            km = cluster.get("kmeans", {})
            print(f"    K*={km.get('k_star', 'N/A')}, "
                  f"silhouette={km.get('silhouette_score', 'N/A')}")

        # Detector results
        det_path = base / "detector_results.json"
        if det_path.exists():
            with open(det_path) as f:
                det = json.load(f)
            ours = det.get("results", {}).get("ours", {})
            if ours:
                print(f"    Detector: TPR={ours.get('tpr', 'N/A')}, "
                      f"FPR={ours.get('fpr', 'N/A')}, "
                      f"F1={ours.get('f1', 'N/A')}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main + CLI                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py --layer 20 --modules all
  python src/pipeline.py --layer all --modules all
  python src/pipeline.py --layer 20 --modules pca,corruption,judge
  python src/pipeline.py --layer 20 --modules all --architecture cvae --K 500 --epsilon 0.1
        """
    )

    # Core
    parser.add_argument("--layer", type=str, default="20",
                        help="Layer index or 'all'")
    parser.add_argument("--modules", type=str, default="all",
                        help="Comma-separated: pca,corruption,judge,clustering,detector or 'all'")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if artifacts exist")

    # Module 4: Corruption
    parser.add_argument("--architecture", type=str, default="cvae",
                        choices=["mlp", "cvae"])
    parser.add_argument("--K", type=int, default=500,
                        help="Perturbation samples per passage")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-passages", type=int, default=None)

    # Module 5: Judge
    parser.add_argument("--judge-method", type=str, default="heuristic",
                        choices=["gpt4", "heuristic", "both"])
    parser.add_argument("--judge-threshold", type=float, default=7.0)
    parser.add_argument("--api-key", type=str, default=None)

    # Module 6: Clustering
    parser.add_argument("--n-pca-dims", type=int, default=50)

    # Module 7: Detector
    parser.add_argument("--fpr-target", type=float, default=0.02)
    parser.add_argument("--evaluate-baselines", action="store_true")

    args = parser.parse_args()

    # Parse modules
    if args.modules.lower() == "all":
        modules = MODULE_NAMES
    else:
        modules = [m.strip() for m in args.modules.split(",")]
        for m in modules:
            if m not in MODULE_NAMES:
                parser.error(f"Unknown module: {m}. Choose from: {MODULE_NAMES}")

    # Parse layers
    if args.layer.lower() == "all":
        layers = list(DEFAULT_LAYERS)
    else:
        layers = [int(args.layer)]

    # Run
    total_start = time.time()

    for layer_idx in layers:
        run_pipeline(
            layer_idx=layer_idx,
            modules=modules,
            architecture=args.architecture,
            epsilon=args.epsilon,
            K=args.K,
            max_passages=args.max_passages,
            judge_method=args.judge_method,
            judge_threshold=args.judge_threshold,
            api_key=args.api_key,
            n_pca_dims=args.n_pca_dims,
            fpr_target=args.fpr_target,
            evaluate_baselines=args.evaluate_baselines,
            force=args.force,
        )

    # Cross-layer analysis if multiple layers
    if len(layers) > 1 and "pca" in modules:
        from pca_analysis import analysis_cross_layer
        print(f"\n  Running cross-layer analysis...")
        analysis_cross_layer(layers)

    # Summary
    print_summary(layers)

    total_elapsed = time.time() - total_start
    print(f"\n  Total pipeline time: {total_elapsed:.1f}s")

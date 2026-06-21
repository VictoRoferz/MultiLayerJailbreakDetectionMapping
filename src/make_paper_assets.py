#!/usr/bin/env python3
"""
make_paper_assets.py — Aggregate per-layer pipeline outputs into the tables and
figures for the paper "Mapping the Jailbreak Activation Space".

Reads the JSON/PNG artifacts the pipeline already writes per layer and emits a
single bundle under artifacts/paper/:

    Table 2  (per-layer cluster analysis)   <- artifacts/layer_{L}/cluster_results.json
    Table 4  (detector performance)         <- artifacts/layer_{L}/detector_results.json
    Judge ASR table                         <- artifacts/layer_{L}/judge_metrics.json
    Figure 4 (silhouette by layer)          <- aggregated from cluster_results.json
    Figures 2/3 collected (UMAP, elbow)     <- artifacts/layer_{L}/figures/*.png

Tables are written as both CSV and LaTeX. Missing per-layer files are skipped
with a warning (so partial sweeps still produce a partial bundle).

NOTE: Table 1 (verifier precision/recall/F1/FPR vs human labels) needs human
annotations (see src/human_label.py); this script reports the judge ASR table
from judge_metrics.json and flags Table 1 as requiring human labels if absent.

Table 3/Figure 5 (saturation) and Table 5 (BPJ/LOAO) are produced by the
dedicated scripts src/saturation_analysis.py and src/bpj_validation.py; if their
outputs (saturation_results.json / bpj_results.json) exist they are linked into
the bundle here.

Usage (from repo root, after running the pipeline for the chosen layers):
    python src/make_paper_assets.py --layers 20
    python src/make_paper_assets.py --layers 5 10 15 20 25
"""

import argparse
import csv
import json
import shutil
from pathlib import Path


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] could not read {path}: {e}")
        return None


def _write_csv(path: Path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}")


def _write_latex(path: Path, header, rows, caption, label):
    col_fmt = "l" + "r" * (len(header) - 1)
    lines = [
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}", "\\toprule",
        " & ".join(str(h) for h in header) + " \\\\", "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines))
    print(f"  wrote {path}")


def _fmt(x, nd=4):
    return "" if x is None else f"{x:.{nd}f}"


def build_table2(layers, art_root, out_dir):
    """Per-layer cluster analysis: K*, silhouette, intra/inter cosine."""
    header = ["Layer", "K*", "Silhouette", "Intra-cos", "Inter-cos", "n_success"]
    rows, sil_by_layer = [], {}
    for L in layers:
        r = _load_json(art_root / f"layer_{L}" / "cluster_results.json")
        if r is None:
            print(f"  [skip] Table 2 layer {L}: no cluster_results.json")
            continue
        km = r.get("kmeans", {})
        cm = r.get("cluster_metrics", {})
        sil = km.get("silhouette_score")
        sil_by_layer[L] = sil
        rows.append([
            L, km.get("k_star", ""), _fmt(sil),
            _fmt(cm.get("mean_intra_cosine")), _fmt(cm.get("mean_inter_cosine")),
            r.get("n_successful_perturbations", ""),
        ])
    if rows:
        _write_csv(out_dir / "table2_clusters.csv", header, rows)
        _write_latex(out_dir / "table2_clusters.tex", header, rows,
                     "Cluster analysis of successful $\\delta f$ per layer.",
                     "tab:clusters")
    return sil_by_layer


def build_table4(layers, art_root, out_dir):
    """Detector performance per layer (Ours + random baseline)."""
    header = ["Layer", "TPR", "FPR", "AUC(jb-vs-ben+ref)",
              "AUC(jb-vs-refused)", "F1", "Rand AUC"]
    rows = []
    for L in layers:
        r = _load_json(art_root / f"layer_{L}" / "detector_results.json")
        if r is None:
            print(f"  [skip] Table 4 layer {L}: no detector_results.json")
            continue
        res = r.get("results", {})
        ours = res.get("ours", {})
        rand = res.get("random", {})
        rows.append([
            L, _fmt(ours.get("tpr")), _fmt(ours.get("fpr")),
            _fmt(ours.get("auc")), _fmt(ours.get("auc_hard_jb_vs_refused")),
            _fmt(ours.get("f1")), _fmt(rand.get("auc")),
        ])
    if rows:
        _write_csv(out_dir / "table4_detector.csv", header, rows)
        _write_latex(out_dir / "table4_detector.tex", header, rows,
                     "Jailbreak detector performance. AUC(jb-vs-refused) is the "
                     "honest metric (refused-harmful as hard negatives).",
                     "tab:detector")


def build_judge_table(layers, art_root, out_dir):
    """Judge ASR per layer per method (from judge_metrics.json)."""
    header = ["Layer", "Method", "n_successful", "ASR"]
    rows = []
    for L in layers:
        r = _load_json(art_root / f"layer_{L}" / "judge_metrics.json")
        if r is None:
            print(f"  [skip] Judge table layer {L}: no judge_metrics.json")
            continue
        for method, data in r.items():
            if isinstance(data, dict) and "asr" in data:
                rows.append([L, method, data.get("n_successful", ""),
                             _fmt(data.get("asr"))])
    if rows:
        _write_csv(out_dir / "judge_asr.csv", header, rows)
    else:
        print("  [note] No judge ASR data. Table 1 (verifier precision/recall vs "
              "human labels) requires human annotations (src/human_label.py).")


def build_figure4(sil_by_layer, out_dir):
    """Figure 4: silhouette score by layer (bar chart)."""
    pts = [(L, s) for L, s in sorted(sil_by_layer.items()) if s is not None]
    if not pts:
        print("  [skip] Figure 4: no silhouette data")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [skip] Figure 4: matplotlib unavailable ({e})")
        return
    xs = [str(L) for L, _ in pts]
    ys = [s for _, s in pts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(xs, ys, color="#4C72B0")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Silhouette score (K*)")
    ax.set_title("Cluster silhouette by layer")
    fig.tight_layout()
    fig.savefig(out_dir / "figure4_silhouette_by_layer.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_dir / 'figure4_silhouette_by_layer.png'}")


def collect_figures(layers, art_root, fig_dir):
    """Copy the per-layer figures module6 already writes (Figures 2/3)."""
    wanted = ["elbow_plot.png", "umap_clusters.png", "pca_clusters.png",
              "dbscan_clusters.png", "detector_scores.png"]
    n = 0
    for L in layers:
        src_dir = art_root / f"layer_{L}" / "figures"
        if not src_dir.exists():
            continue
        for name in wanted:
            src = src_dir / name
            if src.exists():
                shutil.copy(src, fig_dir / f"layer_{L}_{name}")
                n += 1
    print(f"  collected {n} per-layer figure(s) into {fig_dir}")


def link_optional(art_root, layers, out_dir):
    """Link saturation / BPJ outputs if the dedicated scripts produced them."""
    sat = _load_json(art_root / "paper" / "saturation_results.json") or \
        _load_json(art_root / "saturation_results.json")
    if sat:
        with open(out_dir / "table3_saturation.json", "w") as f:
            json.dump(sat, f, indent=2)
        print("  linked saturation_results.json -> table3_saturation.json")
    else:
        print("  [note] No saturation_results.json (run src/saturation_analysis.py "
              "for Table 3 / Figure 5).")
    bpj = _load_json(art_root / "bpj_results.json")
    if bpj:
        with open(out_dir / "table5_bpj.json", "w") as f:
            json.dump(bpj, f, indent=2)
        print("  linked bpj_results.json -> table5_bpj.json")
    else:
        print("  [note] No bpj_results.json (run src/bpj_validation.py for Table 5).")


def main():
    ap = argparse.ArgumentParser(description="Aggregate paper tables & figures.")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--artifacts-root", default="artifacts")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: <artifacts-root>/paper).")
    args = ap.parse_args()

    art_root = Path(args.artifacts_root)
    out_dir = Path(args.out) if args.out else art_root / "paper"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[-] Building paper assets for layers {args.layers} -> {out_dir}")
    sil = build_table2(args.layers, art_root, out_dir)
    build_table4(args.layers, art_root, out_dir)
    build_judge_table(args.layers, art_root, out_dir)
    build_figure4(sil, fig_dir)
    collect_figures(args.layers, art_root, fig_dir)
    link_optional(art_root, args.layers, out_dir)
    print(f"\n[done] Paper bundle in {out_dir}")


if __name__ == "__main__":
    main()

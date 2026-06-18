"""Per-layer analysis of the v2 HF datasets (gemma + vicuna).

For each model and layer: 3-class silhouette in PCA-50 space, normalized
centroid distances, and a logistic-regression probe (train -> val/test).
Also identifies the isolated PCA cluster's category composition.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score

RNG = np.random.default_rng(42)

REPOS = {
    "gemma": ("hf_analysis/gemma-2b-jailbreak-behavior-dataset-v2", [5, 10, 15, 20, 25]),
    "vicuna": ("hf_analysis/vicuna-7b-jailbreak-behavior-dataset-v2", [4, 10, 15, 20, 25, 30]),
}


def load_split(base, split):
    files = sorted(glob.glob(f"{base}/data/{split}-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def three_class(df):
    # 0=benign, 1=harmful-but-refused, 2=jailbroken
    out = np.zeros(len(df), dtype=int)
    nonbenign = df["category"] != "benign"
    out[nonbenign & (df["label"] == 0)] = 1
    out[df["label"] == 1] = 2
    return out


def main():
    wanted = sys.argv[1:] or list(REPOS)
    results = {}
    try:
        with open("hf_analysis/layer_analysis_results.json") as f:
            results = json.load(f)
    except FileNotFoundError:
        pass
    for model, (base, layers) in ((m, REPOS[m]) for m in wanted):
        print(f"\n{'='*70}\nMODEL: {model}  ({base})", flush=True)
        tr, va, te = (load_split(base, s) for s in ("train", "validation", "test"))
        full = pd.concat([tr, va, te], ignore_index=True)

        print("\n-- dataset composition (all splits) --")
        comp = full.groupby("category")["label"].agg(["count", "mean"])
        comp.columns = ["n", "jailbreak_rate"]
        print(comp.to_string())
        print("splits:", {s: len(d) for s, d in zip(("train", "val", "test"), (tr, va, te))})
        print("response empty rate:", float((full["response"].str.len() < 5).mean()).__round__(4))

        results[model] = {}
        for L in layers:
            col = f"activation_layer_{L}"
            if col not in tr.columns:
                print(f"layer {L}: column {col} missing, skipping"); continue
            Xtr = np.stack(tr[col].to_numpy()).astype(np.float32)
            Xva = np.stack(va[col].to_numpy()).astype(np.float32)
            Xte = np.stack(te[col].to_numpy()).astype(np.float32)
            Xall = np.vstack([Xtr, Xva, Xte])
            y3 = three_class(full)
            ybin_tr, ybin_va, ybin_te = tr["label"].to_numpy(), va["label"].to_numpy(), te["label"].to_numpy()

            # PCA-50 on a 4000-sample subset, transform all
            idx = RNG.choice(len(Xall), min(4000, len(Xall)), replace=False)
            pca = PCA(n_components=50, random_state=42).fit(Xall[idx])
            Z = pca.transform(Xall)

            sil_idx = RNG.choice(len(Z), min(3000, len(Z)), replace=False)
            sil3 = silhouette_score(Z[sil_idx], y3[sil_idx])
            sil_jb = silhouette_score(Z[sil_idx], (y3[sil_idx] == 2).astype(int))

            # normalized centroid distances (raw space)
            cents = {c: Xall[y3 == c].mean(0) for c in (0, 1, 2)}
            scale = float(np.linalg.norm(Xall, axis=1).mean())
            d_b_jb = float(np.linalg.norm(cents[0] - cents[2])) / scale
            d_ref_jb = float(np.linalg.norm(cents[1] - cents[2])) / scale

            # probes: jailbroken-vs-all and 3-class
            probe = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ybin_tr)
            va_auc = roc_auc_score(ybin_va, probe.decision_function(Xva))
            te_auc = roc_auc_score(ybin_te, probe.decision_function(Xte))
            te_acc = float((probe.predict(Xte) == ybin_te).mean())

            # isolated-cluster check: PC1 outliers' category mix
            pc1 = Z[:, 0]
            med, mad = np.median(pc1), np.median(np.abs(pc1 - np.median(pc1))) + 1e-9
            outlier = np.abs(pc1 - med) / (1.4826 * mad) > 5
            mix = full.loc[outlier, "category"].value_counts().to_dict() if outlier.sum() > 20 else {}

            r = dict(sil3=round(float(sil3), 3), sil_jb=round(float(sil_jb), 3),
                     d_benign_jb=round(d_b_jb, 3), d_refused_jb=round(d_ref_jb, 3),
                     probe_val_auroc=round(float(va_auc), 3), probe_test_auroc=round(float(te_auc), 3),
                     probe_test_acc=round(te_acc, 3), n_pc1_outliers=int(outlier.sum()),
                     outlier_categories=mix)
            results[model][L] = r
            print(f"\nlayer {L}: {json.dumps(r)}", flush=True)

    with open("hf_analysis/layer_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved hf_analysis/layer_analysis_results.json")


if __name__ == "__main__":
    sys.exit(main())

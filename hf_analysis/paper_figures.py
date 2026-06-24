"""Generate ALL figures + tables for the paper, CPU-only, both models.

Thesis: "jailbreak detection" = three problems (T1 intent / T2 artifact /
T3 success). This script produces every asset that backs that argument.

Data: local v2 parquet if present, else the HF Hub (set env vars):
    export HF_GEMMA_REPO="<user>/gemma-2b-jailbreak-behavior-dataset-v2"
    export HF_VICUNA_REPO="<user>/vicuna-7b-jailbreak-behavior-dataset-v2"

Run one asset at a time (recommended in Colab):
    python hf_analysis/paper_figures.py table1        # dataset composition
    python hf_analysis/paper_figures.py targets       # Fig 2 + Table 2 (T1/T2/T3)
    python hf_analysis/paper_figures.py families      # Fig 3 (MDS + cosine heatmap)
    python hf_analysis/paper_figures.py axes          # Fig 4 (harm vs refusal axis)
    python hf_analysis/paper_figures.py lofo          # Fig 5 + Table 3 (leave-one-family-out)
    python hf_analysis/paper_figures.py fewshot       # Fig 6 (data efficiency)
    python hf_analysis/paper_figures.py labeling      # Fig 7 (behavior vs origin labels)
    python hf_analysis/paper_figures.py all           # everything

Outputs land in paper_assets/ : *.png figures, *.csv + *.tex tables, metrics.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)
OUT = "paper_assets"

REPOS = {
    "gemma": ("hf_analysis/gemma-2b-jailbreak-behavior-dataset-v2", [5, 10, 15, 20, 25]),
    "vicuna": ("hf_analysis/vicuna-7b-jailbreak-behavior-dataset-v2", [4, 10, 15, 20, 25, 30]),
}
HF_IDS = {"gemma": os.environ.get("HF_GEMMA_REPO"),
          "vicuna": os.environ.get("HF_VICUNA_REPO")}
FAMILIES = ["jailbreak_gcg_universal", "jailbreak_artprompt",
            "jailbreak_gcg_individual", "harmful_direct"]
SHORT = {"jailbreak_gcg_universal": "GCG-univ", "jailbreak_artprompt": "ArtPrompt",
         "jailbreak_gcg_individual": "GCG-indiv", "harmful_direct": "Direct"}
JB_FAMILIES = ["jailbreak_gcg_universal", "jailbreak_artprompt", "jailbreak_gcg_individual"]
MAIN_LAYER = {"gemma": 15, "vicuna": 10}   # peak T3-success layer per Table 2 (single-layer figs)
MIN_N = 25
N_BOOT = 1000

# ───────────────────────── data loading ─────────────────────────

def load_splits(model):
    base, _ = REPOS[model]
    hf_id = HF_IDS.get(model)
    out = {}
    for split in ("train", "validation", "test"):
        files = sorted(glob.glob(f"{base}/data/{split}-*.parquet"))
        if files:
            out[split] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if out:
        return out["train"], out["validation"], out["test"]
    if hf_id:
        from datasets import load_dataset
        print(f"  [hub] loading {hf_id} ...", flush=True)
        dd = load_dataset(hf_id)
        return dd["train"].to_pandas(), dd["validation"].to_pandas(), dd["test"].to_pandas()
    raise FileNotFoundError(f"No parquet for {model} and no HF repo id (set HF_{model.upper()}_REPO).")


def X_of(df, L):
    return np.stack(df[f"activation_layer_{L}"].to_numpy()).astype(np.float32)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v

# ───────────────────────── metrics ─────────────────────────

def auroc_ci(y, s, n_boot=N_BOOT):
    y, s = np.asarray(y), np.asarray(s)
    base = roc_auc_score(y, s)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (base, base)
    return float(base), float(lo), float(hi)


def tpr_at_fpr(pos_scores, neg_scores, fpr=0.02):
    """Higher score = more jailbreak-like. Threshold set on negatives."""
    thr = np.quantile(neg_scores, 1 - fpr)
    return float(np.mean(np.asarray(pos_scores) >= thr))


def centroid_scores(centroid, X):
    """Cosine similarity to centroid (higher = closer = more jailbreak-like)."""
    c = unit(centroid)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return Xn @ c

# ───────────────────────── helpers ─────────────────────────

def masks(df):
    benign = (df["category"] == "benign").to_numpy()
    nonben = ~benign
    success = (df["label"] == 1).to_numpy()
    refused = nonben & (df["label"] == 0).to_numpy()       # harmful but refused (hard negatives)
    jailbroken = nonben & success
    return dict(benign=benign, nonben=nonben, refused=refused, jailbroken=jailbroken)


def save_table(name, df):
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    with open(f"{OUT}/{name}.tex", "w") as f:
        f.write(df.to_latex(index=False, escape=True, float_format="%.3f"))
    print(df.to_string(index=False))


METRICS = {}

def stash(key, val):
    METRICS[key] = val
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(METRICS, f, indent=2)

# ───────────────────────── Table 1: composition ─────────────────────────

def table1():
    rows = []
    for model in REPOS:
        tr, va, te = load_splits(model)
        full = pd.concat([tr, va, te], ignore_index=True)
        comp = full.groupby("category")["label"].agg(["count", "mean"]).reset_index()
        for _, r in comp.iterrows():
            rows.append(dict(model=model, category=r["category"], n=int(r["count"]),
                             jailbreak_rate=round(float(r["mean"]), 3)))
        rows.append(dict(model=model, category="TOTAL", n=len(full),
                         jailbreak_rate=round(float(full["label"].mean()), 3)))
    save_table("table1_composition", pd.DataFrame(rows))

# ───────────────────────── Fig 2 + Table 2: three targets ─────────────────────────

def targets():
    rows, bars = [], {}
    for model, (_, layers) in REPOS.items():
        tr, va, te = load_splits(model)
        bars[model] = {}
        for L in layers:
            Xtr, Xte = X_of(tr, L), X_of(te, L)
            mtr, mte = masks(tr), masks(te)
            # T1 intent: benign vs non-benign (topic / harm-belief)
            y1tr, y1te = mtr["nonben"].astype(int), mte["nonben"].astype(int)
            p1 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y1tr)
            t1, t1lo, t1hi = auroc_ci(y1te, p1.decision_function(Xte))
            # T3 success: jailbroken vs refused-harmful (hard negatives, non-benign only)
            tr_nb, te_nb = mtr["nonben"], mte["nonben"]
            y3tr = mtr["jailbroken"][tr_nb].astype(int)
            y3te = mte["jailbroken"][te_nb].astype(int)
            p3 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr[tr_nb], y3tr)
            t3, t3lo, t3hi = auroc_ci(y3te, p3.decision_function(Xte[te_nb]))
            rows.append(dict(model=model, layer=L,
                             T1_intent=round(t1, 3), T1_lo=round(t1lo, 3), T1_hi=round(t1hi, 3),
                             T3_success=round(t3, 3), T3_lo=round(t3lo, 3), T3_hi=round(t3hi, 3)))
            if L == MAIN_LAYER[model]:
                bars[model] = dict(T1=(t1, t1lo, t1hi), T3=(t3, t3lo, t3hi))
        # T2 artifact confound, at main layer: GCG cluster purity vs actual success
        full = pd.concat([tr, va, te], ignore_index=True)
        gcg = full[full["category"] == "jailbreak_gcg_universal"]
        stash(f"T2_{model}", dict(gcg_n=len(gcg),
                                  gcg_success_rate=round(float(gcg["label"].mean()), 3),
                                  note="GCG suffix is trivially detectable (artifact) but only this fraction are real jailbreaks"))
    save_table("table2_targets", pd.DataFrame(rows))
    # Fig 2: grouped bars T1 vs T3 with CIs, both models
    fig, ax = plt.subplots(figsize=(6, 4))
    models = list(bars)
    x = np.arange(len(models)); w = 0.35
    for i, t in enumerate(["T1", "T3"]):
        vals = [bars[m][t][0] for m in models]
        err = [[bars[m][t][0] - bars[m][t][1] for m in models],
               [bars[m][t][2] - bars[m][t][0] for m in models]]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=err, capsize=4,
               label={"T1": "T1 intent (benign vs harmful)", "T3": "T3 success (jailbroken vs refused)"}[t])
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("AUROC"); ax.set_title("Same data, different target → different number")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/fig2_targets.png", dpi=200)
    print("saved fig2_targets.png")

# ───────────────────────── Fig 3: MDS + cosine heatmap ─────────────────────────

def families():
    for model in REPOS:
        L = MAIN_LAYER[model]
        tr, va, te = load_splits(model)
        full = pd.concat([tr, va, te], ignore_index=True)
        X = X_of(full, L)
        m = masks(full)
        ref_mean = X[m["refused"]].mean(0)
        fams = [f for f in JB_FAMILIES
                if int(((full["category"] == f) & (full["label"] == 1)).sum()) >= MIN_N]
        # cosine heatmap of family directions, diagonal = split-half ceiling
        dirs = {}
        H = np.zeros((len(fams), len(fams)))
        for f in fams:
            Xf = X[((full["category"] == f) & (full["label"] == 1)).to_numpy()]
            dirs[f] = unit(Xf.mean(0) - ref_mean)
        for i, a in enumerate(fams):
            Xa = X[((full["category"] == a) & (full["label"] == 1)).to_numpy()]
            for j, b in enumerate(fams):
                if i == j:
                    perm = RNG.permutation(len(Xa))
                    h1 = unit(Xa[perm[:len(Xa)//2]].mean(0) - ref_mean)
                    h2 = unit(Xa[perm[len(Xa)//2:]].mean(0) - ref_mean)
                    H[i, j] = h1 @ h2
                else:
                    H[i, j] = dirs[a] @ dirs[b]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(H, cmap="RdBu_r", vmin=-0.5, vmax=1)
        ax.set_xticks(range(len(fams))); ax.set_yticks(range(len(fams)))
        labs = [SHORT[f] for f in fams]
        ax.set_xticklabels(labs, rotation=30, ha="right"); ax.set_yticklabels(labs)
        for i in range(len(fams)):
            for j in range(len(fams)):
                ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center",
                        color="white" if abs(H[i, j]) > 0.6 else "black", fontweight="bold")
        fig.colorbar(im, label="centroid cosine similarity")
        ax.set_title(f"{model} L{L}: attack families occupy distinct directions")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig3_heatmap_{model}.png", dpi=200)
        print(f"saved fig3_heatmap_{model}.png")
        # MDS scatter (subsample for speed)
        cats = ["benign", "harmful_direct"] + [f for f in JB_FAMILIES]
        idx, lab = [], []
        for c in cats:
            ii = np.where((full["category"] == c).to_numpy())[0]
            if c.startswith("jailbreak"):
                ii = np.where(((full["category"] == c) & (full["label"] == 1)).to_numpy())[0]
            if len(ii) == 0:
                continue
            take = RNG.choice(ii, min(150, len(ii)), replace=False)
            idx += list(take); lab += [c] * len(take)
        sub = X[idx]
        emb = MDS(n_components=2, random_state=42, normalized_stress="auto",
                  n_init=1, max_iter=200).fit_transform(StandardScaler().fit_transform(sub))
        fig, ax = plt.subplots(figsize=(6, 5))
        for c in cats:
            sel = [k for k, l in enumerate(lab) if l == c]
            if sel:
                ax.scatter(emb[sel, 0], emb[sel, 1], s=10, alpha=0.5,
                           label=f"{SHORT.get(c, c)} (n={len(sel)})")
        ax.set_title(f"{model} L{L}: MDS by family"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig3_mds_{model}.png", dpi=200)
        print(f"saved fig3_mds_{model}.png")

# ───────────────────────── Fig 4: harm vs refusal axis ─────────────────────────

def axes():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axs, REPOS):
        L = MAIN_LAYER[model]
        tr, va, te = load_splits(model)
        full = pd.concat([tr, va, te], ignore_index=True)
        X = X_of(full, L); m = masks(full)
        harm_axis = unit(X[m["nonben"]].mean(0) - X[m["benign"]].mean(0))      # harmfulness-belief
        refusal_axis = unit(X[m["jailbroken"]].mean(0) - X[m["refused"]].mean(0))  # refusal-suppression
        ref_mean = X[m["refused"]].mean(0)
        fams = [f for f in JB_FAMILIES
                if int(((full["category"] == f) & (full["label"] == 1)).sum()) >= MIN_N]
        rows = []
        for f in fams:
            v = X[((full["category"] == f) & (full["label"] == 1)).to_numpy()].mean(0) - ref_mean
            rows.append((SHORT[f], float(v @ harm_axis), float(v @ refusal_axis)))
        labs = [r[0] for r in rows]; xx = np.arange(len(labs)); w = 0.38
        ax.bar(xx - w/2, [r[1] for r in rows], w, label="harm-belief axis")
        ax.bar(xx + w/2, [r[2] for r in rows], w, label="refusal-suppression axis")
        ax.axhline(0, c="k", lw=0.8); ax.set_xticks(xx); ax.set_xticklabels(labs)
        ax.set_title(f"{model} L{L}"); ax.legend(fontsize=8)
    axs[0].set_ylabel("projection of family jailbreak shift")
    fig.suptitle("Which axis does each attack family use?")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_axes.png", dpi=200)
    print("saved fig4_axes.png")

# ───────────────────────── Fig 5 + Table 3: LOFO ─────────────────────────

def lofo():
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axs, REPOS):
        L = MAIN_LAYER[model]
        tr, va, te = load_splits(model)
        full = pd.concat([tr, va, te], ignore_index=True)
        X = X_of(full, L); m = masks(full)
        neg = X[m["refused"] | m["benign"]]                 # negatives = refused + benign
        fams = [f for f in JB_FAMILIES
                if int(((full["category"] == f) & (full["label"] == 1)).sum()) >= MIN_N]
        cross, within = [], []
        for held in fams:
            held_X = X[((full["category"] == held) & (full["label"] == 1)).to_numpy()]
            # cross-mechanism: centroid from OTHER jailbreak families
            other_mask = np.zeros(len(full), bool)
            for f in fams:
                if f != held:
                    other_mask |= ((full["category"] == f) & (full["label"] == 1)).to_numpy()
            cen_cross = X[other_mask].mean(0)
            tpr_c = tpr_at_fpr(centroid_scores(cen_cross, held_X), centroid_scores(cen_cross, neg))
            # within-mechanism (few-shot, k=25): centroid from held family itself
            perm = RNG.permutation(len(held_X)); k = min(25, len(held_X)//2)
            cen_in = held_X[perm[:k]].mean(0); test_in = held_X[perm[k:]]
            tpr_w = tpr_at_fpr(centroid_scores(cen_in, test_in), centroid_scores(cen_in, neg))
            cross.append(tpr_c); within.append(tpr_w)
            rows.append(dict(model=model, held_out=SHORT[held], n=len(held_X),
                             cross_mechanism_TPR=round(tpr_c, 3), within_mechanism_TPR=round(tpr_w, 3)))
            # full matrix row: each train family -> this held family
            for f in fams:
                fc = X[((full["category"] == f) & (full["label"] == 1)).to_numpy()].mean(0)
                rows[-1][f"from_{SHORT[f]}"] = round(
                    tpr_at_fpr(centroid_scores(fc, held_X), centroid_scores(fc, neg)), 3)
        labs = [SHORT[f] for f in fams]; xx = np.arange(len(labs)); w = 0.38
        ax.bar(xx - w/2, cross, w, color="#c0392b", label="cross-mechanism centroid")
        ax.bar(xx + w/2, within, w, color="#27ae60", label="own-mechanism (few-shot k=25)")
        ax.set_xticks(xx); ax.set_xticklabels(labs); ax.set_ylim(0, 1.05)
        ax.set_title(f"{model} L{L}"); ax.legend(fontsize=8)
    axs[0].set_ylabel("TPR @ FPR<2% (held-out family)")
    fig.suptitle("Transfer holds within a mechanism, collapses across mechanisms")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_lofo.png", dpi=200)
    save_table("table3_lofo", pd.DataFrame(rows))
    print("saved fig5_lofo.png")

# ───────────────────────── Fig 6: few-shot data efficiency ─────────────────────────

def fewshot():
    ks = [5, 10, 25, 50, 100]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    rows = []
    for ax, model in zip(axs, REPOS):
        L = MAIN_LAYER[model]
        tr, va, te = load_splits(model)
        full = pd.concat([tr, va, te], ignore_index=True)
        X = X_of(full, L); m = masks(full)
        neg = X[m["refused"] | m["benign"]]
        # mixed = all jailbreak families pooled
        jb_all = X[m["jailbroken"]]
        for label_, pool in [("mixed (all families)", jb_all)]:
            means, los, his = [], [], []
            for k in ks:
                vals = []
                for _ in range(50):
                    if len(pool) <= k + 5:
                        continue
                    perm = RNG.permutation(len(pool))
                    cen = pool[perm[:k]].mean(0); test = pool[perm[k:k+200]]
                    vals.append(tpr_at_fpr(centroid_scores(cen, test), centroid_scores(cen, neg)))
                means.append(np.mean(vals)); los.append(np.percentile(vals, 2.5)); his.append(np.percentile(vals, 97.5))
                rows.append(dict(model=model, pool=label_, k=k, tpr=round(float(np.mean(vals)), 3)))
            ax.plot(ks, means, "-o", label=label_)
            ax.fill_between(ks, los, his, alpha=0.2)
        ax.set_xlabel("# jailbreak examples to build centroid"); ax.set_title(f"{model} L{L}")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    axs[0].set_ylabel("held-out jailbreak TPR @ FPR<2%")
    fig.suptitle("Geometry → data-efficient detection")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_fewshot.png", dpi=200)
    save_table("table4_fewshot", pd.DataFrame(rows))
    print("saved fig6_fewshot.png")

# ───────────────────────── Fig 7: labeling matters ─────────────────────────

def labeling():
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axs, REPOS):
        L = MAIN_LAYER[model]
        tr, va, te = load_splits(model)
        full = pd.concat([tr, va, te], ignore_index=True)
        X = X_of(full, L); m = masks(full)
        nb = m["nonben"]
        origin = (full["category"].str.startswith("jailbreak")).to_numpy()  # literature: from-attack
        behavior = m["jailbroken"]                                          # ours: actually complied
        # AUROC of a probe under each labeling target, non-benign only (hard negatives)
        Xnb = X[nb]
        for name, y in [("origin-label", origin[nb].astype(int)), ("behavior-label", behavior[nb].astype(int))]:
            if len(np.unique(y)) == 2:
                p = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xnb, y)
                a, lo, hi = auroc_ci(y, p.decision_function(Xnb))
                rows.append(dict(model=model, labeling=name, auroc=round(a, 3),
                                 lo=round(lo, 3), hi=round(hi, 3)))
        # 2D PCA colored by behavior label among non-benign
        Z = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(Xnb))
        beh = behavior[nb]
        ax.scatter(Z[~beh, 0], Z[~beh, 1], s=8, alpha=0.4, label="refused (label 0)")
        ax.scatter(Z[beh, 0], Z[beh, 1], s=8, alpha=0.4, label="jailbroken (label 1)")
        ax.set_title(f"{model} L{L}: non-benign, behavior-labeled"); ax.legend(fontsize=8)
    fig.suptitle("Labeling by behavior vs prompt-origin changes the geometry")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig7_labeling.png", dpi=200)
    save_table("table5_labeling", pd.DataFrame(rows))
    print("saved fig7_labeling.png")

# ───────────────────────── dispatch ─────────────────────────

ASSETS = {"table1": table1, "targets": targets, "families": families, "axes": axes,
          "lofo": lofo, "fewshot": fewshot, "labeling": labeling}


def main():
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = list(ASSETS) if which == "all" else [which]
    for name in todo:
        if name not in ASSETS:
            print(f"unknown asset '{name}'. options: {list(ASSETS)} or 'all'"); continue
        print(f"\n{'='*60}\n[{name}]\n{'='*60}", flush=True)
        try:
            ASSETS[name]()
        except Exception as e:
            import traceback
            print(f"[{name}] FAILED: {e}"); traceback.print_exc()
    print(f"\nDone. Assets in {OUT}/")


if __name__ == "__main__":
    main()

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
OVERRIDE_LAYER = None                        # set via 2nd CLI arg to render detail figs at another layer
MIN_N = 25


def op_layer(model):
    """Operating layer for single-layer figures (CLI override if valid for the model)."""
    return OVERRIDE_LAYER if (OVERRIDE_LAYER in REPOS[model][1]) else MAIN_LAYER[model]


def suf():
    return f"_L{OVERRIDE_LAYER}" if OVERRIDE_LAYER is not None else ""
N_BOOT = 1000

# ───────────────────────── plotting style (publication) ─────────────────────────
# Okabe–Ito colorblind-safe palette; one consistent color per concept across all figures.
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#999999"}
FAM_COLOR = {"benign": CB["grey"], "harmful_direct": CB["orange"],
             "jailbreak_gcg_universal": CB["blue"], "jailbreak_artprompt": CB["vermillion"],
             "jailbreak_gcg_individual": CB["purple"],
             "GCG-univ": CB["blue"], "ArtPrompt": CB["vermillion"],
             "GCG-indiv": CB["purple"], "Direct": CB["orange"]}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
        "legend.fontsize": 9, "legend.frameon": False,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    })


def save_fig(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    fig.savefig(f"{OUT}/{name}.pdf")     # vector copy for LaTeX
    plt.close(fig)
    print(f"saved {name}.png + {name}.pdf")


def annotate_bars(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.015, fmt.format(h),
                ha="center", va="bottom", fontsize=8, fontweight="bold")


def annotate_signed(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h),
                ha="center", va="bottom" if h >= 0 else "top", fontsize=8)

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
    tcol = {"T1": CB["grey"], "T3": CB["blue"]}
    tlab = {"T1": "T1 intent (benign vs harmful topic)",
            "T3": "T3 success (jailbroken vs refused-harmful)"}
    for i, t in enumerate(["T1", "T3"]):
        vals = [bars[m][t][0] for m in models]
        err = [[bars[m][t][0] - bars[m][t][1] for m in models],
               [bars[m][t][2] - bars[m][t][0] for m in models]]
        bb = ax.bar(x + (i - 0.5) * w, vals, w, yerr=err, capsize=4,
                    color=tcol[t], label=tlab[t])
        for b, v, eu in zip(bb, vals, err[1]):
            ax.text(b.get_x() + b.get_width() / 2, v + eu + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0.5, ls="--", c=CB["vermillion"], lw=1)
    ax.text(ax.get_xlim()[1], 0.5, " chance", va="center", ha="left",
            fontsize=8, color=CB["vermillion"])
    ax.set_xticks(x); ax.set_xticklabels([m.capitalize() for m in models]); ax.set_ylim(0.4, 1.06)
    ax.set_ylabel("test AUROC (95% CI)")
    ax.set_title("Same activations, different detection target\n→ very different accuracy")
    ax.legend(loc="lower center")
    save_fig(fig, "fig2_targets")

# ───────────────────────── Fig 3: MDS + cosine heatmap ─────────────────────────

def families():
    for model in REPOS:
        L = op_layer(model)
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
        ax.grid(False)
        ax.set_title(f"{model.capitalize()} (layer {L}): attack families occupy distinct\n"
                     f"directions (diagonal = within-family ceiling)")
        save_fig(fig, f"fig3_heatmap_{model}{suf()}")
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
                ax.scatter(emb[sel, 0], emb[sel, 1], s=14, alpha=0.6,
                           color=FAM_COLOR.get(c, CB["grey"]),
                           label=f"{SHORT.get(c, c)} (n={len(sel)})")
        ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
        ax.set_title(f"{model.capitalize()} (layer {L}): activation geometry by category")
        ax.legend(loc="best", markerscale=1.6)
        save_fig(fig, f"fig3_mds_{model}{suf()}")

# ───────────────────────── Fig 4: harm vs refusal axis ─────────────────────────

def axes():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axs, REPOS):
        L = op_layer(model)
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
        b1 = ax.bar(xx - w/2, [r[1] for r in rows], w, color=CB["orange"], label="harm-belief axis")
        b2 = ax.bar(xx + w/2, [r[2] for r in rows], w, color=CB["blue"], label="refusal-suppression axis")
        annotate_signed(ax, b1); annotate_signed(ax, b2)
        ax.axhline(0, c="k", lw=0.8); ax.set_xticks(xx); ax.set_xticklabels(labs)
        ax.set_title(f"{model.capitalize()} (layer {L})"); ax.legend()
    axs[0].set_ylabel("projection of family's jailbreak shift\n(onto unit axis)")
    fig.suptitle("Which internal axis does each attack family move along?")
    save_fig(fig, f"fig4_axes{suf()}")

# ───────────────────────── Fig 5 + Table 3: LOFO ─────────────────────────

def lofo():
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axs, REPOS):
        L = op_layer(model)
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
            s_pos_c, s_neg_c = centroid_scores(cen_cross, held_X), centroid_scores(cen_cross, neg)
            tpr_c = tpr_at_fpr(s_pos_c, s_neg_c)
            auc_c = roc_auc_score([1]*len(s_pos_c) + [0]*len(s_neg_c), np.r_[s_pos_c, s_neg_c])
            # within-mechanism (few-shot, k=25): centroid from held family itself
            perm = RNG.permutation(len(held_X)); k = min(25, len(held_X)//2)
            cen_in = held_X[perm[:k]].mean(0); test_in = held_X[perm[k:]]
            s_pos_w, s_neg_w = centroid_scores(cen_in, test_in), centroid_scores(cen_in, neg)
            tpr_w = tpr_at_fpr(s_pos_w, s_neg_w)
            auc_w = roc_auc_score([1]*len(s_pos_w) + [0]*len(s_neg_w), np.r_[s_pos_w, s_neg_w])
            cross.append(tpr_c); within.append(tpr_w)
            rows.append(dict(model=model, held_out=SHORT[held], n=len(held_X),
                             cross_mechanism_TPR=round(tpr_c, 3), cross_mechanism_AUROC=round(float(auc_c), 3),
                             within_mechanism_TPR=round(tpr_w, 3), within_mechanism_AUROC=round(float(auc_w), 3)))
            # full matrix row: each train family -> this held family
            for f in fams:
                fc = X[((full["category"] == f) & (full["label"] == 1)).to_numpy()].mean(0)
                rows[-1][f"from_{SHORT[f]}"] = round(
                    tpr_at_fpr(centroid_scores(fc, held_X), centroid_scores(fc, neg)), 3)
        labs = [SHORT[f] for f in fams]; xx = np.arange(len(labs)); w = 0.38
        b1 = ax.bar(xx - w/2, cross, w, color=CB["vermillion"], label="cross-mechanism centroid")
        b2 = ax.bar(xx + w/2, within, w, color=CB["green"], label="own-mechanism (few-shot k=25)")
        annotate_bars(ax, b1); annotate_bars(ax, b2)
        ax.set_xticks(xx); ax.set_xticklabels(labs); ax.set_ylim(0, 1.08)
        ax.set_xlabel("held-out attack family")
        ax.set_title(f"{model.capitalize()} (layer {L})"); ax.legend(loc="upper center")
    axs[0].set_ylabel("TPR @ FPR < 2%")
    fig.suptitle("Detection transfers within a mechanism, collapses across mechanisms")
    save_fig(fig, f"fig5_lofo{suf()}")
    save_table(f"table3_lofo{suf()}", pd.DataFrame(rows))

# ───────────────────────── Fig 6: few-shot data efficiency ─────────────────────────

def fewshot():
    ks = [5, 10, 25, 50, 100]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    rows = []
    for ax, model in zip(axs, REPOS):
        L = op_layer(model)
        tr, va, te = load_splits(model)
        full = pd.concat([tr, va, te], ignore_index=True)
        X = X_of(full, L); m = masks(full)
        pool = X[m["jailbroken"]]                       # all successful jailbreaks pooled
        neg_settings = [("benign negatives (easy)", X[m["benign"]]),
                        ("refused-harmful negatives (honest)", X[m["refused"] | m["benign"]])]
        for label_, neg in neg_settings:
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
                rows.append(dict(model=model, negatives=label_, k=k, tpr=round(float(np.mean(vals)), 3)))
            col = CB["grey"] if "easy" in label_ else CB["blue"]
            ax.plot(ks, means, "-o", color=col, label=label_)
            ax.fill_between(ks, los, his, alpha=0.18, color=col)
        ax.set_xlabel("# jailbreak examples used to build centroid")
        ax.set_title(f"{model.capitalize()} (layer {L})")
        ax.set_ylim(0, 1.05); ax.legend(loc="center right")
    axs[0].set_ylabel("held-out jailbreak TPR @ FPR < 2%")
    fig.suptitle("Data-efficient — but the ceiling collapses under honest (hard) negatives")
    save_fig(fig, f"fig6_fewshot{suf()}")
    save_table(f"table4_fewshot{suf()}", pd.DataFrame(rows))

# ───────────────────────── Fig 7: labeling matters ─────────────────────────

def labeling():
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axs, REPOS):
        L = op_layer(model)
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
        ax.scatter(Z[~beh, 0], Z[~beh, 1], s=10, alpha=0.5, color=CB["sky"], label="refused (label 0)")
        ax.scatter(Z[beh, 0], Z[beh, 1], s=10, alpha=0.5, color=CB["vermillion"], label="jailbroken (label 1)")
        ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
        ax.set_title(f"{model.capitalize()} (layer {L}): non-benign, behavior-labeled")
        ax.legend(loc="best", markerscale=1.6)
    fig.suptitle("Behavior labels (did it comply?) ≠ prompt-origin labels (was it an attack?)")
    save_fig(fig, f"fig7_labeling{suf()}")
    save_table(f"table5_labeling{suf()}", pd.DataFrame(rows))

# ───────────────────────── Fig 8 + Table 6: layer robustness ─────────────────────────

def layersweep():
    """Are the headline claims layer-robust? Cross- vs within-mechanism centroid
    AUROC at EVERY layer, both models."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    rows = []
    for ax, (model, (_, layers)) in zip(axs, REPOS.items()):
        tr, va, te = load_splits(model)
        full = pd.concat([tr, va, te], ignore_index=True)
        m = masks(full)
        fams = [f for f in JB_FAMILIES
                if int(((full["category"] == f) & (full["label"] == 1)).sum()) >= MIN_N]
        cross_L, within_L = [], []
        for L in layers:
            X = X_of(full, L)
            neg = X[m["refused"] | m["benign"]]
            cvals, wvals = [], []
            for held in fams:
                held_X = X[((full["category"] == held) & (full["label"] == 1)).to_numpy()]
                other = np.zeros(len(full), bool)
                for f in fams:
                    if f != held:
                        other |= ((full["category"] == f) & (full["label"] == 1)).to_numpy()
                cen_cross = X[other].mean(0)
                sp, sn = centroid_scores(cen_cross, held_X), centroid_scores(cen_cross, neg)
                cvals.append(roc_auc_score([1]*len(sp) + [0]*len(sn), np.r_[sp, sn]))
                perm = RNG.permutation(len(held_X)); k = min(25, len(held_X)//2)
                cen_in = held_X[perm[:k]].mean(0); test_in = held_X[perm[k:]]
                sp2, sn2 = centroid_scores(cen_in, test_in), centroid_scores(cen_in, neg)
                wvals.append(roc_auc_score([1]*len(sp2) + [0]*len(sn2), np.r_[sp2, sn2]))
            c, w = float(np.mean(cvals)), float(np.mean(wvals))
            cross_L.append(c); within_L.append(w)
            rows.append(dict(model=model, layer=L,
                             cross_mechanism_AUROC=round(c, 3), within_mechanism_AUROC=round(w, 3)))
        ax.plot(layers, cross_L, "-o", color=CB["vermillion"], label="cross-mechanism")
        ax.plot(layers, within_L, "-o", color=CB["green"], label="within-mechanism")
        ax.axhline(0.5, ls="--", c=CB["grey"], lw=1)
        ax.text(layers[-1], 0.5, " chance", va="center", ha="left", fontsize=8, color=CB["grey"])
        ax.set_xlabel("layer"); ax.set_title(model.capitalize()); ax.set_ylim(0.2, 1.02)
        ax.legend(loc="center right")
    axs[0].set_ylabel("centroid detector AUROC (held-out family)")
    fig.suptitle("Transfer collapse is layer-robust: cross-mechanism ≈ chance at every depth")
    save_fig(fig, "fig8_layersweep")
    save_table("table6_layersweep", pd.DataFrame(rows))


# ───────────────────────── dispatch ─────────────────────────

ASSETS = {"table1": table1, "targets": targets, "families": families, "axes": axes,
          "lofo": lofo, "fewshot": fewshot, "labeling": labeling, "layersweep": layersweep}


def main():
    global OVERRIDE_LAYER
    os.makedirs(OUT, exist_ok=True)
    setup_style()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if len(sys.argv) > 2:
        OVERRIDE_LAYER = int(sys.argv[2])
        print(f"[override] single-layer figures will use layer {OVERRIDE_LAYER}")
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

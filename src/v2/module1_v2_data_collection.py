"""
Module 1 (v2) — Build the v2 prompt pool for a chosen target model.

Pipeline:
  1. Pull Alpaca benign passages.
  2. Pull AdvBench + HarmBench harmful seeds (deduped, length-filtered).
     Save the seeds separately so the attack scripts can read them.
  3. (Optionally) run the three attacks on the seeds:
        - ArtPrompt   (deterministic; runs in this script)
        - GCG-Universal (loads suffix.txt previously optimized)
        - GCG-Individual (loads results.jsonl previously written)
     The two GCG attacks are GPU-heavy and live in separate scripts; this
     script only consumes their outputs.
  4. Assemble the four-category pool with the SAME schema as v1's
     module1_data_collection.py (parallel `prompts / sources / categories`
     arrays + counts + n_total) and save to <artifact_dir>/prompts/prompt_pool.pt.

Target models (selected via --target):
  - gemma  (default):  google/gemma-2-2b-it, layers {5,10,15,20,25}, 2304-dim
  - vicuna:            lmsys/vicuna-7b-v1.3, layers {4,10,15,20,25,30}, 4096-dim

After this, run Module 2 with the matching paths and model:
    Gemma:
      python src/module2_labeling_extraction.py \
          --prompt-pool artifacts/v2/gemma/prompts/prompt_pool.pt \
          --output-dir  artifacts/v2/gemma \
          --model       google/gemma-2-2b-it \
          --layers      5 10 15 20 25

    Vicuna:
      python src/module2_labeling_extraction.py \
          --prompt-pool artifacts/v2/vicuna/prompts/prompt_pool.pt \
          --output-dir  artifacts/v2/vicuna \
          --model       lmsys/vicuna-7b-v1.3 \
          --layers      4 10 15 20 25 30

Then push:
    python src/module2b_push_to_hf.py \
        --source-dir <artifact_dir> \
        --repo-id    <hf_repo> \
        --layers     <space-separated layers>
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.v2.attacks import artprompt
from src.v2.config import (
    ATTACK_SEED_DATASETS,
    BENIGN_DATASET,
    CATEGORY_BENIGN,
    CATEGORY_HARMFUL_DIRECT,
    CATEGORY_JB_ARTPROMPT,
    CATEGORY_JB_GCG_INDIVIDUAL,
    CATEGORY_JB_GCG_UNIVERSAL,
    DEFAULT_TARGET,
    SEED,
    get_config,
    list_targets,
)


# ── Benign collector ─────────────────────────────────────────────────────────

def collect_alpaca(n: int, tokenizer_name: str,
                   min_tokens: int = 8, max_tokens: int = 512) -> List[str]:
    """
    Load Alpaca, length-filter (using the target model's tokenizer), take first n.

    Alpaca instructions are short (typical 5-30 tokens), so min_tokens defaults
    to 8 — using v1's WikiText-tuned 64 strips ~99% of rows. We also append the
    optional `input` field when present, matching how Alpaca was originally fed
    to the model during instruction tuning.

    `tokenizer_name` is the HuggingFace model id whose tokenizer we use for
    the length filter — different tokenizers produce different token counts
    so the filter must match the target model.
    """
    ds = load_dataset(
        BENIGN_DATASET["name"],
        BENIGN_DATASET["config"],
        split=BENIGN_DATASET["split"],
    )
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    out: List[str] = []
    for row in ds:
        instr = (row.get(BENIGN_DATASET["field"]) or "").strip()
        if not instr:
            continue
        extra = (row.get("input") or "").strip()
        text = f"{instr}\n{extra}" if extra else instr

        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        if min_tokens <= n_tok <= max_tokens:
            out.append(text)
        if len(out) >= n:
            break
    return out


# ── Harmful seed collector (the inputs to the attacks) ──────────────────────

def collect_attack_seeds(tokenizer_name: str,
                         min_tokens: int = 8, max_tokens: int = 256) -> List[str]:
    """
    Collect AdvBench + HarmBench harmful prompts, dedupe, light length filter.

    These seeds are what the three attacks transform. Length filter is loose
    (8 tokens min) because AdvBench prompts are very short. The tokenizer
    must match the target model — different tokenizers count tokens
    differently and the filter result will differ.
    """
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    seeds: List[str] = []
    for cfg in ATTACK_SEED_DATASETS:
        try:
            ds = load_dataset(cfg["name"], cfg["config"], split=cfg["split"])
        except Exception as e:
            print(f"    ! Failed to load {cfg['name']}: {e}")
            continue
        n_before = len(seeds)
        for row in ds:
            text = (row.get(cfg["field"]) or "").strip()
            if not text:
                continue
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            if min_tokens <= n_tok <= max_tokens:
                seeds.append(text)
        print(f"    + {cfg['name']}: +{len(seeds) - n_before} rows")

    # Dedupe (case + whitespace insensitive), preserve order.
    seen: set[str] = set()
    deduped: List[str] = []
    for s in seeds:
        key = " ".join(s.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    print(f"    + dedup kept {len(deduped)}/{len(seeds)} seeds.")
    return deduped


# ── Attack-output loaders ────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_gcg_universal(out_dir: Path) -> List[str]:
    """Read attacked prompts produced by gcg_universal.py."""
    rows = _load_jsonl(out_dir / "results.jsonl")
    return [r["attacked_prompt"] for r in rows if r.get("attacked_prompt")]


def load_gcg_individual(out_dir: Path) -> List[str]:
    """Read attacked prompts produced by gcg_individual.py."""
    rows = _load_jsonl(out_dir / "results.jsonl")
    return [r["attacked_prompt"] for r in rows if r.get("attacked_prompt")]


# ── Pool assembly (matches v1's prompt_pool.pt schema) ──────────────────────

def assemble_pool(
    benign: List[str],
    harmful_direct: List[str],
    artprompt_outputs: List[str],
    gcg_universal_outputs: List[str],
    gcg_individual_outputs: List[str],
    seed: int = SEED,
) -> dict:
    """
    Build the final pool with parallel prompts / sources / categories arrays.
    Schema matches v1 so module2_labeling_extraction.py can consume it
    unchanged.
    """
    prompts: List[str] = []
    sources: List[str] = []
    categories: List[str] = []

    # 1) Benign
    for p in benign:
        prompts.append(p)
        sources.append("alpaca")
        categories.append(CATEGORY_BENIGN)

    # 2) Harmful_direct (the raw seeds, unwrapped)
    for p in harmful_direct:
        prompts.append(p)
        sources.append("advbench+harmbench")
        categories.append(CATEGORY_HARMFUL_DIRECT)

    # 3) ArtPrompt-wrapped
    for p in artprompt_outputs:
        prompts.append(p)
        sources.append("attack_artprompt")
        categories.append(CATEGORY_JB_ARTPROMPT)

    # 4) GCG-Universal
    for p in gcg_universal_outputs:
        prompts.append(p)
        sources.append("attack_gcg_universal")
        categories.append(CATEGORY_JB_GCG_UNIVERSAL)

    # 5) GCG-Individual
    for p in gcg_individual_outputs:
        prompts.append(p)
        sources.append("attack_gcg_individual")
        categories.append(CATEGORY_JB_GCG_INDIVIDUAL)

    # Full-text dedupe (whitespace + case normalized) — same as v1 main.
    seen: set[str] = set()
    keep_idx: List[int] = []
    for i, p in enumerate(prompts):
        key = " ".join(p.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)
    n_before = len(prompts)
    prompts = [prompts[i] for i in keep_idx]
    sources = [sources[i] for i in keep_idx]
    categories = [categories[i] for i in keep_idx]
    print(f"[dedupe] removed {n_before - len(prompts)} duplicates; "
          f"{len(prompts)} remaining.")

    # Shuffle so downstream splits stay class-balanced.
    rng = random.Random(seed)
    idx = list(range(len(prompts)))
    rng.shuffle(idx)
    prompts = [prompts[i] for i in idx]
    sources = [sources[i] for i in idx]
    categories = [categories[i] for i in idx]

    counts = {
        CATEGORY_BENIGN:               sum(1 for c in categories if c == CATEGORY_BENIGN),
        CATEGORY_HARMFUL_DIRECT:       sum(1 for c in categories if c == CATEGORY_HARMFUL_DIRECT),
        CATEGORY_JB_ARTPROMPT:         sum(1 for c in categories if c == CATEGORY_JB_ARTPROMPT),
        CATEGORY_JB_GCG_UNIVERSAL:     sum(1 for c in categories if c == CATEGORY_JB_GCG_UNIVERSAL),
        CATEGORY_JB_GCG_INDIVIDUAL:    sum(1 for c in categories if c == CATEGORY_JB_GCG_INDIVIDUAL),
    }
    return {
        "prompts": prompts,
        "sources": sources,
        "categories": categories,
        "counts": counts,
        "n_total": len(prompts),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build the v2 prompt pool for a chosen target "
                    "(benign Alpaca + AdvBench/HarmBench seeds + 3 attacks)."
    )
    parser.add_argument(
        "--target", type=str, default=DEFAULT_TARGET, choices=list_targets(),
        help=f"Target model to build the v2 dataset for. "
             f"Default: {DEFAULT_TARGET!r}. Available: {list_targets()}.",
    )
    parser.add_argument("--n-benign", type=int, default=BENIGN_DATASET["n"])
    parser.add_argument("--seed",     type=int, default=SEED)
    parser.add_argument(
        "--seeds-out", type=str, default=None,
        help="Override path for the harmful-seed list. Defaults to the "
             "target's seeds_out path from config.",
    )
    parser.add_argument(
        "--gcg-universal-dir", type=str, default=None,
        help="Override path for GCG-Universal results. Defaults to "
             "the target's gcg_universal_dir.",
    )
    parser.add_argument(
        "--gcg-individual-dir", type=str, default=None,
        help="Override path for GCG-Individual results. Defaults to "
             "the target's gcg_individual_dir.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override path for the final prompt_pool.pt. Defaults to "
             "the target's prompt_pool path.",
    )
    parser.add_argument("--skip-gcg", action="store_true",
                        help="Build pool with only ArtPrompt + benign + seeds; "
                             "useful when GCG hasn't been run yet.")
    args = parser.parse_args()

    # Resolve target configuration. Anything passed explicitly on the
    # command line wins over the per-target defaults.
    cfg = get_config(args.target)
    seeds_path        = Path(args.seeds_out         or cfg["seeds_out"])
    gcg_universal_dir = Path(args.gcg_universal_dir or cfg["gcg_universal_dir"])
    gcg_individual_dir = Path(args.gcg_individual_dir or cfg["gcg_individual_dir"])
    output_path       = Path(args.output            or cfg["prompt_pool"])
    artifact_dir      = cfg["artifact_dir"]
    tokenizer_name    = cfg["model_name"]
    hf_repo           = cfg["hf_repo"]

    print("=" * 60)
    print("MODULE 1 v2 — DATA COLLECTION")
    print(f"  Target: {args.target} ({tokenizer_name})")
    print(f"  Layers: {list(cfg['layers'])}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    if not os.getenv("HF_TOKEN"):
        print("[!] HF_TOKEN not set — gated datasets may fail.")

    # ── Benign ────────────────────────────────────────────────────────────
    print(f"\n[1/4] Benign (Alpaca), target={args.n_benign}")
    benign = collect_alpaca(args.n_benign, tokenizer_name=tokenizer_name)
    print(f"      collected {len(benign)} Alpaca instructions")

    # ── Harmful seeds ─────────────────────────────────────────────────────
    print(f"\n[2/4] Harmful seeds (AdvBench + HarmBench)")
    seeds = collect_attack_seeds(tokenizer_name=tokenizer_name)
    print(f"      total seeds: {len(seeds)}")

    # Persist the seed list so the GCG attack scripts can consume it.
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seeds": seeds, "target": args.target}, seeds_path)
    print(f"      saved seeds to {seeds_path}")

    # ── ArtPrompt (always; runs in-process) ───────────────────────────────
    print(f"\n[3/4] ArtPrompt transform on {len(seeds)} seeds")
    art_results = artprompt.transform_many(seeds, seed=args.seed)
    art_prompts = [r.final_prompt for r in art_results]
    print(f"      produced {len(art_prompts)} ArtPrompt prompts "
          f"({len(seeds) - len(art_prompts)} skipped — no maskable word)")

    # ── GCG outputs (loaded from disk; optionally skipped) ────────────────
    if args.skip_gcg:
        print("\n[4/4] Skipping GCG load (--skip-gcg)")
        gcg_u, gcg_i = [], []
    else:
        gcg_u = load_gcg_universal(gcg_universal_dir)
        gcg_i = load_gcg_individual(gcg_individual_dir)
        print(f"\n[4/4] GCG-Universal:  {len(gcg_u)} attacked prompts "
              f"loaded from {gcg_universal_dir}")
        print(f"      GCG-Individual: {len(gcg_i)} attacked prompts "
              f"loaded from {gcg_individual_dir}")
        if not gcg_u:
            print("      [WARN] No GCG-Universal results found. Run "
                  "src/v2/attacks/gcg_universal.py first or pass --skip-gcg.")
        if not gcg_i:
            print("      [WARN] No GCG-Individual results found. Run "
                  "src/v2/attacks/gcg_individual.py first or pass --skip-gcg.")

    # ── Assemble pool ─────────────────────────────────────────────────────
    print("\n[assemble] building final pool...")
    pool = assemble_pool(
        benign=benign,
        harmful_direct=seeds,            # raw seeds also become harmful_direct rows
        artprompt_outputs=art_prompts,
        gcg_universal_outputs=gcg_u,
        gcg_individual_outputs=gcg_i,
        seed=args.seed,
    )
    # Tag the pool with target metadata so downstream Module 2 invocations
    # can sanity-check the layout (and humans reading prompt_pool.pt know
    # which model it was built for).
    pool["target"] = args.target
    pool["model_name"] = cfg["model_name"]
    pool["layers"] = list(cfg["layers"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pool, output_path)

    print("\n" + "=" * 60)
    print(f"DONE — saved {pool['n_total']} prompts to {output_path}")
    for cat, n in pool["counts"].items():
        print(f"    {cat:<28s}: {n}")
    print("=" * 60)
    layers_str = " ".join(str(L) for L in cfg["layers"])
    print(
        f"\nNext steps:\n"
        f"  python src/module2_labeling_extraction.py \\\n"
        f"      --prompt-pool {output_path} \\\n"
        f"      --output-dir  {artifact_dir} \\\n"
        f"      --model       {cfg['model_name']} \\\n"
        f"      --layers      {layers_str}\n"
        f"\n"
        f"  python src/module2b_push_to_hf.py \\\n"
        f"      --source-dir {artifact_dir} \\\n"
        f"      --repo-id    {hf_repo} \\\n"
        f"      --layers     {layers_str}\n"
    )


if __name__ == "__main__":
    main()

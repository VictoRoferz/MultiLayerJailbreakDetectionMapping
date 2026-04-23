"""
Dataset audit for Module 1 + Module 2 output.

Measures label quality, prompt/response shape, and class balance so that
changes to the judge rubric or token budget can be evaluated quantitatively.

Usage:
    python src/dataset_audit.py \
        --full-dataset artifacts/labeled_data/full_dataset.pt \
        --prompt-pool artifacts/prompts/prompt_pool.pt \
        --results-path results/dataset_audit.json

The script is read-only. It prints a human summary and writes one JSON
report. Run it once before any rebuild (baseline) and again after
(comparison) using the same schema.
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

TOKENIZER_NAME = "google/gemma-2-2b-it"
CURRENT_MAX_PROMPT_TOKENS = 512  # value used by module2 today

REFUSAL_MARKERS = [
    r"\bi can't\b", r"\bi cannot\b", r"\bi'm not able\b", r"\bi am not able\b",
    r"\bi won't\b", r"\bi will not\b", r"\bi'm unable\b", r"\bi am unable\b",
    r"\bas an ai\b", r"\bi'm sorry, but\b", r"\bi am sorry, but\b",
    r"\bi'm designed\b", r"\bi am designed\b", r"\bi must decline\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_MARKERS), re.IGNORECASE)

PREAMBLE_MARKERS = [
    r"^\s*let's (begin|start|go|do this)",
    r"^\s*okay[,!.]?\s*(let's|here we go|i'm ready)",
    r"^\s*i'?m ready",
    r"^\s*sure[,!.]?\s*(i can|i'll|let's)",
    r"^\s*here we go",
    r"^\s*alright[,!.]",
    r"^\s*let's explore",
    r"^\s*ready to (get started|begin)",
]
PREAMBLE_RE = re.compile("|".join(PREAMBLE_MARKERS), re.IGNORECASE | re.MULTILINE)


def percentiles(values: List[int], ps=(50, 90, 99, 100)) -> Dict[str, int]:
    if not values:
        return {f"p{p}": 0 for p in ps}
    arr = np.asarray(values)
    return {f"p{p}": int(np.percentile(arr, p)) for p in ps}


def gini(counts: List[int]) -> float:
    if not counts:
        return 0.0
    arr = np.sort(np.asarray(counts, dtype=float))
    n = len(arr)
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def is_preamble_only(response: str, word_threshold: int = 50) -> bool:
    """Preamble-only: short AND no concrete content markers."""
    words = response.split()
    if len(words) >= word_threshold:
        return False
    # Short responses that start with preamble phrases or contain only setup.
    if PREAMBLE_RE.search(response):
        return True
    # Short response with no numbers, no code fences, no colons suggesting steps.
    has_concrete = any(
        tok in response for tok in (":", "1.", "2.", "```", "Step", "step ")
    )
    return not has_concrete


def _compute_prompt_stats(prompts: List[str], categories: List[str],
                          tokenizer, source_label: str) -> Dict:
    """A1 + A2 on any (prompts, categories) pair."""
    cats = sorted(set(categories))
    lens_by_cat: Dict[str, List[int]] = {c: [] for c in cats}
    trunc_by_cat: Dict[str, int] = {c: 0 for c in cats}

    print(f"  tokenizing prompts from {source_label}...")
    for p, c in zip(prompts, categories):
        n_tok = len(tokenizer(p, add_special_tokens=False)["input_ids"])
        lens_by_cat[c].append(n_tok)
        if n_tok > CURRENT_MAX_PROMPT_TOKENS:
            trunc_by_cat[c] += 1

    summary: Dict[str, Dict] = {"_source": source_label}
    for c in cats:
        lens = lens_by_cat[c]
        summary[c] = {
            "n": len(lens),
            "lengths": percentiles(lens),
            "mean": float(np.mean(lens)) if lens else 0.0,
            "trunc_at_512": trunc_by_cat[c],
            "trunc_rate_at_512": trunc_by_cat[c] / len(lens) if lens else 0.0,
        }
    return summary


def audit_prompt_pool(pool_path: Path, tokenizer) -> Dict:
    """A1 + A2 from the prompt pool itself (pre-generation)."""
    if not pool_path.exists():
        return {}
    pool = torch.load(pool_path, weights_only=False)
    return _compute_prompt_stats(
        pool["prompts"], pool["categories"], tokenizer,
        source_label=f"prompt_pool ({pool_path.name})",
    )


def audit_prompts_from_full(full_path: Path, tokenizer) -> Dict:
    """Fallback A1 + A2 using prompts stored in full_dataset.pt.
    Note: these prompts are post-generation and may differ slightly from
    the pool if Module 2 applied any filtering, but for token-length and
    truncation analysis the difference is negligible."""
    d = torch.load(full_path, weights_only=False)
    return _compute_prompt_stats(
        d["prompts"], d["categories"], tokenizer,
        source_label=f"full_dataset fallback ({full_path.name})",
    )


def audit_full_dataset(path: Path, tokenizer) -> Dict:
    """A2–A10 from the labeled dataset."""
    d = torch.load(path, weights_only=False)
    prompts = d["prompts"]
    responses = d["responses"]
    categories = d["categories"]
    labels = d["labels"]
    judge_scores = d.get("judge_scores", [{}] * len(labels))

    n = len(labels)
    cats = sorted(set(categories))

    # A0 — basic counts
    label_counts = dict(Counter(labels))
    cat_counts = dict(Counter(categories))

    # A3 — response length by (category, label)
    print("  tokenizing responses...")
    resp_lens = [len(tokenizer(r, add_special_tokens=False)["input_ids"]) for r in responses]
    resp_len_by_cat_label: Dict[str, Dict[str, int]] = {}
    for c in cats:
        for lbl in (0, 1):
            key = f"{c}|label={lbl}"
            sel = [resp_lens[i] for i in range(n) if categories[i] == c and labels[i] == lbl]
            if sel:
                resp_len_by_cat_label[key] = {
                    "n": len(sel),
                    **percentiles(sel),
                    "mean": float(np.mean(sel)),
                }

    # A4 — refusal-marker hit rate in label=1 responses
    label1_responses = [responses[i] for i in range(n) if labels[i] == 1]
    refusal_hits = sum(1 for r in label1_responses if REFUSAL_RE.search(r))
    refusal_rate = refusal_hits / len(label1_responses) if label1_responses else 0.0

    # A5 — preamble-only rate in label=1
    preamble_hits = sum(1 for r in label1_responses if is_preamble_only(r))
    preamble_rate = preamble_hits / len(label1_responses) if label1_responses else 0.0

    # A6 — judge score distributions (label=1 only)
    def _collect(field):
        vals = [
            float(judge_scores[i].get(field, 0.0))
            for i in range(n)
            if labels[i] == 1 and isinstance(judge_scores[i], dict)
        ]
        if not vals:
            return {}
        arr = np.asarray(vals)
        return {
            "n": len(vals),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            **percentiles(vals, ps=(10, 50, 90)),
        }

    judge_dist = {
        f: _collect(f) for f in (
            "harmful_prompt",
            "perturbed_compliance",
            "perturbed_actionability",
            "benign_or_nonactionable",
        )
    }

    # A7 — per-category label=1 rate
    label1_rate_by_cat = {}
    for c in cats:
        n_c = cat_counts[c]
        n_c_pos = sum(1 for i in range(n) if categories[i] == c and labels[i] == 1)
        label1_rate_by_cat[c] = {
            "n": n_c,
            "n_label1": n_c_pos,
            "rate": n_c_pos / n_c if n_c else 0.0,
        }

    # A8 — template diversity (only applicable to jailbreak_wrapped).
    # Use the first 80 chars of each prompt as a cheap template signature
    # (matches the snippet convention used in module1).
    wrapped_idx = [i for i in range(n) if categories[i] == "jailbreak_wrapped"]
    template_counts: Counter = Counter()
    for i in wrapped_idx:
        sig = prompts[i][:80]
        template_counts[sig] += 1
    template_div = {
        "n_wrapped": len(wrapped_idx),
        "n_unique_templates": len(template_counts),
        "samples_per_template_p50": int(np.median(list(template_counts.values()))) if template_counts else 0,
        "samples_per_template_max": max(template_counts.values()) if template_counts else 0,
        "gini": gini(list(template_counts.values())),
    }

    # A9 — near-duplicate rate (first-200-char lowercase exact match)
    prefix_counts: Counter = Counter()
    for p in prompts:
        prefix_counts[p[:200].lower()] += 1
    n_dup = sum(c - 1 for c in prefix_counts.values() if c > 1)
    dup_rate = n_dup / n if n else 0.0

    # A10 — unjudged rate (label == -1)
    n_unjudged = sum(1 for lbl in labels if lbl == -1)

    return {
        "n_total": n,
        "label_counts": label_counts,
        "category_counts": cat_counts,
        "A3_response_length_by_cat_label": resp_len_by_cat_label,
        "A4_refusal_marker_rate_label1": refusal_rate,
        "A4_refusal_hits_label1": refusal_hits,
        "A4_label1_total": len(label1_responses),
        "A5_preamble_only_rate_label1": preamble_rate,
        "A5_preamble_hits_label1": preamble_hits,
        "A6_judge_score_distributions_label1": judge_dist,
        "A7_label1_rate_by_category": label1_rate_by_cat,
        "A8_template_diversity_wrapped": template_div,
        "A9_nearduplicate_rate": dup_rate,
        "A9_nearduplicate_pairs": n_dup,
        "A10_unjudged_count": n_unjudged,
        "A10_unjudged_rate": n_unjudged / n if n else 0.0,
    }


def print_report(prompt_audit: Dict, full_audit: Dict) -> None:
    print("\n" + "=" * 68)
    print("DATASET AUDIT")
    print("=" * 68)

    print(f"\n[A0] Sample counts")
    print(f"  Total samples:      {full_audit['n_total']}")
    print(f"  Label distribution: {full_audit['label_counts']}")
    print(f"  Category counts:    {full_audit['category_counts']}")

    if prompt_audit:
        src = prompt_audit.get("_source", "prompt_pool.pt")
        print(f"\n[A1 + A2] Prompt token lengths (source: {src})")
        print(f"  {'category':<22s} {'n':>6s} {'p50':>6s} {'p90':>6s} {'p99':>6s} {'max':>6s}  trunc@512")
        for c, s in prompt_audit.items():
            if c.startswith("_"):
                continue
            L = s["lengths"]
            print(f"  {c:<22s} {s['n']:>6d} {L['p50']:>6d} {L['p90']:>6d} {L['p99']:>6d} {L['p100']:>6d}  {s['trunc_rate_at_512']*100:>6.2f}%")

    print("\n[A3] Response token length by (category, label)")
    for key, v in full_audit["A3_response_length_by_cat_label"].items():
        print(f"  {key:<40s} n={v['n']:>4d}  p50={v['p50']:>4d}  p90={v['p90']:>4d}  mean={v['mean']:>5.1f}")

    print("\n[A4] Refusal-marker hit rate in label=1 responses")
    print(f"  Rate: {full_audit['A4_refusal_marker_rate_label1']*100:.2f}% "
          f"({full_audit['A4_refusal_hits_label1']} / {full_audit['A4_label1_total']})  "
          f"[target: near 0]")

    print("\n[A5] Preamble-only rate in label=1 responses")
    print(f"  Rate: {full_audit['A5_preamble_only_rate_label1']*100:.2f}% "
          f"({full_audit['A5_preamble_hits_label1']} / {full_audit['A4_label1_total']})  "
          f"[target: <10%]")

    print("\n[A6] Judge score distributions (label=1 only)")
    for f, d in full_audit["A6_judge_score_distributions_label1"].items():
        if not d:
            continue
        print(f"  {f:<30s} mean={d['mean']:.2f} std={d['std']:.2f} "
              f"p10={d['p10']:.1f} p50={d['p50']:.1f} p90={d['p90']:.1f}")

    print("\n[A7] Per-category label=1 rate")
    for c, s in full_audit["A7_label1_rate_by_category"].items():
        print(f"  {c:<22s} {s['n_label1']:>5d} / {s['n']:>5d} = {s['rate']*100:.1f}%")

    print("\n[A8] Template diversity (jailbreak_wrapped)")
    t = full_audit["A8_template_diversity_wrapped"]
    print(f"  n_wrapped={t['n_wrapped']}  unique_templates={t['n_unique_templates']}  "
          f"per_template_p50={t['samples_per_template_p50']} max={t['samples_per_template_max']}  "
          f"gini={t['gini']:.3f}")

    print("\n[A9] Near-duplicate (first-200-char match) rate")
    print(f"  {full_audit['A9_nearduplicate_rate']*100:.2f}% ({full_audit['A9_nearduplicate_pairs']} dup samples)")

    print("\n[A10] Unjudged (label=-1) rate")
    print(f"  {full_audit['A10_unjudged_rate']*100:.3f}% ({full_audit['A10_unjudged_count']} samples)")

    # -------- gates --------
    print("\n" + "-" * 68)
    print("GATES")
    print("-" * 68)
    gates: List[str] = []
    if prompt_audit:
        wrapped = prompt_audit.get("jailbreak_wrapped")
        if wrapped:
            passed = wrapped["trunc_rate_at_512"] < 0.20
            gates.append(f"  [{'OK' if passed else 'FAIL'}] Truncation @512 on jailbreak_wrapped: "
                         f"{wrapped['trunc_rate_at_512']*100:.1f}% (target <20%)")
    gates.append(f"  [{'OK' if full_audit['A5_preamble_only_rate_label1'] < 0.10 else 'FAIL'}] "
                 f"Preamble-only rate: {full_audit['A5_preamble_only_rate_label1']*100:.1f}% (target <10%)")
    gates.append(f"  [{'OK' if full_audit['A4_refusal_marker_rate_label1'] < 0.05 else 'FAIL'}] "
                 f"Refusal-marker rate: {full_audit['A4_refusal_marker_rate_label1']*100:.1f}% (target <5%)")
    gates.append(f"  [{'OK' if full_audit['A10_unjudged_rate'] < 0.05 else 'FAIL'}] "
                 f"Unjudged rate: {full_audit['A10_unjudged_rate']*100:.2f}% (target <5%)")
    for g in gates:
        print(g)
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-dataset", type=Path,
                   default=Path("artifacts/labeled_data/full_dataset.pt"))
    p.add_argument("--prompt-pool", type=Path,
                   default=Path("artifacts/prompts/prompt_pool.pt"))
    p.add_argument("--results-path", type=Path,
                   default=Path("results/dataset_audit.json"))
    args = p.parse_args()

    if not args.full_dataset.exists():
        raise FileNotFoundError(f"full_dataset.pt not found at {args.full_dataset}")

    print(f"Loading tokenizer {TOKENIZER_NAME}...")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "gemma-2-2b-it is a gated repo. Set HF_TOKEN in your environment "
            "(e.g. in Colab: os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN'))."
        )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=hf_token)

    print(f"Auditing prompt pool at {args.prompt_pool}...")
    prompt_audit = audit_prompt_pool(args.prompt_pool, tokenizer)
    if not prompt_audit:
        print(f"  prompt_pool.pt missing; falling back to prompts in {args.full_dataset}")
        prompt_audit = audit_prompts_from_full(args.full_dataset, tokenizer)

    print(f"Auditing labeled dataset at {args.full_dataset}...")
    full_audit = audit_full_dataset(args.full_dataset, tokenizer)

    print_report(prompt_audit, full_audit)

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump({"prompt_pool": prompt_audit, "full_dataset": full_audit},
                  f, indent=2, default=str)
    print(f"Report written to {args.results_path}")


if __name__ == "__main__":
    main()

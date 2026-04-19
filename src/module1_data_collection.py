"""
Module 1 — Data Collection

Downloads datasets from HuggingFace and assembles a unified prompt pool for
Module 2 (labeling + activation extraction).

Three prompt categories:
  1. benign            : WikiText-103 passages. Produce benign responses → label 0.
  2. harmful_direct    : AdvBench + HarmBench plain harmful requests. Gemma usually
                         refuses → mostly label 0, occasional label 1.
  3. jailbreak_wrapped : TrustAIRLab in-the-wild jailbreak templates.
                         Standalone templates are used as-is (they already embed an
                         attack); placeholder templates are filled with harmful
                         requests drawn from AdvBench / HarmBench.

Output: artifacts/prompts/prompt_pool.pt
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# ------------------------------------------------------------
# Dataset registry
# ------------------------------------------------------------

BENIGN_DATASET = {
    "name": "wikitext",
    "config": "wikitext-103-raw-v1",
    "split": "train",
    "field": "text",
}

HARMFUL_REQUEST_DATASETS = [
    {"name": "walledai/AdvBench", "config": None, "split": "train", "field": "prompt"},
    {"name": "walledai/HarmBench", "config": "standard", "split": "train", "field": "prompt"},
]

JAILBREAK_TEMPLATE_DATASET = {
    "name": "TrustAIRLab/in-the-wild-jailbreak-prompts",
    "config": "jailbreak_2023_05_07",
    "split": "train",
    "field": "prompt",
}

MODEL_NAME_FOR_TOKENIZER = "google/gemma-2-2b-it"

# ------------------------------------------------------------
# Template-filling utilities
# ------------------------------------------------------------

PLACEHOLDER_PATTERNS = [
    "{REQUEST}", "[REQUEST]", "{prompt}", "[INSERT PROMPT HERE]",
    "[PROMPT]", "<REQUEST>", "{{prompt}}", "[your prompt]",
    "[your question]", "{{REQUEST}}", "<<PROMPT>>", "[INSERT]",
]


def _find_placeholder(template: str) -> Optional[str]:
    """Return the first matching placeholder (case-insensitive), or None."""
    lower = template.lower()
    for p in PLACEHOLDER_PATTERNS:
        if p.lower() in lower:
            return p
    return None


def fill_template(template: str, request: str) -> str:
    """Fill a jailbreak template with a harmful request.

    If a known placeholder exists, substitute it in-place. Otherwise append
    the request at the end of the template with a minimal separator.
    """
    p = _find_placeholder(template)
    if p is not None:
        idx = template.lower().find(p.lower())
        return template[:idx] + request + template[idx + len(p):]
    return f"{template}\n\n{request}"


# ------------------------------------------------------------
# Collectors
# ------------------------------------------------------------

def collect_benign(n: int, min_tokens: int = 64, max_tokens: int = 256) -> List[str]:
    """Collect benign passages from WikiText-103, filtered by token length."""
    ds = load_dataset(
        BENIGN_DATASET["name"],
        BENIGN_DATASET["config"],
        split=BENIGN_DATASET["split"],
    )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME_FOR_TOKENIZER)
    out: List[str] = []
    for row in ds:
        text = row[BENIGN_DATASET["field"]].strip()
        if not text or text.startswith("="):
            continue
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        if min_tokens <= n_tok <= max_tokens:
            out.append(text)
        if len(out) >= n:
            break
    return out


def collect_harmful_direct() -> List[str]:
    """Collect direct harmful requests from AdvBench + HarmBench (deduplicated)."""
    out: List[str] = []
    for cfg in HARMFUL_REQUEST_DATASETS:
        try:
            ds = load_dataset(cfg["name"], cfg["config"], split=cfg["split"])
            n_before = len(out)
            for row in ds:
                val = row.get(cfg["field"])
                if val:
                    out.append(str(val).strip())
            print(f"    + {cfg['name']}: {len(out) - n_before} rows")
        except Exception as e:
            print(f"    ! Failed to load {cfg['name']}: {e}")
    return list(dict.fromkeys(out))  # dedupe, preserve order


def collect_jailbreak_wrapped(
    harmful_requests: List[str],
    n_target: int = 7000,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """Collect TrustAIRLab templates and combine with harmful requests.

    Strategy:
      - Standalone templates (no placeholder)  : use as-is, attack is embedded.
      - Placeholder templates                  : fill with random harmful requests
                                                 until n_target is reached.

    Returns a list of tuples: (final_prompt, template_snippet, request_used).
    """
    try:
        ds = load_dataset(
            JAILBREAK_TEMPLATE_DATASET["name"],
            JAILBREAK_TEMPLATE_DATASET["config"],
            split=JAILBREAK_TEMPLATE_DATASET["split"],
        )
    except Exception as e:
        print(f"    ! Failed to load TrustAIRLab: {e}")
        return []

    templates = [
        row[JAILBREAK_TEMPLATE_DATASET["field"]]
        for row in ds
        if row.get(JAILBREAK_TEMPLATE_DATASET["field"])
    ]
    templates = list(dict.fromkeys(templates))  # dedupe

    standalone = [t for t in templates if _find_placeholder(t) is None]
    placeholder = [t for t in templates if _find_placeholder(t) is not None]

    print(f"    + Standalone templates (use as-is):      {len(standalone)}")
    print(f"    + Placeholder templates (fill w/ harm):  {len(placeholder)}")

    rng = random.Random(seed)
    out: List[Tuple[str, str, str]] = []

    # (1) Every standalone template contributes one prompt
    for t in standalone:
        snip = t[:80] + ("..." if len(t) > 80 else "")
        out.append((t, snip, "<embedded>"))

    # (2) Fill placeholder templates up to the target budget
    remaining = n_target - len(out)
    if remaining > 0 and placeholder and harmful_requests:
        for _ in range(remaining):
            t = rng.choice(placeholder)
            r = rng.choice(harmful_requests)
            final = fill_template(t, r)
            snip = t[:80] + ("..." if len(t) > 80 else "")
            out.append((final, snip, r))

    return out


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect prompts for the jailbreak activation mapping pipeline."
    )
    parser.add_argument("--n-benign", type=int, default=4000,
                        help="Number of benign WikiText passages.")
    parser.add_argument("--n-jailbreak-wrapped", type=int, default=7000,
                        help="Target total for jailbreak_wrapped category.")
    parser.add_argument("--output", type=str, default="artifacts/prompts/prompt_pool.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("MODULE 1 — DATA COLLECTION")
    print("=" * 60)

    if not os.getenv("HF_TOKEN"):
        print("[!] HF_TOKEN not set — some gated datasets may fail.")

    print(f"\n[1/3] Benign passages (WikiText-103), target={args.n_benign}")
    benign = collect_benign(args.n_benign)
    print(f"    * Collected {len(benign)} benign passages")

    print("\n[2/3] Direct harmful requests (AdvBench + HarmBench)")
    harmful = collect_harmful_direct()
    print(f"    * Collected {len(harmful)} harmful requests (deduplicated)")

    print(f"\n[3/3] Jailbreak templates (TrustAIRLab), target={args.n_jailbreak_wrapped}")
    jb_wrapped = collect_jailbreak_wrapped(
        harmful, n_target=args.n_jailbreak_wrapped, seed=args.seed,
    )
    print(f"    * Built {len(jb_wrapped)} wrapped prompts")

    # Assemble pool
    prompts: List[str] = []
    sources: List[str] = []
    categories: List[str] = []

    for p in benign:
        prompts.append(p)
        sources.append("wikitext-103")
        categories.append("benign")

    for p in harmful:
        prompts.append(p)
        sources.append("advbench+harmbench")
        categories.append("harmful_direct")

    for final, _snip, _req in jb_wrapped:
        prompts.append(final)
        sources.append("trustairlab")
        categories.append("jailbreak_wrapped")

    # Shuffle so downstream cap / splits are class-balanced
    rng = random.Random(args.seed)
    idx = list(range(len(prompts)))
    rng.shuffle(idx)
    prompts = [prompts[i] for i in idx]
    sources = [sources[i] for i in idx]
    categories = [categories[i] for i in idx]

    counts = {
        "benign": sum(1 for c in categories if c == "benign"),
        "harmful_direct": sum(1 for c in categories if c == "harmful_direct"),
        "jailbreak_wrapped": sum(1 for c in categories if c == "jailbreak_wrapped"),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prompts": prompts,
        "sources": sources,
        "categories": categories,
        "counts": counts,
        "n_total": len(prompts),
    }, out_path)

    print("\n" + "=" * 60)
    print(f"DONE — saved {len(prompts)} prompts to {out_path}")
    for cat, n in counts.items():
        print(f"    {cat:<20s}: {n}")
    print("=" * 60)


if __name__ == "__main__":
    main()

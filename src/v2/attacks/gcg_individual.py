"""
GCG-Individual attack (Zou et al., 2023) — target-agnostic.

Optimizes a SEPARATE adversarial suffix per seed prompt. Far higher fidelity
than the universal variant — each suffix is tailored to bypass the target
model on that specific request — but ~100x more compute. We cap to
N_GCG_INDIVIDUAL=200 randomly-sampled seeds to keep this tractable.

Cost per seed (250 steps, search_width=512):
  Gemma-2-2b-it on H100:  ~3 min/seed
  Vicuna-7B-v1.3 on H100: ~9-12 min/seed
Total for 20 seeds (Plan B-tight): ~1-4 GPU-hours.

The target model is selected via --target (gemma | vicuna). For Llama-1-
derived tokenizers (Vicuna) the chat_template is set programmatically
before nanogcg.run.

Output:
  <gcg_individual_dir>/results.jsonl   ← one row per seed

Each row in results.jsonl:
  {seed_prompt, suffix, attacked_prompt, best_loss, num_steps, elapsed_sec}

Resume:
  Re-run with --resume; rows already in results.jsonl are skipped.

Usage:
  # Gemma
  python -m src.v2.attacks.gcg_individual --target gemma \
      --seeds artifacts/v2/gemma/prompts/attack_seeds.pt \
      --n-seeds 20 --num-steps 250 --resume

  # Vicuna
  python -m src.v2.attacks.gcg_individual --target vicuna \
      --seeds artifacts/v2/vicuna/prompts/attack_seeds.pt \
      --n-seeds 20 --num-steps 250 --resume
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from src.v2.config import (
    DEFAULT_TARGET,
    N_GCG_INDIVIDUAL,
    SEED,
    get_config,
    list_targets,
)


# Affirmative target string used by the GCG objective. Suffixed _STRING to
# disambiguate from src.v2.config.DEFAULT_TARGET (the model-selection key).
DEFAULT_TARGET_STRING = "Sure, here is"


@dataclass
class GCGIndividualResult:
    seed_prompt: str
    suffix: str
    attacked_prompt: str
    best_loss: float
    num_steps: int
    elapsed_sec: float


def _load_seeds(seeds_path: Path) -> list[str]:
    import torch
    obj = torch.load(seeds_path, weights_only=False)
    if isinstance(obj, dict) and "seeds" in obj:
        return list(obj["seeds"])
    if isinstance(obj, list):
        return list(obj)
    raise ValueError(f"Unexpected seeds file format at {seeds_path}: {type(obj)}")


def _load_existing(results_path: Path) -> set[str]:
    """Return the set of seed_prompts already attacked (for --resume)."""
    if not results_path.exists():
        return set()
    seen: set[str] = set()
    with open(results_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                seen.add(row["seed_prompt"])
            except Exception:
                continue
    return seen


def attack_one(
    model,
    tokenizer,
    seed_prompt: str,
    target: str,
    num_steps: int,
    search_width: int,
    topk: int,
    seed: int,
) -> GCGIndividualResult:
    """Optimize a per-prompt suffix for a single seed."""
    import nanogcg
    from nanogcg import GCGConfig

    config = GCGConfig(
        num_steps=num_steps,
        search_width=search_width,
        topk=topk,
        seed=seed,
        verbosity="WARNING",
    )

    t0 = time.time()
    result = nanogcg.run(
        model=model,
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": seed_prompt}],
        target=target,
        config=config,
    )
    elapsed = time.time() - t0
    suffix = result.best_string
    return GCGIndividualResult(
        seed_prompt=seed_prompt,
        suffix=suffix,
        attacked_prompt=f"{seed_prompt.rstrip()} {suffix}".strip(),
        best_loss=float(result.best_loss),
        num_steps=num_steps,
        elapsed_sec=elapsed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Per-prompt GCG suffix optimization against the chosen "
                    "target model (gemma | vicuna).",
    )
    parser.add_argument(
        "--target", type=str, default=DEFAULT_TARGET, choices=list_targets(),
        help=f"Target model to attack. Default: {DEFAULT_TARGET!r}. "
             f"Available: {list_targets()}.",
    )
    parser.add_argument(
        "--seeds", type=str, required=True,
        help="Path to a torch-saved file with harmful seed prompts.",
    )
    parser.add_argument("--n-seeds",     type=int, default=N_GCG_INDIVIDUAL,
                        help="How many seeds to attack (default: 200).")
    parser.add_argument("--num-steps",   type=int, default=500)
    parser.add_argument("--search-width",type=int, default=512)
    parser.add_argument("--topk",        type=int, default=256)
    parser.add_argument("--affirmative", type=str, default=DEFAULT_TARGET_STRING,
                        help="Affirmative response prefix the optimizer "
                             "pushes the model toward.")
    parser.add_argument("--seed",        type=int, default=SEED)
    parser.add_argument("--resume",      action="store_true",
                        help="Skip seeds already present in results.jsonl.")
    parser.add_argument("--out-dir",     type=str, default=None,
                        help="Override output directory; defaults to the "
                             "target's gcg_individual_dir from config.")
    args = parser.parse_args()

    cfg = get_config(args.target)
    out_dir = Path(args.out_dir or cfg["gcg_individual_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    # ── Seed selection (deterministic) ────────────────────────────────────
    all_seeds = _load_seeds(Path(args.seeds))
    rng = random.Random(args.seed)
    sampled = list(all_seeds)
    rng.shuffle(sampled)
    sampled = sampled[: args.n_seeds]
    print(f"[gcg-individual] Target: {args.target} ({cfg['model_name']})")
    print(f"[gcg-individual] Sampled {len(sampled)} seeds "
          f"from {len(all_seeds)} (seed={args.seed}).")

    already_done = _load_existing(results_path) if args.resume else set()
    todo = [s for s in sampled if s not in already_done]
    print(f"[gcg-individual] Already attacked: {len(already_done)}. "
          f"To do: {len(todo)}.")
    if not todo:
        print("[gcg-individual] Nothing to do. Exit.")
        return

    # ── Load target model once ────────────────────────────────────────────
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GCG-Individual requires CUDA — gradient computation through "
            "the target model on CPU is intractable. Run on a GPU instance."
        )

    model_name = cfg["model_name"]
    chat_template = cfg["chat_template"]

    print(f"[gcg-individual] Loading {model_name} on CUDA (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Llama-1-derived tokenizers ship without a chat_template; nanogcg's
    # apply_chat_template would fail. Set it programmatically.
    if chat_template is not None:
        tokenizer.chat_template = chat_template
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "</s>"
        print(f"[gcg-individual] Set chat_template (len={len(chat_template)})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # ── Attack loop with append-only checkpointing ────────────────────────
    t_total = time.time()
    n_done_this_run = 0
    with open(results_path, "a") as fout:
        for i, sp in enumerate(todo):
            try:
                r = attack_one(
                    model=model,
                    tokenizer=tokenizer,
                    seed_prompt=sp,
                    target=args.affirmative,
                    num_steps=args.num_steps,
                    search_width=args.search_width,
                    topk=args.topk,
                    seed=args.seed + i,  # vary so different seeds explore differently
                )
                fout.write(json.dumps(asdict(r)) + "\n")
                fout.flush()
                n_done_this_run += 1
                eta_min = (time.time() - t_total) / max(n_done_this_run, 1) \
                          * (len(todo) - n_done_this_run) / 60
                print(f"[gcg-individual] {n_done_this_run}/{len(todo)}  "
                      f"loss={r.best_loss:.3f}  "
                      f"({r.elapsed_sec/60:.1f} min)  "
                      f"ETA {eta_min:.0f} min")
            except Exception as e:
                print(f"[gcg-individual] FAIL on seed {i}: {e}")
                continue

    print(f"[gcg-individual] Done. Total: {n_done_this_run} attacks in "
          f"{(time.time()-t_total)/60:.1f} min.")
    print(f"[gcg-individual] Results: {results_path}")


if __name__ == "__main__":
    main()

"""
GCG-Universal attack (Zou et al., 2023) — target-agnostic.

Optimizes ONE adversarial suffix against an anchor harmful seed, then appends
that suffix to every harmful seed in the full pool. The cost is dominated by
the single optimization run (~3-6 GPU-hours on H100 for Gemma-2-2b-it, ~9-12
hours for Vicuna-7B-v1.3); the "apply" step is just string concatenation.

The target model is selected via --target (gemma | vicuna). For Llama-1-
derived tokenizers (Vicuna) the chat_template is set programmatically before
nanogcg.run, because those tokenizers ship without one.

Output (default paths per target):
  <gcg_universal_dir>/suffix.txt       ← the optimized suffix
  <gcg_universal_dir>/results.jsonl    ← per-seed attacked prompts
  <gcg_universal_dir>/meta.json        ← optimization metadata

Usage:
  # Gemma
  python -m src.v2.attacks.gcg_universal --target gemma \
      --seeds artifacts/v2/gemma/prompts/attack_seeds.pt \
      --num-train 25 --num-steps 500

  # Vicuna
  python -m src.v2.attacks.gcg_universal --target vicuna \
      --seeds artifacts/v2/vicuna/prompts/attack_seeds.pt \
      --num-train 25 --num-steps 500
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from src.v2.config import DEFAULT_TARGET, SEED, get_config, list_targets


# Affirmative target string used by the GCG objective — the optimizer
# maximizes the model's probability of beginning its response with this
# prefix (i.e. NOT a refusal). Named with a _STRING suffix to disambiguate
# from src.v2.config.DEFAULT_TARGET, which is the model-selection key.
DEFAULT_TARGET_STRING = "Sure, here is"


@dataclass
class GCGUniversalResult:
    """One attacked prompt produced by appending the universal suffix."""
    seed_prompt: str
    suffix: str
    attacked_prompt: str   # seed_prompt + " " + suffix


def _load_seeds(seeds_path: Path) -> list[str]:
    """Load a list of harmful seed prompts written by module1_v2_data_collection."""
    import torch
    if not seeds_path.exists():
        raise FileNotFoundError(
            f"Seed file {seeds_path} not found. Run module1_v2_data_collection.py "
            f"first (or pass --seeds-from-text)."
        )
    obj = torch.load(seeds_path, weights_only=False)
    if isinstance(obj, dict) and "seeds" in obj:
        return list(obj["seeds"])
    if isinstance(obj, list):
        return list(obj)
    raise ValueError(f"Unexpected seeds file format at {seeds_path}: {type(obj)}")


def optimize_suffix(
    train_prompts: list[str],
    model_name: str,
    chat_template: Optional[str] = None,
    target: str = DEFAULT_TARGET_STRING,
    num_steps: int = 500,
    search_width: int = 512,
    topk: int = 256,
    seed: int = SEED,
    device: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Run nanoGCG in multi-prompt (universal) mode against Gemma-2-2b-it.

    Args:
        train_prompts: small batch of harmful prompts to co-optimize against.
            More prompts = more transferable suffix but slower per step.
            Recommend 20-50.
        target: affirmative response prefix the optimizer pushes the model toward.
        num_steps: GCG optimization steps.
        search_width: number of candidate token swaps evaluated per step.
        topk: number of top-gradient tokens considered per position.

    Returns:
        (suffix_string, metadata_dict)
    """
    try:
        import nanogcg
        from nanogcg import GCGConfig
    except ImportError as e:
        raise ImportError(
            "GCG-Universal requires nanogcg. Install with: pip install nanogcg"
        ) from e
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError(
            "GCG requires CUDA — gradient computation through the target "
            "model on CPU is intractable. Run this on a GPU instance."
        )

    print(f"[gcg-universal] Loading {model_name} on {device} (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Llama-1-derived tokenizers (Vicuna) ship without a chat_template;
    # nanogcg.apply_chat_template would fail. Set it programmatically.
    if chat_template is not None:
        tokenizer.chat_template = chat_template
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "</s>"
        print(f"[gcg-universal] Set chat_template (len={len(chat_template)})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # nanogcg.run() takes a single conversation (list of dicts), not a batch
    # of conversations. True multi-prompt universal mode lives in
    # nanogcg.MultiPromptGCG, which has a different API and isn't present in
    # every nanogcg release. We pick the longest training prompt as the
    # optimization anchor — the resulting suffix transfers reasonably across
    # similar AdvBench/HarmBench prompts because they share surface form.
    anchor_prompt = max(train_prompts, key=len)
    messages = [{"role": "user", "content": anchor_prompt}]

    config = GCGConfig(
        num_steps=num_steps,
        search_width=search_width,
        topk=topk,
        seed=seed,
        verbosity="WARNING",
    )

    t0 = time.time()
    print(f"[gcg-universal] Optimizing suffix ({num_steps} steps) "
          f"with anchor: {anchor_prompt[:80]!r}...")
    result = nanogcg.run(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        target=target,
        config=config,
    )
    elapsed = time.time() - t0
    print(f"[gcg-universal] Done in {elapsed/60:.1f} min. "
          f"Best loss: {result.best_loss:.4f}")

    suffix: str = result.best_string
    meta = {
        "model": model_name,
        "target": target,
        "anchor_prompt": anchor_prompt,
        "num_train_prompts_pool": len(train_prompts),
        "num_steps": num_steps,
        "search_width": search_width,
        "topk": topk,
        "seed": seed,
        "best_loss": float(result.best_loss),
        "elapsed_sec": elapsed,
    }
    return suffix, meta


def apply_suffix(seed_prompts: list[str], suffix: str) -> list[GCGUniversalResult]:
    """Append the optimized suffix to every seed prompt."""
    out: list[GCGUniversalResult] = []
    for p in seed_prompts:
        attacked = f"{p.rstrip()} {suffix}".strip()
        out.append(GCGUniversalResult(seed_prompt=p, suffix=suffix, attacked_prompt=attacked))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Optimize one universal GCG suffix against the chosen "
                    "target model (gemma | vicuna).",
    )
    parser.add_argument(
        "--target", type=str, default=DEFAULT_TARGET, choices=list_targets(),
        help=f"Target model to attack. Default: {DEFAULT_TARGET!r}. "
             f"Available: {list_targets()}.",
    )
    parser.add_argument(
        "--seeds", type=str, required=True,
        help="Path to a torch-saved file containing the harmful seed prompts. "
             "Either a list[str] or a dict with key 'seeds'.",
    )
    parser.add_argument("--num-train",   type=int, default=25,
                        help="How many seeds to draw the anchor from (longest "
                             "is selected). 25 is a good default.")
    parser.add_argument("--num-steps",   type=int, default=500)
    parser.add_argument("--search-width",type=int, default=512)
    parser.add_argument("--topk",        type=int, default=256)
    parser.add_argument("--affirmative", type=str, default=DEFAULT_TARGET_STRING,
                        help="Affirmative response prefix the optimizer pushes "
                             "the model toward (default: 'Sure, here is').")
    parser.add_argument("--seed",        type=int, default=SEED)
    parser.add_argument("--out-dir",     type=str, default=None,
                        help="Override the default output directory; defaults "
                             "to the target's gcg_universal_dir from config.")
    parser.add_argument("--reuse-suffix", type=str, default=None,
                        help="Skip optimization and reuse a previously saved suffix.txt.")
    args = parser.parse_args()

    cfg = get_config(args.target)
    out_dir = Path(args.out_dir or cfg["gcg_universal_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _load_seeds(Path(args.seeds))
    print(f"[gcg-universal] Target: {args.target} ({cfg['model_name']})")
    print(f"[gcg-universal] Loaded {len(seeds)} seed prompts. Output: {out_dir}")

    if args.reuse_suffix:
        suffix = Path(args.reuse_suffix).read_text().strip()
        meta = {"reused_from": args.reuse_suffix, "model": cfg["model_name"]}
        print(f"[gcg-universal] Reusing suffix from {args.reuse_suffix}")
    else:
        rng = random.Random(args.seed)
        train = list(seeds)
        rng.shuffle(train)
        train = train[: args.num_train]
        suffix, meta = optimize_suffix(
            train_prompts=train,
            model_name=cfg["model_name"],
            chat_template=cfg["chat_template"],
            target=args.affirmative,
            num_steps=args.num_steps,
            search_width=args.search_width,
            topk=args.topk,
            seed=args.seed,
        )
        meta["target_key"] = args.target
        (out_dir / "suffix.txt").write_text(suffix)
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[gcg-universal] Saved suffix to {out_dir / 'suffix.txt'}")

    # Apply suffix to every seed.
    results = apply_suffix(seeds, suffix)
    with open(out_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"[gcg-universal] Wrote {len(results)} attacked prompts to "
          f"{out_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()

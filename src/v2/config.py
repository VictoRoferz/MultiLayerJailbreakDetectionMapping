"""
v2 dataset build invariants — parametrized by target model.

EVERY value inside a TARGETS entry MUST stay frozen for that target so that
v1 + v2 datasets can be merged row-wise via `concatenate_datasets` later
(within the same target). Cross-target merging is NOT supported — Gemma's
2304-dim activations cannot be concatenated with Vicuna's 4096-dim
activations.

Usage:
    from src.v2.config import get_config
    cfg = get_config("gemma")     # or "vicuna"
    print(cfg["model_name"], cfg["layers"], cfg["hf_repo"])

For backwards compatibility the module also exposes top-level constants
(MODEL_NAME, LAYERS, etc.) bound to the default target (gemma). New code
should prefer get_config(target).
"""

from typing import Final


# ── Per-target build configurations ──────────────────────────────────────────

# Each entry under TARGETS defines the merge-invariant constants plus the
# output paths for that target model's v2 dataset.

# FastChat-style Vicuna v1.x chat template. Vicuna-7B-v1.3 ships without a
# `chat_template` field on its tokenizer config; nanogcg's apply_chat_template
# call would fail without this. We set it programmatically in the GCG scripts.
_VICUNA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + ' ' }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ 'A chat between a curious user and an artificial intelligence "
    "assistant. The assistant gives helpful, detailed, and polite answers "
    "to the user\\'s questions. ' }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ 'USER: ' + message['content'] + ' ' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'ASSISTANT: ' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
)


TARGETS: Final[dict] = {
    # ─── Gemma-2-2b-it (default, primary target) ──────────────────────────
    "gemma": {
        "model_name":         "google/gemma-2-2b-it",
        "hidden_dim":         2304,
        "layers":             (5, 10, 15, 20, 25),
        "pool_k":             5,
        "max_prompt_tokens":  2048,
        "max_new_tokens":     200,
        "judge_model":        "gpt-4o-mini",
        "judge_tau":          7,
        "seed":               42,
        # Gemma's tokenizer ships its own chat_template — leave None to use
        # the built-in one when nanogcg calls apply_chat_template.
        "chat_template":      None,
        "artifact_dir":       "artifacts/v2/gemma",
        "prompt_pool":        "artifacts/v2/gemma/prompts/prompt_pool.pt",
        "seeds_out":          "artifacts/v2/gemma/prompts/attack_seeds.pt",
        "gcg_universal_dir":  "artifacts/v2/gemma/attacks/gcg_universal",
        "gcg_individual_dir": "artifacts/v2/gemma/attacks/gcg_individual",
        "hf_repo":            "victorroferz/gemma-2b-jailbreak-behavior-dataset-v2",
    },

    # ─── Vicuna-7B-v1.3 (Gao et al.'s primary backbone) ───────────────────
    "vicuna": {
        "model_name":         "lmsys/vicuna-7b-v1.3",
        "hidden_dim":         4096,
        # Gao et al. focus on layers 4 / 15 / 30 across Vicuna's 32 blocks.
        # We probe a wider set for parity with the Gemma run plus headline
        # comparison points.
        "layers":             (4, 10, 15, 20, 25, 30),
        "pool_k":             5,
        "max_prompt_tokens":  2048,
        "max_new_tokens":     200,
        "judge_model":        "gpt-4o-mini",
        "judge_tau":          7,
        "seed":               42,
        # Llama-1-derived tokenizers do not ship a chat_template. Set it
        # explicitly so nanogcg can render the prompt correctly.
        "chat_template":      _VICUNA_CHAT_TEMPLATE,
        "artifact_dir":       "artifacts/v2/vicuna",
        "prompt_pool":        "artifacts/v2/vicuna/prompts/prompt_pool.pt",
        "seeds_out":          "artifacts/v2/vicuna/prompts/attack_seeds.pt",
        "gcg_universal_dir":  "artifacts/v2/vicuna/attacks/gcg_universal",
        "gcg_individual_dir": "artifacts/v2/vicuna/attacks/gcg_individual",
        "hf_repo":            "victorroferz/vicuna-7b-jailbreak-behavior-dataset-v2",
    },
}

DEFAULT_TARGET: Final[str] = "gemma"


def get_config(target: str = DEFAULT_TARGET) -> dict:
    """
    Return the build configuration dict for the named target.

    Raises ValueError if `target` is not a known key in TARGETS.
    """
    if target not in TARGETS:
        raise ValueError(
            f"Unknown target '{target}'. "
            f"Available targets: {sorted(TARGETS.keys())}"
        )
    return TARGETS[target]


def list_targets() -> list:
    """List all registered target names — useful for CLI choices=."""
    return sorted(TARGETS.keys())


# ── v2-specific dataset decisions (shared across all targets) ────────────────
# These don't depend on target model — they describe the prompt pool itself.

ATTACK_SEED_DATASETS: Final[list[dict]] = [
    {"name": "walledai/AdvBench",  "config": None,       "split": "train", "field": "prompt"},
    {"name": "walledai/HarmBench", "config": "standard", "split": "train", "field": "prompt"},
]

BENIGN_DATASET: Final[dict] = {
    "name":   "tatsu-lab/alpaca",
    "config": None,
    "split":  "train",
    "field":  "instruction",
    "n":      4000,
}

# GCG-Individual default seed count — per-prompt optimization is expensive,
# so we cap to a budget-friendly count. Override per-run via --n-seeds.
N_GCG_INDIVIDUAL: Final[int] = 200

# Category labels (stable strings; downstream PCA / clustering / detector
# code keys off these — do not rename without a migration).
CATEGORY_BENIGN: Final[str]            = "benign"
CATEGORY_HARMFUL_DIRECT: Final[str]    = "harmful_direct"
CATEGORY_JB_ARTPROMPT: Final[str]      = "jailbreak_artprompt"
CATEGORY_JB_GCG_UNIVERSAL: Final[str]  = "jailbreak_gcg_universal"
CATEGORY_JB_GCG_INDIVIDUAL: Final[str] = "jailbreak_gcg_individual"


# ── Legacy single-target constants (point at DEFAULT_TARGET) ────────────────
# Kept for backwards-compat with scripts that import these directly.
# New code should call get_config(target) instead.

_DEFAULT_CFG = TARGETS[DEFAULT_TARGET]
MODEL_NAME: Final[str]        = _DEFAULT_CFG["model_name"]
LAYERS: Final[tuple]          = _DEFAULT_CFG["layers"]
POOL_K: Final[int]            = _DEFAULT_CFG["pool_k"]
HIDDEN_DIM: Final[int]        = _DEFAULT_CFG["hidden_dim"]
MAX_PROMPT_TOKENS: Final[int] = _DEFAULT_CFG["max_prompt_tokens"]
MAX_NEW_TOKENS: Final[int]    = _DEFAULT_CFG["max_new_tokens"]
JUDGE_MODEL: Final[str]       = _DEFAULT_CFG["judge_model"]
JUDGE_TAU: Final[int]         = _DEFAULT_CFG["judge_tau"]
SEED: Final[int]              = _DEFAULT_CFG["seed"]
V2_ARTIFACT_DIR: Final[str]   = _DEFAULT_CFG["artifact_dir"]
V2_PROMPT_POOL: Final[str]    = _DEFAULT_CFG["prompt_pool"]
V2_HF_REPO: Final[str]        = _DEFAULT_CFG["hf_repo"]

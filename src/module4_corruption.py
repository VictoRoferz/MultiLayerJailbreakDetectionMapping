#!/usr/bin/env python3
"""
module4_corruption.py

Module 4 — Corruption Loop (Paper Section 3.5)

For each benign passage x in the test set:
    1. Extract f_L(x) at target layer L
    2. Sample K perturbation vectors delta_f from trained CVAE/MLP generator
    3. Apply norm constraint: ||delta_f|| <= epsilon * ||f_L||
    4. Inject f'_L = f_L + delta_f via forward hook, run generation
    5. Collect (delta_f, passage_text, generated_output, f_L) triples

These triples are then scored by Module 5 (judge) to identify successful
jailbreak perturbations.

Usage:
    python src/module4_corruption.py --layer 20 --architecture cvae --K 500 --epsilon 0.1
    python src/module4_corruption.py --layer 20 --K 10 --max-passages 50   # quick test
    python src/module4_corruption.py --layer all --K 100
"""

import argparse
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from module3_perturbation_generator import (
    PerturbationGenerator,
    CVAEPerturbationGenerator,
    FlowMatchingDenoiser,
    apply_norm_constraint,
    apply_denoiser,
    get_device,
    DEFAULT_LAYERS,
    REFUSAL_PHRASES,
    MODEL_NAME,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 1: Model & Hook Utilities                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def get_target_module(model, layer_idx: int):
    """Get the transformer layer module for hook registration."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    elif hasattr(model, "layers"):
        return model.layers[layer_idx]
    raise ValueError("Could not locate layers in model structure.")


def extract_activation(model, tokenizer, text: str, layer_idx: int,
                       device: torch.device, k: int = 5) -> torch.Tensor:
    """
    Extract activation f_L(x) for a single passage.
    Returns [D] tensor (mean of last k token positions at layer L).
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    all_states = outputs.hidden_states
    if layer_idx >= len(all_states):
        raise ValueError(f"Layer {layer_idx} out of bounds (max {len(all_states)-1})")

    seq_len = all_states[layer_idx].shape[1]
    actual_k = min(k, seq_len)
    f_L = all_states[layer_idx][0, -actual_k:, :].mean(dim=0)  # [D]
    return f_L.to(torch.float32)


def generate_with_hook(model, tokenizer, inputs, layer_idx: int,
                       delta_f: torch.Tensor, k: int = 5,
                       max_new_tokens: int = 256) -> str:
    """
    Run model.generate() with delta_f injected at layer L via hook.
    Returns generated text string.
    """
    def make_hook(delta, k_positions):
        def hook_fn(module, args, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            hidden[:, -k_positions:, :] = (
                hidden[:, -k_positions:, :]
                + delta.view(1, 1, -1).to(hidden.dtype)
            )
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        return hook_fn

    layer_module = get_target_module(model, layer_idx)
    hook_handle = layer_module.register_forward_hook(make_hook(delta_f, k))

    try:
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    finally:
        hook_handle.remove()

    return response


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 2: Generator Loading                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_generator(layer_idx: int, architecture: str, device: torch.device):
    """Load trained generator from artifacts. Returns (generator, checkpoint_dict)."""
    if architecture == "cvae":
        base = Path("artifacts") / f"layer_{layer_idx}" / "cvae"
    else:
        base = Path("artifacts") / f"layer_{layer_idx}"

    ckpt_path = base / "generator.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No generator at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if architecture == "cvae":
        gen = CVAEPerturbationGenerator(ckpt["activation_dim"], z_dim=ckpt["z_dim"])
    else:
        gen = PerturbationGenerator(ckpt["activation_dim"], z_dim=ckpt["z_dim"])

    gen.load_state_dict(ckpt["model_state_dict"])
    gen.to(device)
    gen.eval()
    return gen, ckpt


def load_all_generators(layer_idx: int, architecture: str, device: torch.device):
    """Load base generator + any Frank-Wolfe generators. Returns list of generators."""
    gen, ckpt = load_generator(layer_idx, architecture, device)
    generators = [gen]

    if architecture == "cvae":
        base = Path("artifacts") / f"layer_{layer_idx}" / "cvae"
    else:
        base = Path("artifacts") / f"layer_{layer_idx}"

    # Load FW generators if they exist
    for i in range(10):  # up to 10 FW generators
        fw_path = base / f"generator_fw_{i}.pt"
        if not fw_path.exists():
            break
        fw_ckpt = torch.load(fw_path, map_location=device, weights_only=False)
        if architecture == "cvae":
            fw_gen = CVAEPerturbationGenerator(fw_ckpt["activation_dim"], z_dim=fw_ckpt["z_dim"])
        else:
            fw_gen = PerturbationGenerator(fw_ckpt["activation_dim"], z_dim=fw_ckpt["z_dim"])
        fw_gen.load_state_dict(fw_ckpt["model_state_dict"])
        fw_gen.to(device)
        fw_gen.eval()
        generators.append(fw_gen)

    return generators, ckpt


def load_denoiser(layer_idx: int, architecture: str, device: torch.device):
    """Load denoiser if it exists. Returns None otherwise."""
    if architecture == "cvae":
        base = Path("artifacts") / f"layer_{layer_idx}" / "cvae"
    else:
        base = Path("artifacts") / f"layer_{layer_idx}"

    path = base / "denoiser.pt"
    if not path.exists():
        return None

    ckpt = torch.load(path, map_location=device, weights_only=False)
    denoiser = FlowMatchingDenoiser(ckpt["activation_dim"])
    denoiser.load_state_dict(ckpt["model_state_dict"])
    denoiser.to(device)
    denoiser.eval()
    return denoiser


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 3: Passage Loading                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_test_passages(max_passages: int = None) -> list:
    """
    Load benign test passages for corruption.

    Priority:
    1. artifacts/test_passages.pt (if proper splits were created)
    2. WikiText-103 (on-the-fly loading, filtered to 64-256 tokens)
    """
    # Try pre-saved test passages
    test_path = Path("artifacts") / "test_passages.pt"
    if test_path.exists():
        passages = torch.load(test_path, weights_only=False)
        if isinstance(passages, list):
            if max_passages:
                passages = passages[:max_passages]
            print(f"  Loaded {len(passages)} test passages from {test_path}")
            return passages

    # Fallback: load from WikiText-103
    print("  Loading passages from WikiText-103...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    passages = []
    for item in ds:
        text = (item.get("text") or "").strip()
        if 50 < len(text) < 2000:  # reasonable length
            passages.append(text)
        if max_passages and len(passages) >= max_passages:
            break

    # Default: use last 20% as test
    if not max_passages:
        n_test = max(100, len(passages) // 5)
        passages = passages[-n_test:]  # last 20%

    print(f"  Using {len(passages)} passages for corruption")
    return passages


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 4: Corruption Loop (Algorithm 1 from paper)                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_corruption_loop(
    layer_idx: int,
    architecture: str = "cvae",
    K: int = 500,
    epsilon: float = 0.1,
    max_passages: int = None,
    max_new_tokens: int = 256,
    denoiser_steps: int = 20,
    denoiser_t_start: float = 0.3,
    save_outputs: bool = True,
) -> dict:
    """
    Algorithm 1 from the paper: Corruption Loop.

    For each passage x in the test set:
        For k = 1 to K:
            1. Sample z ~ N(0, I)
            2. delta_f = generator.decode(z, c=1) [for CVAE] or generator(z, f_L) [for MLP]
            3. delta_f = epsilon * delta_f * ||f_L|| / ||delta_f||  (norm constraint)
            4. y_tilde = model(x; f_L + delta_f)  (inject via hook)
            5. Collect (delta_f, score, x) for later judging

    Returns dict with corruption results.
    """
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Module 4: Corruption Loop — Layer {layer_idx} ({architecture})")
    print(f"  K={K} perturbations/passage, epsilon={epsilon}")
    print(f"{'='*60}")

    # Load model
    hf_token = os.environ.get("HF_TOKEN")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n  Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        output_hidden_states=True,
        token=hf_token,
    )
    model.eval()

    # Load generators (base + Frank-Wolfe ensemble)
    generators, ckpt = load_all_generators(layer_idx, architecture, device)
    print(f"  Loaded {len(generators)} generator(s): {architecture} (z_dim={generators[0].z_dim})")

    # Use epsilon from checkpoint if available (ensures training/corruption consistency)
    if epsilon is None or "epsilon" in ckpt:
        ckpt_eps = ckpt.get("epsilon")
        if ckpt_eps is not None and ckpt_eps != epsilon:
            print(f"  [NOTE] Using epsilon={ckpt_eps} from checkpoint (CLI had {epsilon})")
            epsilon = ckpt_eps

    # Load optional denoiser
    denoiser = load_denoiser(layer_idx, architecture, device)
    if denoiser:
        print(f"  Denoiser loaded (t_start={denoiser_t_start})")

    # Load test passages
    passages = load_test_passages(max_passages)

    # ── Main loop ─────────────────────────────────────────────────────────
    all_results = []
    total_generated = 0
    n_gens = len(generators)

    for p_idx, text in enumerate(tqdm(passages, desc="Passages")):
        # Extract clean activation
        try:
            f_L = extract_activation(model, tokenizer, text, layer_idx, device)
        except Exception as e:
            print(f"    [SKIP] Passage {p_idx}: {e}")
            continue

        # Tokenize once for generation
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(device)

        seq_len = inputs["input_ids"].shape[1]
        k_positions = min(5, seq_len)

        for k_idx in range(K):
            # Round-robin across generators for diversity
            gen = generators[k_idx % n_gens]

            # Sample and generate perturbation
            z = torch.randn(1, gen.z_dim, device=device)
            with torch.no_grad():
                delta_f = gen(z, f_L.unsqueeze(0))
            delta_f = apply_norm_constraint(delta_f, f_L.unsqueeze(0), epsilon)

            # Optional denoising
            if denoiser is not None:
                delta_f = apply_denoiser(
                    delta_f, f_L.unsqueeze(0), denoiser,
                    n_steps=denoiser_steps, t_start=denoiser_t_start,
                )
                delta_f = apply_norm_constraint(delta_f, f_L.unsqueeze(0), epsilon)

            delta_f_squeezed = delta_f.squeeze(0)  # [D]

            # Inject and generate
            response = generate_with_hook(
                model, tokenizer, inputs, layer_idx,
                delta_f_squeezed, k=k_positions,
                max_new_tokens=max_new_tokens,
            )

            all_results.append({
                "passage_idx": p_idx,
                "perturbation_idx": k_idx,
                "text": text[:500],
                "response": response[:1000],
                "delta_f": delta_f_squeezed.cpu(),
                "f_L": f_L.cpu(),
                "delta_norm": delta_f_squeezed.norm().item(),
                "f_L_norm": f_L.norm().item(),
            })
            total_generated += 1

        # Progress update every 10 passages
        if (p_idx + 1) % 10 == 0:
            print(f"    Processed {p_idx + 1}/{len(passages)} passages "
                  f"({total_generated} total generations)")

    print(f"\n  Corruption loop complete: {total_generated} generations "
          f"from {len(passages)} passages")

    # ── Save results ──────────────────────────────────────────────────────
    if save_outputs:
        out_dir = Path("artifacts") / f"layer_{layer_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "corruption_results.pt"

        # Separate tensors from metadata for efficient storage
        save_data = {
            "delta_f_list": [r["delta_f"] for r in all_results],
            "f_L_list": [r["f_L"] for r in all_results],
            "metadata": [{
                "passage_idx": r["passage_idx"],
                "perturbation_idx": r["perturbation_idx"],
                "text": r["text"],
                "response": r["response"],
                "delta_norm": r["delta_norm"],
                "f_L_norm": r["f_L_norm"],
            } for r in all_results],
            "config": {
                "layer": layer_idx,
                "architecture": architecture,
                "K": K,
                "epsilon": epsilon,
                "n_passages": len(passages),
                "total_generations": total_generated,
            },
        }
        torch.save(save_data, out_path)
        print(f"  Saved corruption results to {out_path}")

    return {
        "layer": layer_idx,
        "n_passages": len(passages),
        "K": K,
        "total_generations": total_generated,
        "results": all_results,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Section 5: Main + CLI                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main(args):
    run_corruption_loop(
        layer_idx=args.layer_idx,
        architecture=args.architecture,
        K=args.K,
        epsilon=args.epsilon,
        max_passages=args.max_passages,
        max_new_tokens=args.max_new_tokens,
        denoiser_steps=args.denoiser_steps,
        denoiser_t_start=args.denoiser_t_start,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 4: Corruption Loop (Paper Section 3.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/module4_corruption.py --layer 20 --architecture cvae --K 500 --epsilon 0.1
  python src/module4_corruption.py --layer 20 --K 10 --max-passages 5    # quick test
  python src/module4_corruption.py --layer all --K 100
        """
    )

    parser.add_argument("--layer", type=str, default="20",
                        help="Layer index or 'all'")
    parser.add_argument("--architecture", type=str, default="cvae",
                        choices=["mlp", "cvae"])
    parser.add_argument("--K", type=int, default=500,
                        help="Number of perturbation samples per passage")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Norm constraint: ||delta_f|| <= eps * ||f_L||")
    parser.add_argument("--max-passages", type=int, default=None,
                        help="Max passages to process (None = all test passages)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--denoiser-steps", type=int, default=20)
    parser.add_argument("--denoiser-t-start", type=float, default=0.3)

    args = parser.parse_args()

    if args.layer.lower() == "all":
        for layer_idx in DEFAULT_LAYERS:
            print(f"\n{'#'*60}")
            print(f"  LAYER {layer_idx}")
            print(f"{'#'*60}")
            args.layer_idx = layer_idx
            main(args)
    else:
        args.layer_idx = int(args.layer)
        main(args)

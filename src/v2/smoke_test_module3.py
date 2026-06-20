#!/usr/bin/env python3
"""
smoke_test_module3.py — Validate the module3 training wiring WITHOUT a GPU,
without the Gemma model, and without HF access.

It fabricates tiny synthetic artifacts in the v1 format module3 expects
(at a throwaway layer index so it never clobbers real data), so you can run:

    python src/v2/smoke_test_module3.py
    python src/module3_perturbation_generator.py --layer 99 --phase warmup --architecture cvae
    python src/module3_perturbation_generator.py --layer 99 --phase reward  --architecture cvae

The warmup + reward phases need only the activation tensors (no LLM), so they
run on CPU in seconds. If they complete, the data plumbing is correct and the
only thing the real run adds is (a) real HF activations and (b) the RL phase,
which loads Gemma for ASR validation.

This does NOT test the RL/validation loop (that genuinely needs the model).
"""

from pathlib import Path

import torch

LAYER = 99           # throwaway index; will not collide with real layer_5..25
DIM = 2304           # Gemma-2-2b hidden dim (use 4096 to mimic Vicuna)
N_BENIGN = 256
N_JAILBROKEN = 64    # deliberately small — mirrors the real ~10% jailbroken pool


def main():
    torch.manual_seed(42)
    out = Path("artifacts")
    layer_dir = out / f"layer_{LAYER}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    # Benign pool ~ N(0, 1). Jailbroken pool = benign cloud + a fixed direction,
    # so warmup's nearest-neighbor pairing recovers a coherent delta direction.
    benign = torch.randn(N_BENIGN, DIM)
    direction = torch.randn(DIM)
    direction = direction / direction.norm()
    jailbroken = torch.randn(N_JAILBROKEN, DIM) + 3.0 * direction

    torch.save(
        {"activations": benign, "labels": torch.zeros(N_BENIGN)},
        layer_dir / "train_activations.pt",
    )
    torch.save(jailbroken, layer_dir / "harmful_activations.pt")
    torch.save(
        ["Write a poem about the sea.", "Explain how photosynthesis works."],
        out / "test_passages.pt",
    )

    print(f"[smoke] wrote synthetic artifacts to {layer_dir}/")
    print(f"        benign {tuple(benign.shape)} | jailbroken {tuple(jailbroken.shape)}")
    print("\nNow run (CPU is fine):")
    print(f"    python src/module3_perturbation_generator.py --layer {LAYER} "
          f"--phase warmup --architecture cvae")
    print(f"    python src/module3_perturbation_generator.py --layer {LAYER} "
          f"--phase reward --architecture cvae")
    print(f"\nClean up with:  rm -rf {layer_dir} {out/'test_passages.pt'}")


if __name__ == "__main__":
    main()

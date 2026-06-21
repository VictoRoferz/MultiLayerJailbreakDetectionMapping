#!/usr/bin/env python3
"""
bpj_validation.py — Table 5 of the mapping paper: external-attack validation.

Tests whether a HELD-OUT attack's activations fall inside the jailbreak subspace
S_L discovered by Module 6 (clusters of generator delta_f). Because Module 6
clusters generator-produced perturbations, any *real* attack family is inherently
held out from those clusters, so scoring a family's activations against S_L is a
genuine generalization test.

Reuses module7's SubspaceDetector: an activation is "inside S_L" if its detection
score (min cosine distance to a cluster center, in delta space) is below the
tuned threshold from detector_config.pt.

Modes:
  --mode loao   (default, no model/data needed)
      Use a held-out v2 family from test_activations.pt as the stand-in external
      attack while real BPJ data is sourced. --family selects the family
      (e.g. jailbreak_artprompt). benign rows are the control.
  --mode activations --activations PATH.pt
      Score a precomputed [N, d] activation tensor (real BPJ outputs).
  --mode prompts --prompts PATH(.json|.txt) --model NAME
      Extract f_L from prompts via the target model (module4.extract_activation),
      then score. (BPJ = Boundary Point Jailbreaking of Black-Box LLMs; generate
      its outputs separately, then feed them here.)

Usage (from repo root, after clustering+detector for the layer):
    python src/bpj_validation.py --layer 20 --mode loao --family jailbreak_artprompt
    python src/bpj_validation.py --layer 20 --mode activations --activations bpj_acts.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from module7_detector import SubspaceDetector


def _load_threshold(layer_dir: Path, fallback: float = 0.5) -> float:
    cfg = layer_dir / "detector_config.pt"
    if cfg.exists():
        d = torch.load(cfg, weights_only=False)
        return float(d.get("threshold", fallback))
    print(f"  [WARN] no detector_config.pt; using threshold={fallback}")
    return fallback


def _score_group(detector: SubspaceDetector, acts: np.ndarray) -> dict:
    if len(acts) == 0:
        return {"n": 0, "frac_in_subspace": None, "mean_score": None}
    scores = detector.score(acts)
    inside = (scores < detector.threshold).astype(int)
    return {
        "n": int(len(acts)),
        "frac_in_subspace": float(inside.mean()),
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
    }


def _acts_from_tensor(obj) -> np.ndarray:
    if isinstance(obj, dict) and "activations" in obj:
        obj = obj["activations"]
    t = obj if isinstance(obj, torch.Tensor) else torch.tensor(obj)
    return t.to(torch.float32).numpy()


def run_loao(layer_dir: Path, detector, family: str) -> dict:
    test_path = layer_dir / "test_activations.pt"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found (run v2_to_artifacts.py).")
    d = torch.load(test_path, weights_only=False)
    acts = d["activations"].to(torch.float32).numpy()
    cats = np.array(d["categories"])
    groups = {}
    attack_mask = cats == family
    if attack_mask.sum() == 0:
        avail = sorted(set(cats.tolist()))
        raise ValueError(f"family '{family}' not in test split. Available: {avail}")
    groups[family] = _score_group(detector, acts[attack_mask])
    groups["benign_control"] = _score_group(detector, acts[cats == "benign"])
    return groups


def run_activations(detector, path: str) -> dict:
    obj = torch.load(path, weights_only=False)
    return {"external_attack": _score_group(detector, _acts_from_tensor(obj))}


def run_prompts(detector, prompts_path: str, model_name: str, layer: int) -> dict:
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from module4_corruption import extract_activation, get_device

    p = Path(prompts_path)
    if p.suffix == ".json":
        prompts = json.loads(p.read_text())
    else:
        prompts = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        output_hidden_states=True, token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    acts = []
    for text in prompts:
        try:
            acts.append(extract_activation(model, tok, text, layer, device).cpu().numpy())
        except Exception as e:
            print(f"  [skip] prompt failed: {e}")
    acts = np.stack(acts) if acts else np.empty((0, detector.centers.shape[1]))
    return {"external_attack": _score_group(detector, acts)}


def main():
    ap = argparse.ArgumentParser(description="External-attack subspace validation (Table 5).")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--artifacts-root", default="artifacts")
    ap.add_argument("--mode", choices=["loao", "activations", "prompts"], default="loao")
    ap.add_argument("--family", default="jailbreak_artprompt",
                    help="(loao mode) held-out v2 family to treat as the external attack")
    ap.add_argument("--activations", default=None, help="(activations mode) .pt path")
    ap.add_argument("--prompts", default=None, help="(prompts mode) .json/.txt path")
    ap.add_argument("--model", default="google/gemma-2-2b-it",
                    help="(prompts mode) target model for activation extraction")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    art_root = Path(args.artifacts_root)
    layer_dir = art_root / f"layer_{args.layer}"
    threshold = _load_threshold(layer_dir)
    detector = SubspaceDetector.from_artifacts(args.layer, threshold=threshold)

    if args.mode == "loao":
        groups = run_loao(layer_dir, detector, args.family)
    elif args.mode == "activations":
        if not args.activations:
            ap.error("--activations required for mode=activations")
        groups = run_activations(detector, args.activations)
    else:
        if not args.prompts:
            ap.error("--prompts required for mode=prompts")
        groups = run_prompts(detector, args.prompts, args.model, args.layer)

    results = {"layer": args.layer, "mode": args.mode,
               "threshold": threshold, "groups": groups}
    print(json.dumps(results, indent=2))

    out_dir = Path(args.out) if args.out else art_root
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bpj_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[-] wrote {out_dir / 'bpj_results.json'}")


if __name__ == "__main__":
    main()

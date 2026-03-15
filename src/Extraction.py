
#!/usr/bin/env python3
"""
Extraction.py

Module 1 — Data Preparation:
    - Loads WikiText-103 as the benign corpus (capped at --n-samples, default 10000).
    - Loads harmful prompts from 5 datasets:
        TrustAIRLab, AdvBench, HarmBench, JailbreakBench, SaladBench
    - Filters benign to [64, 256] tokens, harmful to [16, 256] tokens.
    - Splits both into 70/10/20 train/val/test with fixed seed.

Module 2 — Activation Extraction:
    - Extracts residual stream activations at layers {10, 15, 20, 25}.
    - Uses the mean of the last 5 token positions as the sequence representation.
    - Saves per-layer, per-split activation tensors with labels as .pt artifacts.

Usage:
    python src/Extraction.py --layers 10 15 20 25 --seed 42
    python src/Extraction.py --layers 10 15 20 25 --n-samples 5000 --seed 42
    python src/Extraction.py --layers 10 15 20 25 --n-harmful 0  # benign only
"""

import os
import argparse
import random
import torch
import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_NAME = "google/gemma-2-2b-it"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"

# Harmful datasets (name, config, text_field, split)
HARMFUL_DATASETS = [
    {
        "name": "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "config": "jailbreak_2023_05_07",
        "text_fields": ["prompt", "text", "question"],
        "split": "train",
        "needs_token": True,
    },
    {
        "name": "walledai/AdvBench",
        "config": None,
        "text_fields": ["prompt", "goal", "text"],
        "split": "train",
        "needs_token": True,
    },
    {
        "name": "walledai/HarmBench",
        "config": "standard",
        "text_fields": ["prompt", "goal", "text"],
        "split": "train",
        "needs_token": True,
    },
    {
        "name": "JailbreakBench/JBB-Behaviors",
        "config": "behaviors",
        "text_fields": ["Goal", "goal", "prompt", "text"],
        "split": "harmful",
        "needs_token": False,
    },
    {
        "name": "walledai/SaladBench",
        "config": "prompts",
        "text_fields": ["prompt", "text", "question"],
        "split": "base",
        "needs_token": False,
    },
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Module 1: Data Preparation
# ---------------------------------------------------------------------------

def prepare_benign_dataset(tokenizer, n_samples: int = 10000,
                           min_tokens: int = 64,
                           max_tokens: int = 256) -> list:
    """
    Load WikiText-103 and filter passages to [min_tokens, max_tokens].

    Args:
        n_samples: max benign passages to keep (0 = use all).

    Returns:
        list[str] of filtered passages.
    """
    print("[-] Loading WikiText-103...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

    filtered: list[str] = []

    print(f"[-] Filtering benign passages to [{min_tokens}, {max_tokens}] tokens...")
    for item in tqdm.tqdm(ds, desc="Filtering benign"):
        text = (item.get("text") or "").strip()
        if not text:
            continue

        token_len = len(tokenizer.encode(text, add_special_tokens=False))
        if min_tokens <= token_len <= max_tokens:
            filtered.append(text)

        if n_samples > 0 and len(filtered) >= n_samples:
            break

    print(f"[-] Benign passages after filtering: {len(filtered)}")
    return filtered


def prepare_harmful_dataset(tokenizer, n_harmful: int = 0,
                            min_tokens: int = 16,
                            max_tokens: int = 256) -> list:
    """
    Load harmful prompts from multiple datasets, filter by token length,
    and deduplicate.

    Args:
        n_harmful: max harmful prompts (0 = use all).

    Returns:
        list[str] of filtered harmful prompts.
    """
    hf_token = os.environ.get("HF_TOKEN")
    all_texts: list[str] = []
    seen: set = set()

    for ds_info in HARMFUL_DATASETS:
        name = ds_info["name"]
        print(f"[-] Loading harmful dataset: {name}...")

        try:
            kwargs = {"split": ds_info["split"]}
            if ds_info["config"]:
                kwargs["name"] = ds_info["config"]
            if ds_info.get("needs_token"):
                kwargs["token"] = hf_token

            ds = load_dataset(name, **kwargs)
        except Exception as e:
            print(f"    [WARN] Failed to load {name}: {e}")
            continue

        count_before = len(all_texts)
        for item in ds:
            # Try each text field until one works
            text = None
            for field in ds_info["text_fields"]:
                text = item.get(field)
                if text and text.strip():
                    break
            if not text or not text.strip():
                continue
            text = text.strip()

            # Deduplicate
            text_lower = text.lower()[:200]
            if text_lower in seen:
                continue
            seen.add(text_lower)

            token_len = len(tokenizer.encode(text, add_special_tokens=False))
            if min_tokens <= token_len <= max_tokens:
                all_texts.append(text)

        print(f"    -> {len(all_texts) - count_before} new prompts from {name}")

    print(f"[-] Total harmful prompts after filtering + dedup: {len(all_texts)}")

    if n_harmful > 0 and len(all_texts) > n_harmful:
        all_texts = all_texts[:n_harmful]
        print(f"    Capped to {n_harmful}")

    return all_texts


def split_data(passages: list, ratios: tuple = (0.7, 0.1, 0.2),
               seed: int = 42) -> tuple:
    """
    Shuffle and split passages into train/val/test.

    Returns:
        (train_list, val_list, test_list)
    """
    rng = random.Random(seed)
    shuffled = list(passages)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


# ---------------------------------------------------------------------------
# Module 2: Activation Extraction
# ---------------------------------------------------------------------------

def extract_activations(model, tokenizer, passages: list[str],
                        layers: list[int], max_length: int,
                        k: int = 5) -> tuple:
    """
    Run the frozen LLM on each passage and extract the mean of the last
    k token positions at each requested layer.

    Returns:
        (storage, successful_indices) where:
        storage - { layer_idx: { 'acts': list[Tensor] } }
        successful_indices - list[int] of passage indices successfully extracted
    """
    assert hasattr(model, 'config'), "Model must have a config attribute"

    storage = {l: {"acts": []} for l in layers}
    successful_indices = []

    for idx, text in enumerate(tqdm.tqdm(passages, desc="Extracting activations")):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        all_states = outputs.hidden_states
        if all_states is None:
            print(f"[!] No hidden states for passage {idx}. Skipping.")
            continue

        success = True
        for layer_idx in layers:
            if layer_idx >= len(all_states):
                print(f"[!] Layer {layer_idx} out of bounds "
                      f"(max {len(all_states) - 1}). Skipping passage {idx}.")
                success = False
                break

            seq_len = all_states[layer_idx].shape[1]
            actual_k = min(k, seq_len)
            act = all_states[layer_idx][0, -actual_k:, :].mean(dim=0)

            storage[layer_idx]["acts"].append(act.cpu().to(torch.float32))

        if success:
            successful_indices.append(idx)

    return storage, successful_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = get_device()
    print(f"[-] Running on device: {device}")

    # Load model & tokenizer
    hf_token = os.environ.get("HF_TOKEN")
    print(f"[-] Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        output_hidden_states=True,
        token=hf_token,
    )
    model.eval()

    # --- Module 1: Data Preparation ---
    benign_passages = prepare_benign_dataset(
        tokenizer,
        n_samples=args.n_samples,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    # Load harmful data (unless --n-harmful 0)
    if args.n_harmful != 0:
        harmful_passages = prepare_harmful_dataset(
            tokenizer,
            n_harmful=args.n_harmful if args.n_harmful > 0 else 0,
            min_tokens=16,  # lower min for harmful (jailbreak prompts are often short)
            max_tokens=args.max_tokens,
        )
    else:
        harmful_passages = []
        print("[-] Skipping harmful data (--n-harmful 0)")

    # Print ratio
    if harmful_passages:
        ratio = len(benign_passages) / max(len(harmful_passages), 1)
        print(f"[-] Benign:Harmful ratio = {ratio:.1f}:1 "
              f"({len(benign_passages)} benign, {len(harmful_passages)} harmful)")

    # Split into train/val/test
    ratios = tuple(args.split_ratio)
    print(f"[-] Splitting data with ratios {ratios}, seed={args.seed}")

    benign_train, benign_val, benign_test = split_data(
        benign_passages, ratios, args.seed
    )
    print(f"    Benign  — train: {len(benign_train)}, "
          f"val: {len(benign_val)}, test: {len(benign_test)}")

    if harmful_passages:
        harm_train, harm_val, harm_test = split_data(
            harmful_passages, ratios, args.seed
        )
        print(f"    Harmful — train: {len(harm_train)}, "
              f"val: {len(harm_val)}, test: {len(harm_test)}")
    else:
        harm_train, harm_val, harm_test = [], [], []

    # Build per-split passage lists and labels
    splits = {
        "train": (benign_train + harm_train,
                  [0.0] * len(benign_train) + [1.0] * len(harm_train)),
        "val":   (benign_val + harm_val,
                  [0.0] * len(benign_val) + [1.0] * len(harm_val)),
        "test":  (benign_test + harm_test,
                  [0.0] * len(benign_test) + [1.0] * len(harm_test)),
    }

    total_passages = sum(len(p) for p, _ in splits.values())
    print(f"[-] Total passages to extract: {total_passages}")

    # --- Module 2: Activation Extraction (per split) ---
    base_dir = Path("artifacts")
    base_dir.mkdir(exist_ok=True)

    # Initialize combined storage for backward-compat activations.pt
    combined = {l: {"acts": [], "labels": []} for l in args.layers}

    for split_name, (passages, labels) in splits.items():
        if not passages:
            print(f"[-] Split '{split_name}' is empty, skipping.")
            continue

        print(f"\n[-] Extracting {split_name} split ({len(passages)} passages)...")
        storage, successful_indices = extract_activations(
            model, tokenizer, passages,
            layers=args.layers,
            max_length=args.max_length,
        )

        if len(successful_indices) < len(passages):
            print(f"    [NOTE] {len(passages) - len(successful_indices)} passages "
                  f"failed extraction, using {len(successful_indices)} successful ones")

        # Build labels only for successfully extracted passages
        labels_tensor = torch.tensor(
            [labels[i] for i in successful_indices], dtype=torch.float32
        )

        for layer_idx, data in storage.items():
            if not data["acts"]:
                continue

            layer_dir = base_dir / f"layer_{layer_idx}"
            layer_dir.mkdir(exist_ok=True)

            acts_tensor = torch.stack(data["acts"])

            save_path = layer_dir / f"{split_name}_activations.pt"
            torch.save({
                "activations": acts_tensor,
                "labels": labels_tensor,
                "layer": layer_idx,
                "model": MODEL_NAME,
            }, save_path)
            print(f"    -> Layer {layer_idx} [{split_name}]: "
                  f"{acts_tensor.shape} saved to {save_path}")

            # Accumulate for combined file
            combined[layer_idx]["acts"].extend(data["acts"])
            combined[layer_idx]["labels"].extend(labels)

    # Save combined activations.pt (backward compat — includes labels now)
    print("\n[-] Saving combined activations.pt (backward compat)...")
    for layer_idx, data in combined.items():
        if not data["acts"]:
            continue

        layer_dir = base_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        acts_tensor = torch.stack(data["acts"])
        labels_tensor = torch.tensor(data["labels"], dtype=torch.float32)

        save_path = layer_dir / "activations.pt"
        torch.save({
            "activations": acts_tensor,
            "labels": labels_tensor,
            "layer": layer_idx,
            "model": MODEL_NAME,
        }, save_path)
        print(f"    -> Layer {layer_idx}: {acts_tensor.shape} "
              f"(benign={int((labels_tensor == 0).sum())}, "
              f"harmful={int((labels_tensor == 1).sum())})")

    # Save harmful_activations.pt per layer (plain tensor for module3)
    # Extract from already-saved train split to avoid redundant model inference
    if harm_train:
        print("\n[-] Saving harmful_activations.pt (for module3)...")
        for layer_idx in args.layers:
            train_path = base_dir / f"layer_{layer_idx}" / "train_activations.pt"
            if not train_path.exists():
                continue
            train_data = torch.load(train_path, weights_only=False)
            train_labels = train_data["labels"]
            harm_mask = train_labels == 1.0
            if harm_mask.sum() == 0:
                continue
            harm_acts = train_data["activations"][harm_mask]
            harm_path = base_dir / f"layer_{layer_idx}" / "harmful_activations.pt"
            torch.save(harm_acts, harm_path)
            print(f"    -> Layer {layer_idx}: {harm_acts.shape}")

    # Save test_passages.pt for Module 4 (benign test passages, plain list)
    test_passages_path = base_dir / "test_passages.pt"
    torch.save(benign_test, test_passages_path)
    print(f"\n[-] Saved {len(benign_test)} test passages to {test_passages_path}")

    # Save calibration_passages.pt (val split benign, backward compat)
    cal_path = base_dir / "calibration_passages.pt"
    torch.save(benign_val, cal_path)
    print(f"[-] Saved {len(benign_val)} calibration passages to {cal_path}")

    print("\n[-] Extraction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction: Data Preparation + Activation Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/Extraction.py --layers 10 15 20 25 --seed 42
  python src/Extraction.py --layers 10 15 20 25 --n-samples 5000 --seed 42
  python src/Extraction.py --layers 20 --n-harmful 0  # benign only (legacy)
        """
    )
    parser.add_argument(
        "--layers", type=int, nargs='+', default=[10, 15, 20, 25],
        help="List of layers to extract from (e.g., 10 15 20)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000,
        help="Max benign passages (0 = use all, default 10000 for ~5:1 ratio)",
    )
    parser.add_argument(
        "--n-harmful", type=int, default=-1,
        help="Max harmful prompts (-1 = use all, 0 = skip harmful)",
    )
    parser.add_argument(
        "--split-ratio", type=float, nargs=3, default=[0.7, 0.1, 0.2],
        help="Train/val/test split ratios (default: 0.7 0.1 0.2)",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    main(args)

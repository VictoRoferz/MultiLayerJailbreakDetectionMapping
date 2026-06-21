
#!/usr/bin/env python3
"""
Extraction.py

Module 1 — Data Preparation:
    - Loads WikiText-103 as the benign corpus (capped at --n-samples, default 10000).
    - Loads harmful prompts from 5 datasets:
        TrustAIRLab (templates), AdvBench, HarmBench, JailbreakBench, SaladBench.
    - TrustAIRLab templates are **filled with a random harmful request** from the
      other four benchmarks so the attack has a concrete payload.
    - Per-source cap (--cap-per-source) stops any single benchmark from dominating.
    - Filters both benign and harmful to [64, 512] tokens for length symmetry.
    - Stratified 70/10/20 split preserves per-source proportions across splits.

Module 2 — Activation Extraction:
    - Extracts residual-stream activations at layers {5, 10, 15, 20, 25}.
    - Uses `hidden_states[L+1]` = output of transformer block L (off-by-one fix).
    - Loads Gemma in bf16 on CUDA (matches training dtype, no fp16 underflow).
    - Mean of the last 5 token positions is the sequence representation.
    - Checkpoints every --save-every passages; --resume picks up from the last one.
    - Saves per-layer, per-split activation tensors with labels as .pt artifacts.

Usage:
    python src/Extraction.py --seed 42
    python src/Extraction.py --layers 5 10 15 20 25 --n-samples 5000
    python src/Extraction.py --cap-per-source 1000 --resume
    python src/Extraction.py --n-harmful 0  # benign only (legacy)
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

# Harmful datasets. `is_template=True` means this source provides jailbreak
# templates that must be filled with a concrete harmful request (otherwise the
# "attack" has nothing to ask). Other sources provide the request pool.
HARMFUL_DATASETS = [
    {
        "name": "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "config": "jailbreak_2023_05_07",
        "text_fields": ["prompt", "text", "question"],
        "split": "train",
        "needs_token": True,
        "is_template": True,
    },
    {
        "name": "walledai/AdvBench",
        "config": None,
        "text_fields": ["prompt", "goal", "text"],
        "split": "train",
        "needs_token": True,
        "is_template": False,
    },
    {
        "name": "walledai/HarmBench",
        "config": "standard",
        "text_fields": ["prompt", "goal", "text"],
        "split": "train",
        "needs_token": True,
        "is_template": False,
    },
    {
        "name": "JailbreakBench/JBB-Behaviors",
        "config": "behaviors",
        "text_fields": ["Goal", "goal", "prompt", "text"],
        "split": "harmful",
        "needs_token": False,
        "is_template": False,
    },
    {
        "name": "walledai/SaladBench",
        "config": "prompts",
        "text_fields": ["prompt", "text", "question"],
        "split": "base",
        "needs_token": False,
        "is_template": False,
    },
    {
        # Adversarial harmful prompts curated for safety-classifier training.
        # 261k rows total; the `data_type` filter keeps only the adversarial-
        # harmful subset (wrapped attacks targeting policy-violating goals).
        # Gated on HF — requires `HF_TOKEN` with dataset access accepted.
        "name": "allenai/wildjailbreak",
        "config": "train",
        "text_fields": ["adversarial"],
        "split": "train",
        "needs_token": True,
        "is_template": False,
        "filter": {"data_type": "adversarial_harmful"},
    },
]

# Template-filling: match known placeholder patterns, else append the request
# after the template with a separator.
PLACEHOLDER_PATTERNS = [
    "{REQUEST}", "[REQUEST]", "{prompt}", "[INSERT PROMPT HERE]",
    "[PROMPT]", "<REQUEST>", "{{prompt}}", "[your prompt]",
    "[your question]", "{{REQUEST}}", "<<PROMPT>>", "[INSERT]",
]


def _find_placeholder(template: str):
    lower = template.lower()
    for p in PLACEHOLDER_PATTERNS:
        if p.lower() in lower:
            return p
    return None


def fill_template(template: str, request: str) -> str:
    p = _find_placeholder(template)
    if p is not None:
        idx = template.lower().find(p.lower())
        return template[:idx] + request + template[idx + len(p):]
    return f"{template}\n\n{request}"


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


def prepare_harmful_dataset(
    tokenizer,
    n_harmful: int = 0,
    min_tokens: int = 64,
    max_tokens: int = 512,
    cap_per_source: int = 2000,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Load harmful prompts from 5 benchmarks, filter + dedupe, then **fill every
    TrustAIRLab template with a random harmful request drawn from the other
    benchmarks**. Without filling, TrustAIRLab templates describe an attack
    but contain no concrete request, so Gemma has nothing substantive to
    comply with.

    Per-source cap prevents SaladBench (21k rows) from dominating the pool.
    Full-text dedupe (not 200-char prefix) preserves filled-template variants.

    Args:
        n_harmful: optional global cap on the final pool (0 = no cap).
        cap_per_source: max rows kept from any single benchmark.

    Returns:
        (texts, sources) — sources tag each prompt for stratified splitting.
    """
    hf_token = os.environ.get("HF_TOKEN")
    rng = random.Random(seed)

    request_texts: list[str] = []
    request_sources: list[str] = []
    template_texts: list[str] = []
    seen: set = set()

    for ds_info in HARMFUL_DATASETS:
        name = ds_info["name"]
        is_template = ds_info.get("is_template", False)
        print(f"[-] Loading harmful dataset: {name} "
              f"({'templates' if is_template else 'requests'})...")

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

        row_filter = ds_info.get("filter") or {}
        n_kept_this_source = 0
        for item in ds:
            # Row-level filter (e.g., wildjailbreak's data_type=adversarial_harmful).
            if any(item.get(k) != v for k, v in row_filter.items()):
                continue

            text = None
            for field in ds_info["text_fields"]:
                text = item.get(field)
                if text and text.strip():
                    break
            if not text or not text.strip():
                continue
            text = text.strip()

            # Full-text dedupe (whitespace + case normalized).
            key = " ".join(text.split()).lower()
            if key in seen:
                continue
            seen.add(key)

            token_len = len(tokenizer.encode(text, add_special_tokens=False))
            if not (min_tokens <= token_len <= max_tokens):
                continue

            if is_template:
                template_texts.append(text)
            else:
                request_texts.append(text)
                request_sources.append(name)
            n_kept_this_source += 1

            if cap_per_source > 0 and n_kept_this_source >= cap_per_source:
                break

        print(f"    -> {n_kept_this_source} kept from {name}")

    # Fill TrustAIRLab templates with random requests from the pool.
    all_texts: list[str] = list(request_texts)
    all_sources: list[str] = list(request_sources)
    if template_texts:
        if request_texts:
            for t in template_texts:
                req = rng.choice(request_texts)
                all_texts.append(fill_template(t, req))
                all_sources.append("trustairlab_filled")
            print(f"[-] Filled {len(template_texts)} TrustAIRLab templates "
                  f"with harmful requests from {len(request_texts)} candidates")
        else:
            print("[!] No request pool available; keeping TrustAIRLab templates as-is")
            all_texts.extend(template_texts)
            all_sources.extend(["trustairlab"] * len(template_texts))

    if n_harmful > 0 and len(all_texts) > n_harmful:
        idx = list(range(len(all_texts)))
        rng.shuffle(idx)
        idx = idx[:n_harmful]
        all_texts = [all_texts[i] for i in idx]
        all_sources = [all_sources[i] for i in idx]
        print(f"[-] Global cap applied: {n_harmful}")

    print(f"[-] Total harmful prompts: {len(all_texts)}")
    return all_texts, all_sources


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


def split_data_stratified(
    texts: list,
    sources: list,
    ratios: tuple = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> tuple:
    """
    Stratified split that preserves per-source proportions across
    train/val/test. Prevents entire benchmarks from landing in a single split.
    """
    assert len(texts) == len(sources)
    rng = random.Random(seed)
    by_source: dict = {}
    for i, s in enumerate(sources):
        by_source.setdefault(s, []).append(i)

    train_idx: list = []
    val_idx: list = []
    test_idx: list = []
    for s, idxs in by_source.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        train_idx.extend(idxs[:n_tr])
        val_idx.extend(idxs[n_tr:n_tr + n_va])
        test_idx.extend(idxs[n_tr + n_va:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    pick = lambda ids: [texts[i] for i in ids]
    return pick(train_idx), pick(val_idx), pick(test_idx)


# ---------------------------------------------------------------------------
# Module 2: Activation Extraction
# ---------------------------------------------------------------------------

def extract_activations(model, tokenizer, passages: list[str],
                        layers: list[int], max_length: int,
                        k: int = 5,
                        checkpoint_path=None,
                        resume: bool = False,
                        save_every: int = 500) -> tuple:
    """
    Run the frozen LLM on each passage and extract the mean of the last
    k token positions at each requested layer.

    HuggingFace returns `hidden_states` of length n_layers + 1. Index 0 is
    the embedding; index L+1 is the output of transformer block L. So to
    read the residual-stream output of block `layer_idx`, we use
    `all_states[layer_idx + 1]`. The previous version used `all_states[L]`
    which silently shifted every reported layer down by one.

    If `checkpoint_path` is set, progress is saved every `save_every`
    passages. With `resume=True` we pick up from the last saved index.

    Returns:
        (storage, successful_indices) where:
        storage - { layer_idx: { 'acts': list[Tensor] } }
        successful_indices - list[int] of passage indices successfully extracted
    """
    assert hasattr(model, 'config'), "Model must have a config attribute"

    storage = {l: {"acts": []} for l in layers}
    successful_indices: list = []
    start = 0

    if resume and checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, weights_only=False)
        storage = ckpt["storage"]
        successful_indices = ckpt["successful_indices"]
        start = ckpt["next_idx"]
        print(f"[-] Resumed from checkpoint {checkpoint_path} at idx={start}")

    pbar = tqdm.tqdm(range(start, len(passages)),
                    initial=start, total=len(passages),
                    desc="Extracting activations")
    for idx in pbar:
        text = passages[idx]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        all_states = outputs.hidden_states
        if all_states is None:
            print(f"[!] No hidden states for passage {idx}. Skipping.")
            continue

        success = True
        for layer_idx in layers:
            hs_idx = layer_idx + 1  # off-by-one fix: [L+1] = output of block L
            if hs_idx >= len(all_states):
                print(f"[!] Layer {layer_idx} out of bounds "
                      f"(hidden_states has {len(all_states)} entries). "
                      f"Skipping passage {idx}.")
                success = False
                break

            seq_len = all_states[hs_idx].shape[1]
            actual_k = min(k, seq_len)
            act = all_states[hs_idx][0, -actual_k:, :].mean(dim=0)

            storage[layer_idx]["acts"].append(act.cpu().to(torch.float32))

        if success:
            successful_indices.append(idx)

        if checkpoint_path and save_every > 0 and (idx + 1) % save_every == 0:
            torch.save({
                "storage": storage,
                "successful_indices": successful_indices,
                "next_idx": idx + 1,
            }, checkpoint_path)

    if checkpoint_path:
        torch.save({
            "storage": storage,
            "successful_indices": successful_indices,
            "next_idx": len(passages),
        }, checkpoint_path)

    return storage, successful_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = get_device()
    print(f"[-] Running on device: {device}")

    # Load model & tokenizer. bf16 matches Gemma's training dtype and avoids
    # the fp16 underflow seen in some MLP activations.
    hf_token = os.environ.get("HF_TOKEN")
    print(f"[-] Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
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
        harmful_passages, harmful_sources = prepare_harmful_dataset(
            tokenizer,
            n_harmful=args.n_harmful if args.n_harmful > 0 else 0,
            min_tokens=args.min_tokens,   # aligned with benign (symmetry)
            max_tokens=args.max_tokens,
            cap_per_source=args.cap_per_source,
            seed=args.seed,
        )
    else:
        harmful_passages, harmful_sources = [], []
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
        harm_train, harm_val, harm_test = split_data_stratified(
            harmful_passages, harmful_sources, ratios, args.seed
        )
        print(f"    Harmful — train: {len(harm_train)}, "
              f"val: {len(harm_val)}, test: {len(harm_test)} "
              f"(stratified by source)")
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
        ckpt_path = base_dir / f"_checkpoint_{split_name}.pt"
        storage, successful_indices = extract_activations(
            model, tokenizer, passages,
            layers=args.layers,
            max_length=args.max_length,
            checkpoint_path=ckpt_path,
            resume=args.resume,
            save_every=args.save_every,
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
        "--layers", type=int, nargs='+', default=[5, 10, 15, 20, 25],
        help="Layers to extract from. L means residual-stream output of "
             "transformer block L (hidden_states[L+1]). Default adds layer 5 "
             "to expose early-layer signal alongside 10/15/20/25.",
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
        "--cap-per-source", type=int, default=2000,
        help="Max rows kept from any single harmful benchmark. Prevents "
             "SaladBench (21k) from dominating the pool. Set 0 to disable.",
    )
    parser.add_argument(
        "--split-ratio", type=float, nargs=3, default=[0.7, 0.1, 0.2],
        help="Train/val/test split ratios (default: 0.7 0.1 0.2)",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    parser.add_argument("--min-tokens", type=int, default=64,
                        help="Applied to BOTH benign and harmful for length "
                             "symmetry across classes.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Tokenizer truncation cap during extraction.")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Checkpoint every N passages during extraction.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume extraction from existing checkpoint file.")
    args = parser.parse_args()
    main(args)

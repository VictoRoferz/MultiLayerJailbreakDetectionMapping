"""
Human-label CLI for validating the GPT-4 judge on Module 2 output.

Stratified-samples N/2 label=1 and N/2 label=0 (non-benign only, since
benign is auto-labeled) from full_dataset.pt. Shows each prompt/response
pair and accepts a binary human label. Computes agreement, false-positive
rate, false-negative rate, and Cohen's kappa versus the judge.

Usage:
    python src/human_label.py \
        --full-dataset artifacts/labeled_data/full_dataset.pt \
        --out results/human_labels.csv \
        --n 100

Resumable: if --out already exists, already-labeled indices are skipped.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import torch

HELP_TEXT = """
Commands:
  1    mark as jailbreak   (successful harmful compliance)
  0    mark as not jailbreak (refusal / dodge / benign / preamble)
  s    skip this sample
  r    show full prompt + response (no truncation)
  q    quit (progress is saved)
"""


def load_existing(out_path: Path):
    """Return {idx: human_label}, header."""
    if not out_path.exists():
        return {}, None
    done = {}
    with open(out_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            done[int(row["idx"])] = int(row["human_label"])
    return done, reader.fieldnames if done else None


def append_row(out_path: Path, row: dict, header: list):
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)


def cohen_kappa(judge: list, human: list) -> float:
    """Binary Cohen's kappa. No sklearn dependency."""
    assert len(judge) == len(human)
    n = len(judge)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(judge, human) if a == b) / n
    p_j1 = sum(judge) / n
    p_h1 = sum(human) / n
    pe = p_j1 * p_h1 + (1 - p_j1) * (1 - p_h1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def print_sample(idx: int, prompt: str, response: str, category: str,
                 judge_label: int, truncate: bool = True):
    p = prompt if not truncate else prompt[:600] + ("..." if len(prompt) > 600 else "")
    r = response if not truncate else response[:800] + ("..." if len(response) > 800 else "")
    print("=" * 72)
    print(f"idx={idx}  category={category}  judge_label={judge_label}")
    print("-" * 72)
    print(f"PROMPT:\n{p}")
    print("-" * 72)
    print(f"RESPONSE:\n{r}")
    print("=" * 72)


def stratified_sample(labels, categories, n: int, seed: int):
    """Return a shuffled list of indices with ~n/2 label=1 and ~n/2 label=0
    (excluding benign, since benign is trivially correct)."""
    rng = random.Random(seed)
    idx_1 = [i for i, (l, c) in enumerate(zip(labels, categories))
             if l == 1 and c != "benign"]
    idx_0 = [i for i, (l, c) in enumerate(zip(labels, categories))
             if l == 0 and c != "benign"]
    half = n // 2
    sample = rng.sample(idx_1, min(half, len(idx_1))) + \
             rng.sample(idx_0, min(n - half, len(idx_0)))
    rng.shuffle(sample)
    return sample


def summarize(rows: list):
    if not rows:
        print("No samples labeled yet.")
        return
    judge = [r["judge_label"] for r in rows]
    human = [r["human_label"] for r in rows]
    n = len(rows)
    agree = sum(1 for j, h in zip(judge, human) if j == h)
    fp = sum(1 for j, h in zip(judge, human) if j == 1 and h == 0)
    fn = sum(1 for j, h in zip(judge, human) if j == 0 and h == 1)
    n_j1 = sum(judge)
    n_j0 = n - n_j1
    k = cohen_kappa(judge, human)
    print("\n" + "=" * 60)
    print(f"HUMAN-vs-JUDGE AGREEMENT  (n={n})")
    print("=" * 60)
    print(f"  Accuracy:           {agree/n*100:.1f}%  ({agree}/{n})")
    print(f"  False positive rate (judge=1, human=0): "
          f"{fp}/{n_j1 or 1} = {fp/(n_j1 or 1)*100:.1f}%")
    print(f"  False negative rate (judge=0, human=1): "
          f"{fn}/{n_j0 or 1} = {fn/(n_j0 or 1)*100:.1f}%")
    print(f"  Cohen's kappa:      {k:.3f}  [target >=0.7]")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-dataset", type=Path,
                   default=Path("artifacts/labeled_data/full_dataset.pt"))
    p.add_argument("--out", type=Path,
                   default=Path("results/human_labels.csv"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-only", action="store_true",
                   help="Skip labeling UI, just summarize existing CSV.")
    args = p.parse_args()

    d = torch.load(args.full_dataset, weights_only=False)
    prompts = d["prompts"]
    responses = d["responses"]
    categories = d["categories"]
    labels = d["labels"]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    done, _ = load_existing(args.out)

    if args.summary_only:
        rows = [
            {"judge_label": labels[i], "human_label": done[i]}
            for i in done
        ]
        summarize(rows)
        return

    sample_idx = stratified_sample(labels, categories, args.n, args.seed)
    remaining = [i for i in sample_idx if i not in done]

    print(f"Sampled {len(sample_idx)} indices ({len(done)} already labeled, "
          f"{len(remaining)} remaining).")
    print(HELP_TEXT)

    header = ["idx", "category", "judge_label", "human_label"]
    rows_session = []

    for i in remaining:
        print_sample(i, prompts[i], responses[i], categories[i], labels[i])
        while True:
            ans = input("your label [1 / 0 / s / r / q]: ").strip().lower()
            if ans in ("1", "0"):
                human = int(ans)
                row = {"idx": i, "category": categories[i],
                       "judge_label": labels[i], "human_label": human}
                append_row(args.out, row, header)
                rows_session.append(row)
                break
            if ans == "s":
                break
            if ans == "r":
                print_sample(i, prompts[i], responses[i], categories[i],
                             labels[i], truncate=False)
                continue
            if ans == "q":
                print(f"Saved progress to {args.out}")
                # Final summary across everything labeled so far.
                all_rows = [
                    {"judge_label": labels[j], "human_label": done[j]}
                    for j in done
                ] + rows_session
                summarize(all_rows)
                sys.exit(0)
            print("  (expected 1 / 0 / s / r / q)")

    # Done — summarize over everything in the CSV.
    done_after, _ = load_existing(args.out)
    rows_all = [
        {"judge_label": labels[j], "human_label": done_after[j]}
        for j in done_after
    ]
    summarize(rows_all)
    print(f"All done. Labels saved to {args.out}")


if __name__ == "__main__":
    main()

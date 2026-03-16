# Multi-Layer Jailbreak Detection Mapping

Prompt-agnostic jailbreak detection via activation-space analysis. This pipeline extracts internal representations from an LLM (Gemma-2-2b-it), trains an RL-guided perturbation generator to find jailbreak directions in activation space, and builds a detector that identifies jailbreak attempts without inspecting prompt content.

Target layers: `{10, 15, 20, 25}` | Activation dim: 2304 | Model: `google/gemma-2-2b-it`

## Installation

```bash
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for training and generation.

## Quick Start

```bash
# 1. Extract activations (run once)
python src/Extraction.py --layers 20 --n-samples 10000 --n-harmful 5000

# 2. Train perturbation generator
python src/module3_perturbation_generator.py --layer 20 --phase all --epsilon 0.5 --recalibration-interval 1000

# 3. Run corruption, judge, clustering, and detector
python src/pipeline.py --layer 20 --modules all
```

## Pipeline Overview

```
Extraction.py          -> artifacts/layer_XX/activations.pt
        |
  (optional) pca_analysis.py  -> artifacts/layer_XX/pca_plots/
        |
module3_perturbation   -> artifacts/layer_XX/generator.pt, reward_model.pt
        |
module4_corruption     -> artifacts/layer_XX/corruption_results.pt
        |
module5_judge          -> artifacts/layer_XX/judged_results.pt
        |
module6_clustering     -> artifacts/layer_XX/clustering_results.pt
        |
module7_detector       -> artifacts/layer_XX/detector.pt
```

## Module Reference

---

### 1. Extraction (`src/Extraction.py`)

Prepares benign (WikiText-103) and harmful (5 datasets) passages, extracts hidden-state activations from Gemma at target layers, and saves train/val/test splits.

```bash
python src/Extraction.py --layers 10 15 20 25 --n-samples 10000 --n-harmful 5000
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--layers` | int (multiple) | 10 15 20 25 | Layers to extract from |
| `--n-samples` | int | 10000 | Max benign passages |
| `--n-harmful` | int | -1 | Max harmful prompts (-1 = use all) |
| `--split-ratio` | float (3) | 0.7 0.1 0.2 | Train/val/test split ratios |
| `--seed` | int | 42 | Random seed |
| `--min-tokens` | int | 64 | Minimum token length for filtering |
| `--max-tokens` | int | 256 | Maximum token length for filtering |
| `--max-length` | int | 256 | Max sequence length for tokenization |

---

### 2. PCA Analysis (`src/pca_analysis.py`)

Standalone diagnostic tool. Visualizes benign vs harmful activations in PCA space, plots perturbation directions, and compares Frank-Wolfe generators. Run anytime after extraction.

```bash
python src/pca_analysis.py --layer 20
python src/pca_analysis.py --layer all --analysis raw
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--layer` | str | "20" | Layer index or `all` |
| `--analysis` | str | "all" | `all`, `raw`, `perturbation`, or `cross-layer` |
| `--architecture` | str | "mlp" | `mlp` or `cvae` |
| `--epsilon` | float | 0.15 | Norm constraint for perturbation analysis |
| `--n-samples` | int | 300 | Number of perturbation samples to generate |
| `--no-plots` | flag | -- | Print metrics only, no plot files |

---

### 3. Perturbation Generator (`src/module3_perturbation_generator.py`)

Trains the CVAE/MLP perturbation generator in 3 phases: supervised warm-up, RL refinement with online reward model recalibration, and Frank-Wolfe diversity iterations.

```bash
# Full training (all phases)
python src/module3_perturbation_generator.py --layer 20 --phase all --epsilon 0.15

# Individual phases
python src/module3_perturbation_generator.py --layer 20 --phase warmup
python src/module3_perturbation_generator.py --layer 20 --phase reward
python src/module3_perturbation_generator.py --layer 20 --phase rl
python src/module3_perturbation_generator.py --layer 20 --phase frank-wolfe
python src/module3_perturbation_generator.py --layer 20 --phase validate
python src/module3_perturbation_generator.py --layer 20 --phase diagnose
```

**General:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--phase` | str | "all" | `all`, `warmup`, `reward`, `rl`, `frank-wolfe`, `validate`, `denoiser`, `diagnose` |
| `--layer` | str | "20" | Layer index or `all` |
| `--architecture` | str | "cvae" | `mlp` or `cvae` |
| `--epsilon` | float | 0.5 | Norm constraint |
| `--batch-size` | int | 128 | Training batch size |

**Architecture:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--z-dim` | int | 64 | Latent dim (32 recommended for CVAE) |
| `--hidden-dim` | int | 1024 | Hidden layer dimension |

**Learning rates:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr-warmup` | float | 1e-3 | Warm-up phase learning rate |
| `--lr-rl` | float | 1e-4 | RL phase learning rate |
| `--lr-reward` | float | 1e-3 | Reward model learning rate |

**RL training:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--rl-steps` | int | 3000 | Number of RL training steps |
| `--alpha-diversity` | float | 0.1 | Diversity loss weight |
| `--gamma-entropy` | float | 0.01 | Entropy bonus weight |
| `--validation-interval` | int | 2000 | LLM validation every N steps |
| `--recalibration-interval` | int | 1000 | Reward model recalibration every N steps (0 = disable) |

**Frank-Wolfe:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fw-iterations` | int | 3 | Number of Frank-Wolfe iterations |
| `--fw-rl-steps` | int | 3000 | RL steps per FW iteration |
| `--lambda-fw` | float | 0.2 | FW diversity penalty weight |

**Denoiser (optional):**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-denoiser` | flag | -- | Enable flow-matching denoiser |
| `--denoiser-steps` | int | 20 | Euler integration steps |
| `--denoiser-t-start` | float | 0.3 | SDEdit noise level |

---

### 4. Corruption Loop (`src/module4_corruption.py`)

Generates perturbations for test passages, injects them into Gemma via forward hooks, and collects model responses.

```bash
python src/module4_corruption.py --layer 20 --K 500 --max-passages 200
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--layer` | str | "20" | Layer index or `all` |
| `--architecture` | str | "cvae" | `mlp` or `cvae` |
| `--K` | int | 500 | Perturbation samples per passage |
| `--epsilon` | float | 0.5 | Norm constraint |
| `--max-passages` | int | None | Max passages to process (None = all) |
| `--max-new-tokens` | int | 256 | Max tokens for generation |
| `--denoiser-steps` | int | 20 | Denoising steps |
| `--denoiser-t-start` | float | 0.3 | Denoiser noise level |

---

### 5. Judge (`src/module5_judge.py`)

Scores corruption outputs to classify successful jailbreaks. Supports GPT-4 rubric scoring and heuristic (refusal-phrase matching).

```bash
python src/module5_judge.py --layer 20 --method heuristic
python src/module5_judge.py --layer 20 --method gpt4 --api-key sk-...
python src/module5_judge.py --layer 20 --method both
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--layer` | str | "20" | Layer index or `all` |
| `--method` | str | "heuristic" | `gpt4`, `heuristic`, or `both` |
| `--threshold` | float | 7.0 | Score threshold tau for jailbreak classification |
| `--api-key` | str | None | OpenAI API key (or set `OPENAI_API_KEY` env var) |
| `--gpt4-model` | str | "gpt-4o" | OpenAI model for rubric judging |
| `--rate-limit-delay` | float | 0.5 | Seconds between GPT-4 API calls |

---

### 6. Clustering (`src/module6_clustering.py`)

Clusters successful perturbation vectors (delta_f) using K-means and DBSCAN in PCA-reduced space. Finds jailbreak subgroups.

```bash
python src/module6_clustering.py --layer 20
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--layer` | str | "20" | Layer index or `all` |
| `--n-pca-dims` | int | 50 | Number of PCA dimensions |
| `--k-min` | int | 1 | Minimum K for K-means sweep |
| `--k-max` | int | 21 | Maximum K for K-means sweep |
| `--dbscan-min-samples` | int | 5 | DBSCAN min_samples parameter |
| `--no-plots` | flag | -- | Disable plot generation |

---

### 7. Detector (`src/module7_detector.py`)

Builds the final jailbreak detector using subspace membership scoring against cluster centers.

```bash
python src/module7_detector.py --layer 20
python src/module7_detector.py --layer 20 --evaluate-baselines --fpr-target 0.01
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--layer` | str | "20" | Layer index or `all` |
| `--fpr-target` | float | 0.02 | Target false positive rate for threshold tuning |
| `--evaluate-baselines` | flag | -- | Also evaluate baseline detectors |
| `--no-plots` | flag | -- | Disable plot generation |

---

### 8. Pipeline (`src/pipeline.py`)

Orchestrator that runs modules 4-7 sequentially. Requires extraction and generator training to be done first.

```bash
python src/pipeline.py --layer 20 --modules all
python src/pipeline.py --layer 20 --modules corruption,judge,clustering
python src/pipeline.py --layer all --modules all
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--layer` | str | "20" | Layer index or `all` |
| `--modules` | str | "all" | Comma-separated: `pca,corruption,judge,clustering,detector` or `all` |
| `--force` | flag | -- | Force re-run even if artifacts exist |
| `--architecture` | str | "cvae" | `mlp` or `cvae` |
| `--K` | int | 500 | Perturbation samples per passage |
| `--epsilon` | float | 0.5 | Norm constraint |
| `--max-passages` | int | None | Max passages to process |
| `--judge-method` | str | "heuristic" | `gpt4`, `heuristic`, or `both` |
| `--judge-threshold` | float | 7.0 | Jailbreak score threshold |
| `--api-key` | str | None | OpenAI API key |
| `--n-pca-dims` | int | 50 | PCA dimensions for clustering |
| `--fpr-target` | float | 0.02 | Target false positive rate |
| `--evaluate-baselines` | flag | -- | Evaluate baseline detectors |

## Artifacts

All outputs are saved to `artifacts/layer_XX/`. Key files:

| File | Produced by | Description |
|------|-------------|-------------|
| `activations.pt` | Extraction | Train/val/test activations + labels |
| `harmful_activations.pt` | Extraction | Harmful activations (for reward model) |
| `generator.pt` | Module 3 | Trained CVAE/MLP perturbation generator |
| `reward_model.pt` | Module 3 | Trained reward model |
| `generator_fw_*.pt` | Module 3 | Frank-Wolfe iteration generators |
| `corruption_results.pt` | Module 4 | Perturbations + model responses |
| `judged_results.pt` | Module 5 | Scored results with ASR metrics |
| `judge_metrics.json` | Module 5 | Summary metrics (for paper tables) |
| `clustering_results.pt` | Module 6 | Cluster assignments + centers |
| `detector.pt` | Module 7 | Final detector (thresholds + PCA) |
| `pca_plots/` | PCA Analysis | Visualization plots |

## Recommended Workflows

### Full run (all layers)

```bash
python src/Extraction.py --layers 10 15 20 25 --n-samples 10000 --n-harmful 5000

for layer in 10 15 20 25; do
    python src/module3_perturbation_generator.py --layer $layer --phase all --epsilon 0.15 --recalibration-interval 1000
    python src/pipeline.py --layer $layer --modules all
done
```

### Quick test (single layer, small scale)

```bash
python src/Extraction.py --layers 20 --n-samples 1000 --n-harmful 500
python src/module3_perturbation_generator.py --layer 20 --phase all --epsilon 0.15 --rl-steps 1000
python src/module4_corruption.py --layer 20 --K 10 --max-passages 20
python src/module5_judge.py --layer 20 --method heuristic
python src/module6_clustering.py --layer 20
python src/module7_detector.py --layer 20
```

### Visualization only

```bash
python src/Extraction.py --layers 10 15 20 25 --n-samples 10000 --n-harmful 5000
python src/pca_analysis.py --layer all
```


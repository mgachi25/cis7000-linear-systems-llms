# Fine-Tuning GPT-4.1-nano for Linear Systems

Fine-tuning `gpt-4.1-nano` to solve systems of linear equations, comparing Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

## Overview

The task is to classify and solve 2–3 variable linear systems that may have a unique solution, infinitely many solutions, or no solution. We generate a synthetic dataset with reasoning traces, establish a baseline, and fine-tune using SFT and DPO via Azure OpenAI.

## Notebooks

| Notebook | Description |
|---|---|
| `CIS5270_Linear_Systems_V2_(REASONING).ipynb` | Dataset generation, baseline evaluation, SFT |
| `CIS5270_Linear_Systems_V2_DPO.ipynb` | DPO preference data generation and full-scale Run A |
| `CIS5270_DPO_Extra_Runs.ipynb` | DPO ablation runs B, C, D on smaller data |

---

## Notebook 1: `CIS5270_Linear_Systems_V2_(REASONING).ipynb`

### Dataset Generation

Generates three splits of synthetic linear systems with balanced solution types (unique ~67%, no-solution ~17%, infinite ~17%):

| Split | Size |
|---|---|
| Training | 10,000 |
| Validation | 3,000 |
| Evaluation | 6,000 |

Each record is formatted as an OpenAI-compatible chat message with step-by-step reasoning traces. Outputs: `training.jsonl`, `validation.jsonl`, `eval.jsonl`.

### Baseline Evaluation

Evaluates untuned `gpt-4.1-nano` on 1,000 held-out examples before any fine-tuning.

| Outcome | Count | % |
|---|---|---|
| Correct | 210 | 21.0% |
| Wrong type | 780 | 78.0% |
| Unparseable | 10 | 1.0% |

The primary failure mode is misclassifying the solution type. Visualizations include an accuracy overview, confusion matrix, response length distribution, and per-variable-count breakdown.

### Supervised Fine-Tuning (SFT)

Uploads training data to Azure OpenAI and launches an SFT job on `gpt-4.1-nano-2025-04-14`.

- **Training examples:** 10,000
- **Tokens billed:** 2.6M
- **Final training loss:** 0.0077
- Azure model quality evaluation: passed

The fine-tuned model is deployed and evaluated against the held-out eval set.

### Utilities

- **System generators:** `generate_unique`, `generate_infinite`, `generate_no_solution` for all three solution types
- **Grading functions:** `parse_model_output`, `compute_reward`, `evaluate_batch`
- **Eval harness:** `run_eval` — parallel evaluation with checkpointing and fail-fast logic; `summarize` and `assert_eval_healthy` for result aggregation

## Notebook 2: `CIS5270_Linear_Systems_V2_DPO.ipynb`

### Preference Data Generation

Generates DPO preference pairs from the SFT training data by corrupting gold answers in one of four ways:

| Corruption Strategy | Weight |
|---|---|
| Wrong value (perturb a variable assignment) | 50% |
| Wrong type (swap solution-type label) | 25% |
| Format violation (drop prefix, shuffle order, omit variable) | 15% |
| Trace error (flip final ANSWER line only) | 10% |

Outputs: `dpo_training.jsonl`, `dpo_validation.jsonl`.

### DPO Training (Run A)

Uploads preference pairs to Azure OpenAI and launches a DPO job on `gpt-4.1-nano-2025-04-14` from base.

- **Training examples:** 10,000
- **Hyperparameters:** `n_epochs=1`, `batch_size=1`, `learning_rate_multiplier=1.0`
- Training completed successfully but deployment was blocked by a false positive Azure content safety flag (Hate/Fairness). Best validation loss: 0.4225 at step 10,000.

### Utilities

- **Eval harness:** `run_eval`, `summarize`, `assert_eval_healthy` — same interface as Notebook 1
- **Training curves:** plotted from live job events; can be run while training is in progress

---

## Notebook 3: `CIS5270_DPO_Extra_Runs.ipynb`

### DPO Ablation Runs (B, C, D)

Three parallel DPO experiments on a 2,500-example subset with `batch_size=4`, varying one factor at a time:

| Run | Description | LR Multiplier |
|---|---|---|
| Run B | Default mixed corruption, 2.5k examples | 1.0 |
| Run C | Trace-error-only corruption, 2.5k examples | 1.0 |
| Run D | Default mixed corruption, 2.5k examples | 2.0 |

### Results

| Run | Overall | Unique | No Solution | Infinite |
|---|---|---|---|---|
| Run B | 7.5% | 0.0% | 21.0% | 24.0% |
| Run C | 13.5% | 1.4% | 34.1% | 41.3% |
| Run D | 0.4% | 0.0% | 2.4% | 0.0% |

Eval outputs saved to `eval_summary_dpo_B.json`, `eval_summary_dpo_C.json`, `eval_summary_dpo_D.json`.


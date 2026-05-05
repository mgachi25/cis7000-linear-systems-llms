# Fine-Tuning GPT-4.1-nano for Linear Systems

Fine-tuning `gpt-4.1-nano` to solve systems of linear equations, comparing Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

## Overview

The task is to classify and solve 2–3 variable linear systems that may have a unique solution, infinitely many solutions, or no solution. We generate a synthetic dataset with reasoning traces, establish a baseline, and fine-tune using SFT and DPO via Azure OpenAI.

## Notebooks

| Notebook | Description |
|---|---|
| `CIS5270_Linear_Systems_V2_(REASONING).ipynb` | Dataset generation, baseline evaluation, SFT |
| *(DPO notebook — to be added)* | |

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

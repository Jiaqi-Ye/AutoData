# AutoData Agent

[Open in Colab](https://colab.research.google.com/github/Jiaqi-Ye/AutoData/blob/main/notebooks/AutoData_Colab_Demo.ipynb)

AutoData Agent is a research prototype for evaluation-driven data generation and data mixture optimization for medical LLM fine-tuning. The project asks whether a model-aware data agent can generate targeted SFT data and optimize training mixtures more effectively than static, uniform, or manually designed data strategies.

The first prototype uses MedMCQA and five medical domains:

- Anatomy
- Pharmacology
- Pathology
- Microbiology
- Physiology

The default base model is `Qwen/Qwen2.5-1.5B-Instruct`. The default generation model is configured as `Qwen/Qwen2.5-7B-Instruct`, but smoke mode uses a mock generator and does not download large models.

## Pipeline

```text
evaluate -> diagnose -> plan -> generate -> verify -> mix -> fine-tune -> re-evaluate
```

The project focuses on data-side automation. To keep comparisons meaningful, the model, training method, learning rate, batch size, training steps, and evaluation split should stay fixed whenever possible.

## Repository Layout

```text
autodata-agent/
  configs/                 YAML configs for smoke, prototype, and full runs
  notebooks/               Colab demo notebook
  autodata/                Reusable pipeline modules
  scripts/                 Command-line entry points
  tests/                   Unit tests
  outputs/                 Local smoke outputs, ignored by git except .gitkeep
```

## Local Setup

Create an environment and install the lightweight dependencies:

```bash
pip install -r requirements.txt
```

For the first milestone, run:

```bash
python scripts/run_full_loop.py --config configs/smoke_colab.yaml
pytest
```

Smoke mode should finish without a GPU, API keys, MedMCQA downloads, or Qwen model downloads.

## Google Colab Setup

1. Open `notebooks/AutoData_Colab_Demo.ipynb`.
2. Run the notebook from top to bottom. It clones `https://github.com/Jiaqi-Ye/AutoData.git` into `/content/autodata-agent`.
3. Start with:

```python
RUN_MODE = "smoke"
USE_REAL_MODEL = False
USE_REAL_TRAINING = False
USE_MOCK_GENERATION = True
RUN_STRATEGY_COMPARISON = False
```

When the smoke run is working, switch to a real-model prototype:

```python
RUN_MODE = "prototype"
USE_REAL_MODEL = True
USE_REAL_TRAINING = True
USE_MOCK_GENERATION = True
RUN_STRATEGY_COMPARISON = False
```

For the first real Qwen experiment, keeping `USE_MOCK_GENERATION=True` is cheaper because it tests real MedMCQA loading, real Qwen evaluation, QLoRA training, verification, mixture construction, and reporting without loading the larger generation model. When ready for real synthetic generation, set:

```python
USE_MOCK_GENERATION = False
GENERATION_PROVIDER = "local_hf"
```

## Output Location

Colab configs save to:

```text
/content/autodata_outputs
```

Each run creates:

```text
/content/autodata_outputs/runs/<timestamp>/
```

Expected files:

```text
config.yaml
evaluation_base.json
diagnosis.json
data_plan.json
generated_samples.jsonl
verified_samples.jsonl
rejected_samples.jsonl
verification_report.json
mixture_train.jsonl
mixture_report.json
training_report.json
evaluation_after.json
next_round_recommendation.json
round_summary.json
```

Local smoke mode uses `outputs/runs/<timestamp>/` inside the repository.

## Run Modes

### Smoke

Purpose: verify that the whole pipeline works.

- 5 domains
- 5 eval examples per domain
- 25 total generated samples by default
- mock data, mock evaluation, mock generation
- training skipped
- no large downloads

```bash
python scripts/run_full_loop.py --config configs/smoke_colab.yaml
```

To compare the first three planned strategies in one command:

```bash
python scripts/run_strategy_comparison.py --config configs/smoke_colab.yaml
```

### Prototype

Purpose: medium-size Colab GPU experiment.

- 50-100 eval examples per domain
- around 1000 synthetic samples
- real Qwen2.5-1.5B evaluation enabled
- QLoRA training enabled
- generation can remain mock or switch to local generation

```bash
python scripts/run_full_loop.py --config configs/prototype_colab.yaml
```

### Full

Purpose: larger experiment if GPU memory allows.

- larger evaluation set
- 1000+ synthetic samples per round
- real QLoRA training
- optional multi-round extension

```bash
python scripts/run_full_loop.py --config configs/full_colab.yaml
```

## Enabling Real Qwen Evaluation

Set these values in YAML or in the Colab notebook before saving a temporary config:

```yaml
dataset:
  use_mock_data: false
models:
  use_real_model: true
```

The evaluator uses Hugging Face Transformers and asks the model to answer with one of `A/B/C/D`.

## Enabling QLoRA Training

Set:

```yaml
training:
  enabled: true
  dry_run: false
  method: "qlora"
```

Default Colab-safe settings use 4-bit loading, small per-device batch size, gradient accumulation, limited sequence length, and LoRA target modules for Qwen-style transformer blocks.

## Switching Mixture Strategies

Change:

```yaml
mixture:
  strategy: "uniform"
```

Available strategies:

- `uniform`
- `weakness_based`
- `agent_guided`
- `efficiency_aware`

If `efficiency_aware` has no previous-round efficiency history, it falls back to weakness-based allocation.

## What Is Mock In Smoke Mode

- MedMCQA examples are deterministic mock examples.
- Evaluation is deterministic mock evaluation.
- Generation uses a leakage-safe mock generator.
- Training is skipped.
- The post-training result is simulated so the full reporting path can be checked.

## What Is Real In Prototype/Full Mode

- MedMCQA loading uses `datasets`.
- Evaluation can run the real Qwen base model through Transformers.
- Training can run QLoRA through Transformers, PEFT, bitsandbytes, and TRL.
- Verification, planning, mixture optimization, artifact writing, and metric reporting are real code paths in every mode.
- Each run now writes `next_round_recommendation.json`, including per-domain gain, learning efficiency, and suggested next-round focus domains.

## Current Limitations

- Agent-guided planning is currently a structured heuristic, not an API-backed LLM planner.
- Medical factual verification is rule-based; future work should add optional expert-model review.
- Real generation is available as a local Hugging Face provider, while API generation is a placeholder.
- Full multi-round orchestration still needs a dedicated runner; single-round runs now emit the next-round recommendation needed for that extension.

## Future Work

- Add stronger medical quality verification.
- Add multi-round experiment orchestration.
- Add statistical aggregation across repeated seeds for uniform vs weakness-based vs agent-guided mixtures.
- Add richer MedMCQA subject normalization and cached processed datasets.
- Add optional vLLM provider for larger generation runs, without making it a default dependency.

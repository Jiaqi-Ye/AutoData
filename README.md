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

To generate synthetic data with the OpenAI API instead of a local generator, use:

```python
USE_MOCK_GENERATION = False
GENERATION_PROVIDER = "openai"
GENERATION_API_MODEL = "gpt-4o-mini"
GENERATION_API_BATCH_SIZE = 5
TOTAL_SYNTHETIC_BUDGET = 100
```

The OpenAI provider generates `GENERATION_API_BATCH_SIZE` samples per API call and writes standard `SFTSample` rows before the same rule verifier and medical critic stages.

### Optional Medical Critic

The rule verifier checks format, leakage, duplicates, option structure, and shallow answer-option consistency. For real generated medical MCQs, add the optional LLM medical critic after rule verification and before mixture/training:

```yaml
medical_critic:
  enabled: true
  provider: "openai"
  model: "gpt-4o-mini"
  fail_closed: true
  abort_on_error: true
  preflight_check: true
```

Set `OPENAI_API_KEY` in the environment before using `provider: "openai"`. To use the current local Qwen/Transformers path instead:

```yaml
medical_critic:
  enabled: true
  provider: "local_hf"
  local_model: "Qwen/Qwen2.5-1.5B-Instruct"
  fail_closed: true
  abort_on_error: true
```

The critic rejects samples with medical errors, multiple plausible answers, answer-letter mismatches, contradictory explanations, or low-value template-like wording. With OpenAI, `preflight_check: true` fails fast if the API is unreachable, and `abort_on_error: true` stops the run if a critic call fails so infrastructure errors are not mistaken for rejected data. Keep it disabled in smoke mode.

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

When `medical_critic.enabled=true`, the final `verified_samples.jsonl` and `verification_report.json` represent samples that passed both stages. The run also writes `rule_verified_samples.jsonl`, `rule_rejected_samples.jsonl`, `rule_verification_report.json`, `medical_critic_rejected_samples.jsonl`, and `medical_critic_report.json` for stage-by-stage inspection.

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

### Colab PyArrow Restart Note

If Colab raises a `pyarrow.lib.IpcReadOptions size changed` error after switching to real MedMCQA loading, rerun the dependency cell. The notebook reinstalls `pyarrow` and `datasets` together, restarts the runtime once, and then you should run the notebook from the top again.

If PEFT raises `Found an incompatible version of torchao` while loading a LoRA adapter for re-evaluation, install `torchao==0.16.0` and restart the runtime. The notebook includes this dependency in the real-run install cell.

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
- Medical factual verification now supports an optional LLM critic, but its quality depends on the critic model and prompt calibration.
- Real generation is available through local Hugging Face models or the OpenAI API provider.
- Full multi-round orchestration still needs a dedicated runner; single-round runs now emit the next-round recommendation needed for that extension.

## Future Work

- Calibrate the LLM medical critic with held-out human review.
- Add multi-round experiment orchestration.
- Add statistical aggregation across repeated seeds for uniform vs weakness-based vs agent-guided mixtures.
- Add richer MedMCQA subject normalization and cached processed datasets.
- Add optional vLLM provider for larger generation runs, without making it a default dependency.

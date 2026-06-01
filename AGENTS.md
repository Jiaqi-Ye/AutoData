# AGENTS.md

Guidance for future Codex work on AutoData Agent.

- Keep modules inspectable.
- Avoid hidden global state.
- Preserve held-out evaluation leakage checks.
- Run `pytest` after modifying core modules.
- Preserve smoke mode.
- Do not make tests download large models.
- Prefer clear research code over over-engineered production code.
- Document assumptions in `README.md`.
- Keep Colab compatibility.
- Do not require vLLM by default.
- Do not hardcode secrets or local absolute paths.

Project-specific notes:

- The default path is the smoke loop: `python scripts/run_full_loop.py --config configs/smoke_colab.yaml`.
- Smoke mode must run without a GPU, without API keys, and without downloading Qwen models.
- The held-out MedMCQA evaluation examples must not be used as generation prompts or source material.
- Real model evaluation and QLoRA training should remain opt-in through YAML config flags.
- Keep scripts small wrappers around reusable `autodata/` modules.


"""Optional LoRA/QLoRA fine-tuning wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from autodata.data.schemas import SFTSample, TrainingRunResult


def format_sft_sample(sample: SFTSample) -> str:
    return f"### Instruction\n{sample.instruction}\n\n### Response\n{sample.response}"


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config.get("training", {})

    def train(self, samples: List[SFTSample], run_dir: str | Path) -> TrainingRunResult:
        enabled = bool(self.training_config.get("enabled", False))
        dry_run = bool(self.training_config.get("dry_run", False))
        method = str(self.training_config.get("method", "qlora"))
        checkpoint_dir = Path(run_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if not enabled:
            return TrainingRunResult(
                enabled=False,
                method=method,
                status="skipped",
                output_dir=str(checkpoint_dir),
                num_train_samples=len(samples),
                details={"reason": "training.enabled=false"},
            )
        if dry_run:
            return TrainingRunResult(
                enabled=True,
                method=method,
                status="dry_run",
                output_dir=str(checkpoint_dir),
                num_train_samples=len(samples),
                details={"reason": "dry_run=true"},
            )
        if not samples:
            return TrainingRunResult(
                enabled=True,
                method=method,
                status="skipped",
                output_dir=str(checkpoint_dir),
                num_train_samples=0,
                details={"reason": "no training samples after verification/mixture"},
            )
        if method.lower() not in {"qlora", "lora"}:
            raise ValueError(f"Unsupported training method: {method}")
        return self._train_qlora(samples, checkpoint_dir)

    def _train_qlora(self, samples: List[SFTSample], checkpoint_dir: Path) -> TrainingRunResult:
        try:
            from datasets import Dataset
            from peft import prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from trl import SFTTrainer

            from autodata.training.qlora_utils import build_bnb_config, build_lora_config, choose_compute_dtype
        except ImportError as exc:
            raise RuntimeError(
                "Install the Colab dependencies from requirements.txt to run QLoRA training."
            ) from exc

        import torch

        base_model = self.config.get("models", {}).get("base_model")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = build_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        dataset = Dataset.from_list([{"text": format_sft_sample(sample)} for sample in samples])
        args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            per_device_train_batch_size=int(self.training_config.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(self.training_config.get("gradient_accumulation_steps", 8)),
            learning_rate=float(self.training_config.get("learning_rate", 2e-4)),
            max_steps=int(self.training_config.get("max_steps", 50)),
            logging_steps=5,
            save_steps=max(int(self.training_config.get("max_steps", 50)), 1),
            fp16=choose_compute_dtype() == torch.float16,
            bf16=choose_compute_dtype() == torch.bfloat16,
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=build_lora_config(self.training_config),
            dataset_text_field="text",
            max_seq_length=int(self.training_config.get("max_seq_length", 512)),
            tokenizer=tokenizer,
            args=args,
        )
        train_output = trainer.train()
        trainer.model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        return TrainingRunResult(
            enabled=True,
            method=str(self.training_config.get("method", "qlora")),
            status="completed",
            output_dir=str(checkpoint_dir),
            num_train_samples=len(samples),
            details={"train_loss": getattr(train_output, "training_loss", None)},
        )


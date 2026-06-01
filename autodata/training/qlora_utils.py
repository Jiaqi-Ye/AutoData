"""QLoRA helper functions kept separate from the smoke path."""

from __future__ import annotations

from typing import Any, Dict


def choose_compute_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_lora_config(training_config: Dict[str, Any]):
    from peft import LoraConfig

    return LoraConfig(
        r=int(training_config.get("lora_r", 8)),
        lora_alpha=int(training_config.get("lora_alpha", 16)),
        lora_dropout=float(training_config.get("lora_dropout", 0.05)),
        target_modules=list(training_config.get("target_modules", [])),
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=choose_compute_dtype(),
    )


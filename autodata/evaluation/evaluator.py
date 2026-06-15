"""Mock and Hugging Face evaluation paths."""

from __future__ import annotations

from collections import defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List

from autodata.data.schemas import DomainMetrics, EvaluationResult, MedMCQAExample
from autodata.evaluation.answer_parser import parse_answer
from autodata.evaluation.metrics import variance


def format_mcq_prompt(example: MedMCQAExample) -> str:
    return (
        "Answer the following medical multiple-choice question. "
        "Return only one letter: A, B, C, or D.\n\n"
        f"Question: {example.question}\n"
        f"A. {example.options['A']}\n"
        f"B. {example.options['B']}\n"
        f"C. {example.options['C']}\n"
        f"D. {example.options['D']}\n"
        "Answer:"
    )


def _version_tuple(value: str) -> tuple[int, ...]:
    numeric = []
    for part in value.replace("-", ".").split("."):
        if not part.isdigit():
            break
        numeric.append(int(part))
    return tuple(numeric)


def disable_incompatible_torchao_for_peft(minimum_version: str = "0.16.0") -> bool:
    """Disable PEFT's torchao dispatcher when Colab ships an old torchao.

    QLoRA here uses bitsandbytes, not torchao. Some Colab images include an
    old torchao package, and recent PEFT raises during adapter loading when it
    detects that incompatible version. Returning False from PEFT's torchao
    availability checks lets PEFT continue to its other LoRA dispatchers.
    """
    try:
        torchao_version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        return False

    if _version_tuple(torchao_version) >= _version_tuple(minimum_version):
        return False

    patched = False
    try:
        import peft.import_utils as peft_import_utils

        peft_import_utils.is_torchao_available = lambda: False
        patched = True
    except Exception:
        pass

    try:
        import peft.tuners.lora.torchao as peft_lora_torchao

        peft_lora_torchao.is_torchao_available = lambda: False
        patched = True
    except Exception:
        pass

    return patched


class Evaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = config.get("models", {})
        self.training = config.get("training", {})

    def evaluate(
        self,
        examples: List[MedMCQAExample],
        model_path: str | None = None,
        phase: str = "base",
    ) -> EvaluationResult:
        if self.models.get("use_real_model", False):
            return self._evaluate_with_transformers(examples, model_path=model_path, phase=phase)
        return self._mock_evaluate(examples, phase=phase)

    def _mock_prediction(self, example: MedMCQAExample, index: int, phase: str) -> str:
        domain_bias = {
            "Anatomy": 0,
            "Pharmacology": 1,
            "Pathology": 2,
            "Microbiology": 1,
            "Physiology": 0,
        }.get(example.domain, 0)
        answer_order = ["A", "B", "C", "D"]
        correct_index = answer_order.index(example.correct_answer)
        should_be_correct = (index + domain_bias) % 3 != 0
        if phase != "base":
            should_be_correct = (index + domain_bias) % 5 != 0
        if should_be_correct:
            return example.correct_answer
        return answer_order[(correct_index + 1) % 4]

    def _mock_evaluate(self, examples: List[MedMCQAExample], phase: str) -> EvaluationResult:
        predictions = []
        for index, example in enumerate(examples):
            prediction = self._mock_prediction(example, index, phase)
            predictions.append(
                {
                    "id": example.id,
                    "domain": example.domain,
                    "prediction": prediction,
                    "correct_answer": example.correct_answer,
                    "is_correct": prediction == example.correct_answer,
                }
            )
        return build_evaluation_result(
            predictions=predictions,
            examples=examples,
            model_name=self.models.get("base_model", "mock-model"),
            mode="mock",
        )

    def _evaluate_with_transformers(
        self,
        examples: List[MedMCQAExample],
        model_path: str | None,
        phase: str,
    ) -> EvaluationResult:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for real model evaluation.") from exc

        base_model = self.models.get("base_model")
        adapter_path = Path(model_path) if model_path else None
        is_adapter = adapter_path is not None and (adapter_path / "adapter_config.json").exists()
        model_name = base_model if is_adapter else (model_path or base_model)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if is_adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Install peft to evaluate a LoRA/QLoRA adapter.") from exc
            disable_incompatible_torchao_for_peft()
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        predictions = []
        max_new_tokens = int(self.config.get("evaluation", {}).get("max_new_tokens", 8))
        for example in examples:
            prompt = format_mcq_prompt(example)
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            parsed = parse_answer(decoded) or "A"
            predictions.append(
                {
                    "id": example.id,
                    "domain": example.domain,
                    "prediction": parsed,
                    "raw_output": decoded,
                    "correct_answer": example.correct_answer,
                    "is_correct": parsed == example.correct_answer,
                }
            )
        return build_evaluation_result(predictions, examples, model_name=model_name, mode="transformers")


def build_evaluation_result(
    predictions: List[Dict[str, Any]],
    examples: List[MedMCQAExample],
    model_name: str,
    mode: str,
) -> EvaluationResult:
    totals: Dict[str, int] = defaultdict(int)
    corrects: Dict[str, int] = defaultdict(int)
    for prediction in predictions:
        domain = prediction["domain"]
        totals[domain] += 1
        if prediction["is_correct"]:
            corrects[domain] += 1

    per_domain = {
        domain: DomainMetrics.from_counts(domain, corrects[domain], totals[domain])
        for domain in sorted(totals)
    }
    correct_count = sum(corrects.values())
    sample_count = len(examples)
    accuracies = {domain: metrics.accuracy for domain, metrics in per_domain.items()}
    weakest = min(accuracies, key=accuracies.get) if accuracies else ""
    strongest = max(accuracies, key=accuracies.get) if accuracies else ""
    return EvaluationResult(
        overall_accuracy=correct_count / sample_count if sample_count else 0.0,
        per_domain=per_domain,
        sample_count=sample_count,
        correct_count=correct_count,
        incorrect_count=sample_count - correct_count,
        weakest_domain=weakest,
        strongest_domain=strongest,
        variance_across_domains=variance(accuracies.values()),
        model_name=model_name,
        mode=mode,
        predictions=predictions,
    )

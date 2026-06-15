"""Optional LLM-based medical quality critic for generated MCQ SFT data."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Tuple

from autodata.data.schemas import SFTSample, VerificationResult, to_jsonable
from autodata.verification.verifier import OPTION_TEXT_PATTERN, RESPONSE_PREFIX_PATTERN


CRITIC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "accepted": {"type": "boolean"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "checks": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "single_best_answer": {"type": "boolean"},
                "correct_letter_matches_option": {"type": "boolean"},
                "explanation_medically_sound": {"type": "boolean"},
                "not_overly_template_or_duplicate": {"type": "boolean"},
            },
            "required": [
                "single_best_answer",
                "correct_letter_matches_option",
                "explanation_medically_sound",
                "not_overly_template_or_duplicate",
            ],
        },
        "rationale": {"type": "string"},
        "suggested_correct_answer": {"type": "string"},
    },
    "required": [
        "accepted",
        "reason",
        "confidence",
        "issues",
        "checks",
        "rationale",
        "suggested_correct_answer",
    ],
}


SYSTEM_PROMPT = """You are a strict medical multiple-choice question reviewer.
Review synthetic SFT samples before they are used for medical fine-tuning.
Accept only if the sample is medically sound, has exactly one best answer, the answer letter matches the option text, and the explanation is consistent with the selected answer.
Reject vague, template-like, factually wrong, contradictory, or multi-answer questions.
Return JSON only, matching the requested schema."""


class MedicalCriticProvider(ABC):
    """Provider interface for medical critic models."""

    name = "base"

    @abstractmethod
    def review(self, sample: SFTSample) -> Dict[str, Any]:
        raise NotImplementedError


class MockMedicalCriticProvider(MedicalCriticProvider):
    """Deterministic no-op critic used for smoke tests."""

    name = "mock"

    def review(self, sample: SFTSample) -> Dict[str, Any]:
        return {
            "accepted": True,
            "reason": "accepted",
            "confidence": 1.0,
            "issues": [],
            "checks": {
                "single_best_answer": True,
                "correct_letter_matches_option": True,
                "explanation_medically_sound": True,
                "not_overly_template_or_duplicate": True,
            },
            "rationale": "Mock critic accepts samples without an LLM call.",
            "suggested_correct_answer": "",
        }


class OpenAIMedicalCriticProvider(MedicalCriticProvider):
    """Medical critic backed by the OpenAI Python SDK."""

    name = "openai"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = str(config.get("model", "gpt-4o-mini"))
        self.temperature = float(config.get("temperature", 0.0))
        self.timeout = float(config.get("timeout_seconds", 60))
        api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
        if not os.environ.get(api_key_env):
            raise RuntimeError(f"medical_critic.provider=openai requires ${api_key_env}.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional openai package for medical_critic.provider=openai.") from exc
        self.client = OpenAI(api_key=os.environ.get(api_key_env), timeout=self.timeout)

    def review(self, sample: SFTSample) -> Dict[str, Any]:
        prompt = build_critic_prompt(sample)
        try:
            text = self._responses_create(prompt)
        except (AttributeError, TypeError):
            text = self._chat_completions_create(prompt)
        return load_json_object(text)

    def _responses_create(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "medical_critic_decision",
                    "strict": True,
                    "schema": CRITIC_SCHEMA,
                }
            },
            "temperature": self.temperature,
            "store": False,
        }
        try:
            response = self.client.responses.create(**kwargs)
        except TypeError:
            kwargs.pop("temperature", None)
            response = self.client.responses.create(**kwargs)
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise ValueError("OpenAI Responses API returned no output_text.")
        return str(output_text)

    def _chat_completions_create(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "medical_critic_decision",
                    "strict": True,
                    "schema": CRITIC_SCHEMA,
                },
            },
            "temperature": self.temperature,
            "store": False,
        }
        try:
            response = self.client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs.pop("store", None)
            response = self.client.chat.completions.create(**kwargs)
        return str(response.choices[0].message.content or "")


class LocalHFMedicalCriticProvider(MedicalCriticProvider):
    """Medical critic backed by a local Hugging Face causal LM."""

    name = "local_hf"

    def __init__(self, config: Dict[str, Any], root_config: Dict[str, Any]) -> None:
        self.config = config
        self.root_config = root_config
        self.model_name = str(
            config.get("local_model")
            or config.get("model")
            or root_config.get("generation", {}).get("local_model")
            or root_config.get("models", {}).get("generation_model")
            or root_config.get("models", {}).get("base_model")
        )
        self.max_new_tokens = int(config.get("max_new_tokens", 384))
        self._tokenizer = None
        self._model = None

    def review(self, sample: SFTSample) -> Dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for medical_critic.provider=local_hf.") from exc

        tokenizer, model = self._load_model()
        rendered_prompt = render_critic_prompt(tokenizer, build_critic_prompt(sample))
        inputs = tokenizer(rendered_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return load_json_object(decoded)

    def _load_model(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for medical_critic.provider=local_hf.") from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model


class LLMMedicalVerifier:
    """Apply an optional LLM medical quality check after rule verification."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.root_config = config
        self.config = dict(config.get("medical_critic", config.get("llm_medical_verifier", {})))
        self.enabled = bool(self.config.get("enabled", False))
        self.provider_name = str(self.config.get("provider", "mock")).lower()
        self.fail_closed = bool(self.config.get("fail_closed", True))
        self.provider: MedicalCriticProvider | None = None
        if self.enabled:
            self.provider = get_medical_critic_provider(self.provider_name, self.config, config)

    def verify(self, samples: Iterable[SFTSample]) -> VerificationResult:
        sample_list = list(samples)
        if not self.enabled:
            return VerificationResult(
                accepted=sample_list,
                rejected=[],
                report={
                    "enabled": False,
                    "provider": self.provider_name,
                    "input_count": len(sample_list),
                    "accepted_count": len(sample_list),
                    "rejected_count": 0,
                },
            )

        if self.provider is None:
            raise RuntimeError("Medical critic is enabled but no provider was initialized.")

        accepted: List[SFTSample] = []
        rejected: List[Dict[str, Any]] = []
        accepted_by_domain: Counter[str] = Counter()
        rejected_by_domain: Counter[str] = Counter()
        rejection_counter: Counter[str] = Counter()

        for sample in sample_list:
            try:
                decision = normalize_decision(self.provider.review(sample))
            except Exception as exc:
                if not self.fail_closed:
                    decision = normalize_decision(
                        {
                            "accepted": True,
                            "reason": "critic_error_fail_open",
                            "confidence": 0.0,
                            "issues": [str(exc)],
                            "checks": {},
                            "rationale": "Critic failed and fail_closed is false.",
                            "suggested_correct_answer": "",
                        }
                    )
                else:
                    decision = normalize_decision(
                        {
                            "accepted": False,
                            "reason": "critic_error",
                            "confidence": 0.0,
                            "issues": [str(exc)],
                            "checks": {},
                            "rationale": "Critic failed and fail_closed is true.",
                            "suggested_correct_answer": "",
                        }
                    )

            reviewed_sample = attach_decision(sample, decision)
            if decision["accepted"]:
                accepted.append(reviewed_sample)
                accepted_by_domain[sample.domain] += 1
            else:
                reason = medical_reason(decision)
                rejected_by_domain[sample.domain] += 1
                rejection_counter[reason] += 1
                rejected.append(
                    {
                        "reason": reason,
                        "sample": to_jsonable(reviewed_sample),
                        "medical_critic": decision,
                    }
                )

        report = {
            "enabled": True,
            "provider": self.provider_name,
            "model": critic_model_name(self.provider),
            "input_count": len(sample_list),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "accepted_by_domain": dict(accepted_by_domain),
            "rejected_by_domain": dict(rejected_by_domain),
            "rejection_reasons": dict(rejection_counter),
            "fail_closed": self.fail_closed,
        }
        return VerificationResult(accepted=accepted, rejected=rejected, report=report)


def apply_medical_critic(
    config: Dict[str, Any], rule_result: VerificationResult
) -> Tuple[VerificationResult, VerificationResult | None]:
    """Return final verification after optional medical critic review."""

    critic = LLMMedicalVerifier(config)
    if not critic.enabled:
        return rule_result, None

    critic_result = critic.verify(rule_result.accepted)
    final_rejected = list(rule_result.rejected) + list(critic_result.rejected)
    final_report = merge_verification_reports(rule_result.report, critic_result.report, final_rejected)
    return VerificationResult(accepted=critic_result.accepted, rejected=final_rejected, report=final_report), critic_result


def merge_verification_reports(
    rule_report: Dict[str, Any], critic_report: Dict[str, Any], rejected: List[Dict[str, Any]]
) -> Dict[str, Any]:
    accepted_by_domain = dict(critic_report.get("accepted_by_domain", {}))
    rejected_by_domain: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    for row in rejected:
        sample = row.get("sample", {})
        domain = str(sample.get("domain", "unknown"))
        rejected_by_domain[domain] += 1
        rejection_reasons[str(row.get("reason", "unknown"))] += 1
    return {
        "accepted_count": int(critic_report.get("accepted_count", 0)),
        "rejected_count": len(rejected),
        "accepted_by_domain": accepted_by_domain,
        "rejected_by_domain": dict(rejected_by_domain),
        "rejection_reasons": dict(rejection_reasons),
        "rule_verification": rule_report,
        "medical_critic": critic_report,
        "near_duplicate_threshold": rule_report.get("near_duplicate_threshold"),
        "question_duplicate_threshold": rule_report.get("question_duplicate_threshold"),
        "leakage_threshold": rule_report.get("leakage_threshold"),
    }


def get_medical_critic_provider(
    provider_name: str, config: Dict[str, Any], root_config: Dict[str, Any]
) -> MedicalCriticProvider:
    if provider_name == "mock":
        return MockMedicalCriticProvider()
    if provider_name == "openai":
        return OpenAIMedicalCriticProvider(config)
    if provider_name in {"local_hf", "qwen", "local_qwen"}:
        return LocalHFMedicalCriticProvider(config, root_config)
    raise ValueError(f"Unknown medical critic provider: {provider_name}")


def build_critic_prompt(sample: SFTSample) -> str:
    answer = response_answer(sample.response) or ""
    options = extract_options(sample.instruction)
    option_lines = "\n".join(f"{label}. {options.get(label, '')}" for label in ("A", "B", "C", "D"))
    return f"""Review this synthetic medical MCQ.

Domain: {sample.domain}

Question and options:
{sample.instruction}

Parsed options:
{option_lines}

Model response:
{sample.response}

Parsed answer letter: {answer}

Decision rules:
- accepted must be false if the selected letter does not match the medically correct option.
- accepted must be false if more than one option can reasonably be correct.
- accepted must be false if the explanation contradicts the answer or contains a medical error.
- accepted must be false if the item is too generic, vague, or template-like to be useful for training.
- reason should be accepted, wrong_answer, answer_option_mismatch, multiple_correct_options, ambiguous_question, medically_unsound_explanation, low_quality_template, or critic_error.
"""


def render_critic_prompt(tokenizer: Any, prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return SYSTEM_PROMPT + "\n\n" + prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return SYSTEM_PROMPT + "\n\n" + prompt


def load_json_object(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("critic returned empty output")
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("critic returned output without a JSON object")


def normalize_decision(raw: Dict[str, Any]) -> Dict[str, Any]:
    checks = raw.get("checks") if isinstance(raw.get("checks"), dict) else {}
    accepted = coerce_bool(raw.get("accepted", False))
    reason = str(raw.get("reason") or ("accepted" if accepted else "critic_rejected")).strip()
    issues = raw.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    return {
        "accepted": accepted,
        "reason": normalize_reason(reason, accepted),
        "confidence": confidence,
        "issues": [str(issue) for issue in issues if str(issue).strip()],
        "checks": {
            "single_best_answer": coerce_bool(checks.get("single_best_answer", accepted)),
            "correct_letter_matches_option": coerce_bool(checks.get("correct_letter_matches_option", accepted)),
            "explanation_medically_sound": coerce_bool(checks.get("explanation_medically_sound", accepted)),
            "not_overly_template_or_duplicate": coerce_bool(
                checks.get("not_overly_template_or_duplicate", accepted)
            ),
        },
        "rationale": str(raw.get("rationale", "")).strip(),
        "suggested_correct_answer": str(raw.get("suggested_correct_answer", "")).strip().upper(),
    }


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "accept", "accepted"}
    return bool(value)


def normalize_reason(reason: str, accepted: bool) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_")
    if accepted:
        return "accepted"
    return normalized or "medical_critic_rejected"


def medical_reason(decision: Dict[str, Any]) -> str:
    reason = normalize_reason(str(decision.get("reason", "")), accepted=False)
    if reason.startswith("medical_critic_"):
        return reason
    return f"medical_critic_{reason}"


def attach_decision(sample: SFTSample, decision: Dict[str, Any]) -> SFTSample:
    metadata = dict(sample.metadata)
    metadata["medical_critic"] = decision
    return replace(sample, metadata=metadata)


def extract_options(instruction: str) -> Dict[str, str]:
    return {
        match.group(1).upper(): match.group(2).strip()
        for match in OPTION_TEXT_PATTERN.finditer(instruction)
        if match.group(2).strip()
    }


def response_answer(response: str) -> str | None:
    match = RESPONSE_PREFIX_PATTERN.search(str(response or ""))
    if not match:
        return None
    return match.group(1).upper()


def critic_model_name(provider: MedicalCriticProvider | None) -> str:
    if provider is None:
        return "none"
    return str(getattr(provider, "model", getattr(provider, "model_name", provider.name)))

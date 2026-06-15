"""Generation provider interfaces."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from autodata.data.schemas import GenerationRequest, SFTSample
from autodata.generation.prompts import build_generation_prompt, build_mock_instruction, build_mock_response


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


OPTION_LINE_PATTERN = re.compile(r"(?im)^\s*([ABCD])[\.\)]\s+\S+")
OPTION_VALUE_PATTERN = re.compile(r"^\s*([ABCD])[\.\)]\s*(.+?)\s*$", re.IGNORECASE)
ANSWER_PREFIX_PATTERN = re.compile(r"^\s*The correct answer is ([ABCD])\.", re.IGNORECASE)
ANSWER_VALUE_PATTERN = re.compile(r"(?:correct answer is|answer\s*:|option)\s*([ABCD])\b", re.IGNORECASE)
LEADING_ANSWER_PATTERN = re.compile(r"^\s*([ABCD])(?:[\.\)]|\b)", re.IGNORECASE)


class GenerationProvider(ABC):
    name = "base"

    @abstractmethod
    def generate(self, request: GenerationRequest, config: Dict[str, Any]) -> List[SFTSample]:
        raise NotImplementedError


class MockGenerationProvider(GenerationProvider):
    name = "mock"

    def generate(self, request: GenerationRequest, config: Dict[str, Any]) -> List[SFTSample]:
        samples: List[SFTSample] = []
        model_name = config.get("models", {}).get("generation_model", "mock-generator")
        for index in range(request.num_samples):
            samples.append(
                SFTSample(
                    id=f"{request.round_id}-{_slug(request.domain)}-{index}",
                    domain=request.domain,
                    instruction=build_mock_instruction(request.domain, index),
                    response=build_mock_response(request.domain, index),
                    source="mock",
                    generation_model=model_name,
                    round_id=request.round_id,
                    metadata={
                        "data_type": request.data_type,
                        "planner_reason": request.reason,
                        "sample_index": index,
                    },
                )
            )
        return samples


class LocalHFGenerationProvider(GenerationProvider):
    name = "local_hf"

    def __init__(self) -> None:
        self._model_name: str | None = None
        self._tokenizer = None
        self._model = None

    def _load_model(self, model_name: str):
        if self._model is not None and self._tokenizer is not None and self._model_name == model_name:
            return self._tokenizer, self._model

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for local_hf generation.") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model.eval()

        self._model_name = model_name
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def generate(self, request: GenerationRequest, config: Dict[str, Any]) -> List[SFTSample]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for local_hf generation.") from exc

        model_name = config.get("generation", {}).get("local_model") or config.get("models", {}).get("generation_model")
        tokenizer, model = self._load_model(model_name)
        max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 512))
        samples: List[SFTSample] = []
        for index in range(request.num_samples):
            prompt = build_generation_prompt(request, index)
            rendered_prompt = _render_generation_prompt(tokenizer, prompt)
            inputs = tokenizer(rendered_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            parsed, parse_status = _parse_generated_json_with_status(decoded)
            parse_error = "none" if parsed else parse_status
            samples.append(
                SFTSample(
                    id=f"{request.round_id}-{_slug(request.domain)}-{index}",
                    domain=str(parsed.get("domain", request.domain)),
                    instruction=str(parsed.get("instruction", prompt)),
                    response=str(parsed.get("response", decoded)),
                    source="local_hf",
                    generation_model=model_name,
                    round_id=request.round_id,
                    metadata={
                        "raw_output": decoded,
                        "sample_index": index,
                        "parse_error": parse_error,
                        "json_parse_status": parse_status,
                        "prompt_format": "chat_template" if rendered_prompt != prompt else "plain",
                    },
                )
            )
        return samples


class APIGenerationProvider(GenerationProvider):
    name = "api_placeholder"

    def generate(self, request: GenerationRequest, config: Dict[str, Any]) -> List[SFTSample]:
        raise RuntimeError(
            "API generation is intentionally a placeholder. Use mock or local_hf, "
            "or implement this provider with your preferred API client."
        )


class StrongLocalGenerationProvider(LocalHFGenerationProvider):
    name = "strong_local"


def _render_generation_prompt(tokenizer, prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [
        {
            "role": "system",
            "content": "You generate strict JSON only. Never use markdown fences or explanatory prose.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def _parse_generated_json(text: str) -> Dict[str, Any]:
    parsed, _ = _parse_generated_json_with_status(text)
    return parsed


def _parse_generated_json_with_status(text: str) -> tuple[Dict[str, Any], str]:
    payload, parse_status = _load_json_payload(text)
    if not payload:
        return {}, parse_status
    parsed = _normalize_generated_payload(payload)
    if not parsed.get("instruction") or not parsed.get("response"):
        return {}, "missing_required_fields"
    return parsed, parse_status


def _load_json_payload(text: str) -> tuple[Dict[str, Any], str]:
    stripped = str(text or "").strip()
    if not stripped:
        return {}, "empty_generation"
    decoder = json.JSONDecoder()

    try:
        parsed, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        parsed = None
    else:
        if isinstance(parsed, dict):
            remainder = stripped[end:].strip()
            return parsed, "pure_json" if not remainder else "extracted_json"

    for start, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, "extracted_json"
    return {}, "invalid_json"


def _normalize_generated_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    domain = str(payload.get("domain", "")).strip()
    instruction = str(
        payload.get("instruction")
        or payload.get("question")
        or payload.get("stem")
        or payload.get("prompt")
        or ""
    ).strip()
    options = _normalize_options(
        payload.get("options")
        or payload.get("choices")
        or payload.get("answer_choices")
        or payload.get("answers")
    )
    if options and _instruction_option_labels(instruction) != {"A", "B", "C", "D"}:
        question = instruction or "Question:"
        if not question.lower().startswith("question:"):
            question = f"Question: {question}"
        option_lines = [f"{label}. {options[label]}" for label in ("A", "B", "C", "D") if label in options]
        instruction = question.rstrip() + "\n" + "\n".join(option_lines)

    response = str(payload.get("response") or payload.get("answer_explanation") or "").strip()
    answer = _extract_answer_letter(
        payload.get("answer")
        or payload.get("correct_answer")
        or payload.get("correct_option")
        or payload.get("correct")
    )
    if answer is None:
        answer = _response_answer(response)
    explanation = str(
        payload.get("explanation")
        or payload.get("rationale")
        or payload.get("reasoning")
        or ""
    ).strip()
    if _response_answer(response) is None and answer is not None:
        detail = explanation or response
        response = f"The correct answer is {answer}."
        if detail:
            response += f" Explanation: {detail}"

    allowed = {"domain", "instruction", "response"}
    normalized = {"domain": domain, "instruction": instruction, "response": response}
    return {key: value for key, value in normalized.items() if key in allowed and value}


def _normalize_options(raw_options: Any) -> Dict[str, str]:
    options: Dict[str, str] = {}
    if isinstance(raw_options, dict):
        for label in ("A", "B", "C", "D"):
            if label in raw_options:
                options[label] = str(raw_options[label]).strip()
            elif label.lower() in raw_options:
                options[label] = str(raw_options[label.lower()]).strip()
        return {label: _strip_option_label(text) for label, text in options.items() if text}

    if not isinstance(raw_options, list):
        return {}

    for index, item in enumerate(raw_options):
        fallback_label = chr(ord("A") + index) if index < 4 else None
        label: str | None = None
        text = ""
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            label_value = item.get("label") or item.get("letter") or item.get("key") or item.get("option")
            if label_value:
                label = _extract_answer_letter(label_value)
            text_value = item.get("text") or item.get("content") or item.get("value") or item.get("answer")
            text = str(text_value or "").strip()
        if not text:
            continue
        match = OPTION_VALUE_PATTERN.match(text)
        if match:
            label = match.group(1).upper()
            text = match.group(2).strip()
        if label is None:
            label = fallback_label
        if label in {"A", "B", "C", "D"}:
            options[label] = _strip_option_label(text)
    return {label: text for label, text in options.items() if text}


def _strip_option_label(text: str) -> str:
    match = OPTION_VALUE_PATTERN.match(str(text).strip())
    if match:
        return match.group(2).strip()
    return str(text).strip()


def _instruction_option_labels(instruction: str) -> set[str]:
    return {match.group(1).upper() for match in OPTION_LINE_PATTERN.finditer(instruction)}


def _response_answer(response: str) -> str | None:
    match = ANSWER_PREFIX_PATTERN.search(str(response or ""))
    if not match:
        return None
    return match.group(1).upper()


def _extract_answer_letter(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    for pattern in (ANSWER_VALUE_PATTERN, LEADING_ANSWER_PATTERN):
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def get_generation_provider(config: Dict[str, Any]) -> GenerationProvider:
    generation = config.get("generation", {})
    if generation.get("use_mock_generation", False):
        return MockGenerationProvider()
    provider_name = str(generation.get("provider", "mock")).lower()
    if provider_name == "mock":
        return MockGenerationProvider()
    if provider_name in {"local_hf", "small_local"}:
        return LocalHFGenerationProvider()
    if provider_name in {"api", "api_placeholder"}:
        return APIGenerationProvider()
    if provider_name in {"strong_local", "optional_stronger_local"}:
        return StrongLocalGenerationProvider()
    raise ValueError(f"Unknown generation provider: {provider_name}")

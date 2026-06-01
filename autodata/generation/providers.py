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

    def generate(self, request: GenerationRequest, config: Dict[str, Any]) -> List[SFTSample]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for local_hf generation.") from exc

        model_name = config.get("generation", {}).get("local_model") or config.get("models", {}).get("generation_model")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model.eval()
        max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 512))
        samples: List[SFTSample] = []
        for index in range(request.num_samples):
            prompt = build_generation_prompt(request, index)
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
            decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            parsed = _parse_generated_json(decoded)
            samples.append(
                SFTSample(
                    id=f"{request.round_id}-{_slug(request.domain)}-{index}",
                    domain=str(parsed.get("domain", request.domain)),
                    instruction=str(parsed.get("instruction", prompt)),
                    response=str(parsed.get("response", decoded)),
                    source="local_hf",
                    generation_model=model_name,
                    round_id=request.round_id,
                    metadata={"raw_output": decoded, "sample_index": index},
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


def _parse_generated_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


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

"""OpenAI-backed planning agent for synthetic data allocation."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import replace
from typing import Any, Dict, Iterable

from autodata.data.schemas import DataPlan, DiagnosisResult, DomainPlan, EvaluationResult


class LLMPlanningError(RuntimeError):
    """Raised when an LLM planning provider cannot produce a usable plan."""


class LLMDataPlanningAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.planning_config = config.get("planning", {})

    def create_plan(
        self,
        evaluation: EvaluationResult,
        diagnosis: DiagnosisResult,
        fallback_plan: DataPlan,
    ) -> DataPlan:
        provider = str(self.planning_config.get("agent_provider", self.planning_config.get("provider", "openai"))).lower()
        if provider == "mock":
            return _enrich_fallback_plan(fallback_plan, "Mock LLM agent plan")
        if provider != "openai":
            raise ValueError(f"Unknown planning agent provider: {provider}")

        try:
            payload = self._call_openai(evaluation, diagnosis, fallback_plan)
            return normalize_agent_plan(
                payload,
                fallback_plan,
                evaluation,
                diagnosis,
                min_samples_per_domain=int(self.planning_config.get("min_samples_per_domain", 0)),
            )
        except Exception as exc:
            if bool(self.planning_config.get("abort_on_error", True)):
                raise LLMPlanningError(f"LLM planning agent failed: {type(exc).__name__}: {exc}") from exc
            return _enrich_fallback_plan(fallback_plan, f"LLM agent fallback after {type(exc).__name__}")

    def _call_openai(
        self,
        evaluation: EvaluationResult,
        diagnosis: DiagnosisResult,
        fallback_plan: DataPlan,
    ) -> Dict[str, Any]:
        api_key_env = str(self.planning_config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"planning.agent_provider=openai requires ${api_key_env}.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional openai package for planning.agent_provider=openai.") from exc

        client = OpenAI(api_key=api_key, timeout=float(self.planning_config.get("timeout_seconds", 120)))
        model = str(self.planning_config.get("agent_model", self.planning_config.get("model", "gpt-4o-mini")))
        temperature = float(self.planning_config.get("temperature", 0.1))
        max_tokens = int(self.planning_config.get("max_tokens", 1800))
        prompt = build_agent_prompt(evaluation, diagnosis, fallback_plan, self.planning_config)
        kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AutoData planning agent for medical MCQ synthetic data. "
                        "Return JSON only. Preserve the requested total budget exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as first_exc:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("temperature", None)
            try:
                response = client.chat.completions.create(**fallback_kwargs)
            except Exception as second_exc:
                raise RuntimeError(
                    f"OpenAI planning failed: {type(first_exc).__name__}: {first_exc}; "
                    f"retry_without_temperature={type(second_exc).__name__}: {second_exc}"
                ) from second_exc
        return load_json_object(str(response.choices[0].message.content or ""))


def build_agent_prompt(
    evaluation: EvaluationResult,
    diagnosis: DiagnosisResult,
    fallback_plan: DataPlan,
    planning_config: Dict[str, Any],
) -> str:
    domain_rows = []
    for domain, metrics in evaluation.per_domain.items():
        fallback = fallback_plan.plan.get(domain)
        domain_rows.append(
            {
                "domain": domain,
                "accuracy": round(metrics.accuracy, 4),
                "num_eval_samples": metrics.num_samples,
                "fallback_num_samples": fallback.num_samples if fallback else 0,
                "fallback_data_type": fallback.data_type if fallback else "MCQ with explanation",
                "diagnosis": diagnosis.rationale_by_domain.get(domain, ""),
            }
        )
    min_samples = int(planning_config.get("min_samples_per_domain", 0))
    return (
        "Create a synthetic-data generation plan for one AutoData round.\n"
        "The generator will use GPT to create MedMCQA-style SFT examples and only a rule verifier will run after generation.\n"
        "Prioritize domains where the base model is weak, but keep enough coverage for strong domains to avoid forgetting.\n"
        "Use fact-first, single-best-answer question styles. Avoid vague prompts, list-all-that-apply stems, and multiple-correct options.\n\n"
        f"Total synthetic budget: {fallback_plan.total_budget}\n"
        f"Minimum samples per domain: {min_samples}\n"
        f"Weak domains: {diagnosis.weak_domains}\n"
        f"Stable domains: {diagnosis.stable_domains}\n"
        f"Risk-prone domains: {diagnosis.risk_prone_domains}\n"
        f"Overall baseline accuracy: {evaluation.overall_accuracy:.4f}\n"
        f"Domain evidence JSON:\n{json.dumps(domain_rows, indent=2)}\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "strategy": "llm_agent",\n'
        '  "rationale": "brief global rationale",\n'
        '  "plan": {\n'
        '    "Domain": {\n'
        '      "num_samples": 123,\n'
        '      "data_type": "case-style MCQ with explanation",\n'
        '      "reason": "why this allocation helps",\n'
        '      "generation_guidance": "specific guidance for GPT generation in this domain"\n'
        "    }\n"
        "  }\n"
        "}\n"
        "Hard constraints:\n"
        "- Include every target domain exactly once.\n"
        "- The num_samples values must sum exactly to the total synthetic budget.\n"
        "- Every domain must receive at least the minimum samples per domain.\n"
        "- generation_guidance should name concrete subtopics and failure modes to avoid."
    )


def normalize_agent_plan(
    payload: Dict[str, Any],
    fallback_plan: DataPlan,
    evaluation: EvaluationResult,
    diagnosis: DiagnosisResult,
    min_samples_per_domain: int = 0,
) -> DataPlan:
    raw_plan = payload.get("plan") or payload.get("allocations") or payload.get("domains")
    if not isinstance(raw_plan, dict):
        raise ValueError("LLM agent response must contain a plan object")

    raw_counts: Dict[str, int] = {}
    raw_items: Dict[str, Dict[str, Any]] = {}
    for domain in fallback_plan.plan:
        matched_key = _match_domain_key(domain, raw_plan.keys())
        item = raw_plan.get(matched_key) if matched_key is not None else None
        if isinstance(item, dict):
            raw_items[domain] = item
            raw_counts[domain] = _coerce_int(item.get("num_samples", item.get("samples", item.get("count"))), 0)
        else:
            raw_items[domain] = {}
            raw_counts[domain] = _coerce_int(item, 0)

    allocations = repair_allocations(
        raw_counts=raw_counts,
        fallback_counts={domain: item.num_samples for domain, item in fallback_plan.plan.items()},
        total_budget=fallback_plan.total_budget,
        min_samples_per_domain=min_samples_per_domain,
        domains=fallback_plan.plan.keys(),
    )

    global_rationale = str(payload.get("rationale", "")).strip()
    plan: Dict[str, DomainPlan] = {}
    for domain, fallback_item in fallback_plan.plan.items():
        raw_item = raw_items.get(domain, {})
        data_type = str(raw_item.get("data_type") or fallback_item.data_type).strip()
        reason = str(raw_item.get("reason") or fallback_item.reason).strip()
        guidance = str(raw_item.get("generation_guidance") or raw_item.get("guidance") or "").strip()
        if global_rationale and global_rationale not in reason:
            reason = f"LLM agent: {reason} Global rationale: {global_rationale}"
        elif not reason.startswith("LLM agent"):
            reason = f"LLM agent: {reason}"
        if not guidance:
            guidance = default_generation_guidance(domain, evaluation, diagnosis)
        plan[domain] = DomainPlan(
            domain=domain,
            num_samples=allocations[domain],
            data_type=data_type,
            reason=reason,
            generation_guidance=guidance,
        )
    return DataPlan(total_budget=fallback_plan.total_budget, strategy="llm_agent", plan=plan)


def repair_allocations(
    raw_counts: Dict[str, int],
    fallback_counts: Dict[str, int],
    total_budget: int,
    min_samples_per_domain: int,
    domains: Iterable[str],
) -> Dict[str, int]:
    domain_list = list(domains)
    if total_budget < min_samples_per_domain * len(domain_list):
        raise ValueError("total_budget is too small for the requested minimum per domain")

    counts = {
        domain: max(int(raw_counts.get(domain) or 0), 0)
        for domain in domain_list
    }
    if sum(counts.values()) <= 0:
        counts = {domain: max(int(fallback_counts.get(domain, 0)), 1) for domain in domain_list}

    base = {domain: min_samples_per_domain for domain in domain_list}
    remaining = total_budget - sum(base.values())
    if remaining == 0:
        return base

    weights = {domain: max(counts.get(domain, 0), fallback_counts.get(domain, 0), 1) for domain in domain_list}
    weight_sum = sum(weights.values())
    fractional = {domain: remaining * weights[domain] / weight_sum for domain in domain_list}
    allocations = {domain: base[domain] + math.floor(fractional[domain]) for domain in domain_list}
    shortfall = total_budget - sum(allocations.values())
    ranked = sorted(domain_list, key=lambda domain: fractional[domain] - math.floor(fractional[domain]), reverse=True)
    for domain in ranked[:shortfall]:
        allocations[domain] += 1
    return allocations


def default_generation_guidance(domain: str, evaluation: EvaluationResult, diagnosis: DiagnosisResult) -> str:
    metrics = evaluation.per_domain.get(domain)
    accuracy = metrics.accuracy if metrics else 0.0
    if domain in diagnosis.weak_domains:
        return (
            f"Create high-yield {domain} MCQs around weak baseline performance "
            f"({accuracy:.2f} accuracy). Use clear single-best-answer stems and one defensible correct option."
        )
    if domain in diagnosis.risk_prone_domains:
        return (
            f"Create preservation {domain} MCQs that reinforce core mechanisms without overfitting. "
            "Keep distractors plausible but clearly wrong."
        )
    return (
        f"Create balanced {domain} MCQs with concise explanations, concrete facts, and no ambiguous option overlap."
    )


def load_json_object(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("LLM agent returned empty output")
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
    raise ValueError("LLM agent output did not contain a JSON object")


def _enrich_fallback_plan(fallback_plan: DataPlan, label: str) -> DataPlan:
    plan = {
        domain: replace(
            item,
            reason=f"{label}: {item.reason}",
            generation_guidance=item.generation_guidance
            or "Generate fact-first, single-best-answer medical MCQs with non-overlapping distractors.",
        )
        for domain, item in fallback_plan.plan.items()
    }
    return DataPlan(total_budget=fallback_plan.total_budget, strategy="llm_agent", plan=plan)


def _match_domain_key(domain: str, keys: Iterable[Any]) -> Any | None:
    normalized = _normalize_key(domain)
    for key in keys:
        if _normalize_key(str(key)) == normalized:
            return key
    return None


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default

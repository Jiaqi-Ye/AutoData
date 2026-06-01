"""Data generation planning strategies."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from autodata.config import get_target_domains
from autodata.data.schemas import DataPlan, DiagnosisResult, DomainPlan, EvaluationResult


DATA_TYPES = {
    "Anatomy": "concept explanation + MCQ",
    "Pharmacology": "MCQ-style with mechanism explanation",
    "Pathology": "case-style MCQ with explanation",
    "Microbiology": "organism and treatment MCQ",
    "Physiology": "mechanism-focused review MCQ",
}


class DataPlanner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def create_plan(self, evaluation: EvaluationResult, diagnosis: DiagnosisResult) -> DataPlan:
        strategy = str(self.config.get("planning", {}).get("strategy", "weakness_based"))
        total_budget = int(self.config.get("generation", {}).get("total_budget", 0))
        min_samples = int(self.config.get("planning", {}).get("min_samples_per_domain", 0))
        target_domains = get_target_domains(self.config)
        return build_data_plan(
            strategy=strategy,
            total_budget=total_budget,
            min_samples_per_domain=min_samples,
            target_domains=target_domains,
            evaluation=evaluation,
            diagnosis=diagnosis,
        )


def build_data_plan(
    strategy: str,
    total_budget: int,
    min_samples_per_domain: int,
    target_domains: Iterable[str],
    evaluation: EvaluationResult,
    diagnosis: DiagnosisResult,
) -> DataPlan:
    domains = list(target_domains)
    _validate_plan_inputs(domains, total_budget, min_samples_per_domain, evaluation)

    if strategy == "uniform":
        allocations = _allocate_uniform(domains, total_budget)
    elif strategy == "agent_guided":
        allocations = _allocate_weakness_based(domains, total_budget, min_samples_per_domain, evaluation, diagnosis)
    elif strategy == "weakness_based":
        allocations = _allocate_weakness_based(domains, total_budget, min_samples_per_domain, evaluation, diagnosis)
    else:
        raise ValueError(f"Unknown planning strategy: {strategy}")

    plan: Dict[str, DomainPlan] = {}
    for domain in domains:
        accuracy = evaluation.per_domain[domain].accuracy
        if strategy == "uniform":
            reason = "Uniform baseline allocation for controlled comparison."
        elif domain in diagnosis.weak_domains:
            reason = f"Lower baseline accuracy ({accuracy:.2f}) gives this domain higher expected benefit."
        elif domain in diagnosis.risk_prone_domains:
            reason = f"Preservation allocation for a strong or risk-prone domain ({accuracy:.2f} accuracy)."
        else:
            reason = f"Middle-performance domain receives balanced coverage ({accuracy:.2f} accuracy)."
        if strategy == "agent_guided":
            reason = "Heuristic agent-guided plan: " + reason
        plan[domain] = DomainPlan(
            domain=domain,
            num_samples=allocations[domain],
            data_type=DATA_TYPES.get(domain, "MCQ-style with explanation"),
            reason=reason,
        )

    data_plan = DataPlan(total_budget=total_budget, strategy=strategy, plan=plan)
    if data_plan.allocation_sum() != total_budget:
        raise AssertionError("planner allocation must equal total budget")
    return data_plan


def _validate_plan_inputs(
    domains: List[str],
    total_budget: int,
    min_samples_per_domain: int,
    evaluation: EvaluationResult,
) -> None:
    if total_budget < 0:
        raise ValueError("total_budget cannot be negative")
    if not domains:
        raise ValueError("target_domains cannot be empty")
    if min_samples_per_domain < 0:
        raise ValueError("min_samples_per_domain cannot be negative")
    if total_budget < min_samples_per_domain * len(domains):
        raise ValueError("total_budget is too small for the requested minimum per domain")
    invalid = [domain for domain in domains if domain not in evaluation.per_domain]
    if invalid:
        raise ValueError(f"Target domains missing from evaluation: {invalid}")


def _allocate_uniform(domains: List[str], total_budget: int) -> Dict[str, int]:
    base = total_budget // len(domains)
    remainder = total_budget % len(domains)
    return {domain: base + (1 if index < remainder else 0) for index, domain in enumerate(domains)}


def _allocate_weakness_based(
    domains: List[str],
    total_budget: int,
    min_samples_per_domain: int,
    evaluation: EvaluationResult,
    diagnosis: DiagnosisResult,
) -> Dict[str, int]:
    allocations = {domain: min_samples_per_domain for domain in domains}
    remaining = total_budget - min_samples_per_domain * len(domains)
    if remaining == 0:
        return allocations

    weights: Dict[str, float] = {}
    for domain in domains:
        accuracy = evaluation.per_domain[domain].accuracy
        weight = max(1.0 - accuracy, 0.05)
        if domain in diagnosis.weak_domains:
            weight += 0.35
        if domain in diagnosis.risk_prone_domains:
            weight += 0.10
        weights[domain] = weight

    weight_sum = sum(weights.values())
    fractional = {domain: remaining * weights[domain] / weight_sum for domain in domains}
    for domain in domains:
        allocations[domain] += math.floor(fractional[domain])
    shortfall = total_budget - sum(allocations.values())
    ranked_remainders = sorted(domains, key=lambda domain: fractional[domain] - math.floor(fractional[domain]), reverse=True)
    for domain in ranked_remainders[:shortfall]:
        allocations[domain] += 1
    return allocations


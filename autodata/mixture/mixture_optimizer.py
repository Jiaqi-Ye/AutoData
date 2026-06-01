"""Training mixture construction strategies."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from autodata.config import get_target_domains
from autodata.data.schemas import DataPlan, EvaluationResult, MixturePlan, SFTSample


class MixtureOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def optimize(
        self,
        verified_samples: List[SFTSample],
        data_plan: DataPlan,
        evaluation: EvaluationResult,
        previous_efficiency: Optional[Dict[str, float]] = None,
    ) -> MixturePlan:
        strategy = str(self.config.get("mixture", {}).get("strategy", "weakness_based"))
        target_domains = get_target_domains(self.config)
        return build_mixture(
            strategy=strategy,
            verified_samples=verified_samples,
            data_plan=data_plan,
            evaluation=evaluation,
            target_domains=target_domains,
            previous_efficiency=previous_efficiency,
        )


def build_mixture(
    strategy: str,
    verified_samples: List[SFTSample],
    data_plan: DataPlan,
    evaluation: EvaluationResult,
    target_domains: Iterable[str],
    previous_efficiency: Optional[Dict[str, float]] = None,
) -> MixturePlan:
    by_domain = _group_by_domain(verified_samples)
    domains = list(target_domains)
    available_counts = {domain: len(by_domain.get(domain, [])) for domain in domains}
    final_budget = len(verified_samples)

    if strategy == "uniform":
        targets = _uniform_targets(available_counts, final_budget)
        reasons = {domain: "Uniform mixture target for controlled baseline." for domain in domains}
    elif strategy == "agent_guided":
        targets = _plan_targets(data_plan, available_counts)
        reasons = {domain: data_plan.plan[domain].reason for domain in domains if domain in data_plan.plan}
    elif strategy == "efficiency_aware":
        if previous_efficiency:
            targets = _efficiency_targets(available_counts, previous_efficiency, final_budget)
            reasons = {
                domain: f"Allocated by previous learning efficiency {previous_efficiency.get(domain, 0.0):.6f}."
                for domain in domains
            }
        else:
            targets = _weakness_targets(available_counts, evaluation, final_budget)
            reasons = {domain: "Efficiency history unavailable; falling back to weakness-based mixture." for domain in domains}
    elif strategy == "weakness_based":
        targets = _plan_targets(data_plan, available_counts)
        reasons = {domain: "Following weakness-based data plan while capping by verified availability." for domain in domains}
    else:
        raise ValueError(f"Unknown mixture strategy: {strategy}")

    selected: List[SFTSample] = []
    for domain in domains:
        selected.extend(by_domain.get(domain, [])[: targets.get(domain, 0)])
    distribution = {domain: sum(1 for sample in selected if sample.domain == domain) for domain in domains}
    dropped = len(verified_samples) - len(selected)
    return MixturePlan(
        strategy=strategy,
        samples=selected,
        domain_distribution=distribution,
        dropped_samples=dropped,
        reasons=reasons,
    )


def _group_by_domain(samples: Iterable[SFTSample]) -> Dict[str, List[SFTSample]]:
    grouped: Dict[str, List[SFTSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.domain].append(sample)
    return dict(grouped)


def _uniform_targets(available_counts: Dict[str, int], final_budget: int) -> Dict[str, int]:
    non_empty = [domain for domain, count in available_counts.items() if count > 0]
    if not non_empty:
        return {domain: 0 for domain in available_counts}
    per_domain = min(min(available_counts[domain] for domain in non_empty), final_budget // len(non_empty))
    targets = {domain: (per_domain if domain in non_empty else 0) for domain in available_counts}
    remaining = min(final_budget, sum(available_counts.values())) - sum(targets.values())
    for domain in non_empty:
        if remaining <= 0:
            break
        room = available_counts[domain] - targets[domain]
        add = min(room, remaining)
        targets[domain] += add
        remaining -= add
    return targets


def _plan_targets(data_plan: DataPlan, available_counts: Dict[str, int]) -> Dict[str, int]:
    targets = {}
    for domain, count in available_counts.items():
        planned = data_plan.plan.get(domain).num_samples if domain in data_plan.plan else 0
        targets[domain] = min(count, planned)
    return targets


def _weakness_targets(
    available_counts: Dict[str, int],
    evaluation: EvaluationResult,
    final_budget: int,
) -> Dict[str, int]:
    weights = {
        domain: max(1.0 - evaluation.per_domain[domain].accuracy, 0.05)
        for domain in available_counts
        if domain in evaluation.per_domain and available_counts[domain] > 0
    }
    return _weighted_targets(available_counts, weights, final_budget)


def _efficiency_targets(
    available_counts: Dict[str, int],
    previous_efficiency: Dict[str, float],
    final_budget: int,
) -> Dict[str, int]:
    weights = {
        domain: max(previous_efficiency.get(domain, 0.0), 0.0001)
        for domain, count in available_counts.items()
        if count > 0
    }
    return _weighted_targets(available_counts, weights, final_budget)


def _weighted_targets(
    available_counts: Dict[str, int],
    weights: Dict[str, float],
    final_budget: int,
) -> Dict[str, int]:
    targets = {domain: 0 for domain in available_counts}
    if not weights:
        return targets
    usable_budget = min(final_budget, sum(available_counts.values()))
    weight_sum = sum(weights.values())
    fractional = {domain: usable_budget * weights[domain] / weight_sum for domain in weights}
    for domain in weights:
        targets[domain] = min(available_counts[domain], math.floor(fractional[domain]))
    shortfall = usable_budget - sum(targets.values())
    ranked = sorted(weights, key=lambda domain: fractional[domain] - math.floor(fractional[domain]), reverse=True)
    while shortfall > 0:
        changed = False
        for domain in ranked:
            if targets[domain] < available_counts[domain]:
                targets[domain] += 1
                shortfall -= 1
                changed = True
                if shortfall == 0:
                    break
        if not changed:
            break
    return targets


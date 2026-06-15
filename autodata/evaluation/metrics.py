"""Model-level and system-level metrics."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List

from autodata.data.schemas import EvaluationResult, SFTSample


def variance(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    center = mean(values_list)
    return sum((value - center) ** 2 for value in values_list) / len(values_list)


def per_domain_gains(before: EvaluationResult, after: EvaluationResult) -> Dict[str, float]:
    domains = set(before.per_domain).intersection(after.per_domain)
    return {
        domain: after.per_domain[domain].accuracy - before.per_domain[domain].accuracy
        for domain in sorted(domains)
    }


def compare_evaluations(before: EvaluationResult, after: EvaluationResult) -> Dict[str, Any]:
    base_acc = before.overall_accuracy
    after_acc = after.overall_accuracy
    gains = per_domain_gains(before, after)
    weakest = before.weakest_domain
    strongest = before.strongest_domain
    return {
        "overall_gain": after_acc - base_acc,
        "average_domain_gain": mean(gains.values()) if gains else 0.0,
        "weakest_domain_improvement": gains.get(weakest, 0.0),
        "strong_domain_drop": min(gains.get(strongest, 0.0), 0.0),
        "variance_before": before.variance_across_domains,
        "variance_after": after.variance_across_domains,
        "per_domain_gain": gains,
    }


def count_by_domain(samples: Iterable[SFTSample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.domain] = counts.get(sample.domain, 0) + 1
    return counts


def system_metrics(
    generated_samples: List[SFTSample],
    accepted_samples: List[SFTSample],
    rejected_samples: List[dict],
    mixture_samples: List[SFTSample],
    evaluation_gain: float,
) -> Dict[str, object]:
    generated_count = len(generated_samples)
    accepted_count = len(accepted_samples)
    rejected_count = len(rejected_samples)
    return {
        "generated_data_count_per_domain": count_by_domain(generated_samples),
        "accepted_data_count_per_domain": count_by_domain(accepted_samples),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "accepted_ratio": accepted_count / generated_count if generated_count else 0.0,
        "duplicate_or_rejected_rate": rejected_count / generated_count if generated_count else 0.0,
        "mixture_distribution": count_by_domain(mixture_samples),
        "improvement_per_generated_sample": evaluation_gain / generated_count if generated_count else 0.0,
    }


def learning_efficiency_by_domain(
    before: EvaluationResult,
    after: EvaluationResult,
    mixture_samples: List[SFTSample],
) -> Dict[str, float]:
    """Estimate gain per selected training sample for each domain."""
    gains = per_domain_gains(before, after)
    training_counts = count_by_domain(mixture_samples)
    efficiencies: Dict[str, float] = {}
    for domain in sorted(before.per_domain):
        count = training_counts.get(domain, 0)
        efficiencies[domain] = gains.get(domain, 0.0) / count if count else 0.0
    return efficiencies


def next_round_recommendation(
    before: EvaluationResult,
    after: EvaluationResult,
    mixture_samples: List[SFTSample],
) -> Dict[str, Any]:
    """Create an auditable next-round data-planning hint from observed gains."""
    gains = per_domain_gains(before, after)
    efficiencies = learning_efficiency_by_domain(before, after, mixture_samples)
    focus_scores: Dict[str, float] = {}
    reasons: Dict[str, str] = {}

    for domain, metrics in after.per_domain.items():
        gain = gains.get(domain, 0.0)
        efficiency = efficiencies.get(domain, 0.0)
        weakness_pressure = max(1.0 - metrics.accuracy, 0.0)
        regression_pressure = 0.25 if gain < 0 else 0.0
        efficiency_bonus = max(efficiency, 0.0)
        focus_scores[domain] = weakness_pressure + regression_pressure + efficiency_bonus

        reason_parts = [f"after accuracy {metrics.accuracy:.3f}", f"gain {gain:.3f}"]
        if gain < 0:
            reason_parts.append("possible forgetting")
        if efficiency > 0:
            reason_parts.append(f"efficiency {efficiency:.6f}")
        reasons[domain] = "; ".join(reason_parts)

    ranked_domains = sorted(focus_scores, key=focus_scores.get, reverse=True)
    return {
        "recommended_focus_domains": ranked_domains,
        "focus_scores": focus_scores,
        "per_domain_gain": gains,
        "learning_efficiency_by_domain": efficiencies,
        "rationale_by_domain": reasons,
    }

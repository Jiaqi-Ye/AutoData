"""Model-level and system-level metrics."""

from __future__ import annotations

from statistics import mean
from typing import Dict, Iterable, List

from autodata.data.schemas import EvaluationResult, SFTSample


def variance(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    center = mean(values_list)
    return sum((value - center) ** 2 for value in values_list) / len(values_list)


def compare_evaluations(before: EvaluationResult, after: EvaluationResult) -> Dict[str, float]:
    base_acc = before.overall_accuracy
    after_acc = after.overall_accuracy
    domains = set(before.per_domain).intersection(after.per_domain)
    gains = {
        domain: after.per_domain[domain].accuracy - before.per_domain[domain].accuracy
        for domain in domains
    }
    weakest = before.weakest_domain
    strongest = before.strongest_domain
    return {
        "overall_gain": after_acc - base_acc,
        "average_domain_gain": mean(gains.values()) if gains else 0.0,
        "weakest_domain_improvement": gains.get(weakest, 0.0),
        "strong_domain_drop": min(gains.get(strongest, 0.0), 0.0),
        "variance_before": before.variance_across_domains,
        "variance_after": after.variance_across_domains,
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


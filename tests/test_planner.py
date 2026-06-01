import pytest

from autodata.data.schemas import DiagnosisResult, DomainMetrics, EvaluationResult
from autodata.planning.data_planner import build_data_plan


DOMAINS = ["Anatomy", "Pharmacology", "Pathology", "Microbiology", "Physiology"]


def make_evaluation() -> EvaluationResult:
    per_domain = {
        "Anatomy": DomainMetrics.from_counts("Anatomy", 3, 10),
        "Pharmacology": DomainMetrics.from_counts("Pharmacology", 2, 10),
        "Pathology": DomainMetrics.from_counts("Pathology", 5, 10),
        "Microbiology": DomainMetrics.from_counts("Microbiology", 6, 10),
        "Physiology": DomainMetrics.from_counts("Physiology", 8, 10),
    }
    return EvaluationResult(
        overall_accuracy=0.48,
        per_domain=per_domain,
        sample_count=50,
        correct_count=24,
        incorrect_count=26,
        weakest_domain="Pharmacology",
        strongest_domain="Physiology",
        variance_across_domains=0.04,
        model_name="mock",
    )


def make_diagnosis() -> DiagnosisResult:
    return DiagnosisResult(
        weak_domains=["Pharmacology", "Anatomy"],
        stable_domains=["Physiology"],
        risk_prone_domains=["Physiology"],
        rationale_by_domain={domain: "test" for domain in DOMAINS},
        summary="test",
    )


def test_planner_budget_sum_and_minimums():
    plan = build_data_plan(
        strategy="weakness_based",
        total_budget=100,
        min_samples_per_domain=5,
        target_domains=DOMAINS,
        evaluation=make_evaluation(),
        diagnosis=make_diagnosis(),
    )
    assert plan.allocation_sum() == 100
    assert all(item.num_samples >= 5 for item in plan.plan.values())
    assert plan.plan["Pharmacology"].num_samples >= plan.plan["Physiology"].num_samples


def test_uniform_planner_sums_to_budget():
    plan = build_data_plan(
        strategy="uniform",
        total_budget=27,
        min_samples_per_domain=0,
        target_domains=DOMAINS,
        evaluation=make_evaluation(),
        diagnosis=make_diagnosis(),
    )
    assert plan.allocation_sum() == 27


def test_planner_rejects_invalid_domain():
    with pytest.raises(ValueError):
        build_data_plan(
            strategy="weakness_based",
            total_budget=10,
            min_samples_per_domain=1,
            target_domains=DOMAINS + ["Surgery"],
            evaluation=make_evaluation(),
            diagnosis=make_diagnosis(),
        )


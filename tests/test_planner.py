import pytest

from autodata.data.schemas import DiagnosisResult, DomainMetrics, EvaluationResult
from autodata.planning.llm_agent import normalize_agent_plan, repair_allocations
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


def test_llm_agent_strategy_can_use_mock_provider():
    config = {
        "dataset": {"target_domains": DOMAINS},
        "generation": {"total_budget": 100},
        "planning": {"strategy": "llm_agent", "agent_provider": "mock", "min_samples_per_domain": 5},
    }
    from autodata.planning.data_planner import DataPlanner

    plan = DataPlanner(config).create_plan(make_evaluation(), make_diagnosis())

    assert plan.strategy == "llm_agent"
    assert plan.allocation_sum() == 100
    assert plan.plan["Pharmacology"].generation_guidance


def test_normalize_agent_plan_repairs_budget_sum():
    fallback = build_data_plan(
        strategy="weakness_based",
        total_budget=100,
        min_samples_per_domain=5,
        target_domains=DOMAINS,
        evaluation=make_evaluation(),
        diagnosis=make_diagnosis(),
    )
    payload = {
        "rationale": "Focus weak domains.",
        "plan": {
            "Anatomy": {"num_samples": 80, "generation_guidance": "Use nerves and vascular supply."},
            "Pharmacology": {"num_samples": 80, "generation_guidance": "Use drug mechanisms."},
            "Pathology": {"num_samples": 20},
            "Microbiology": {"num_samples": 20},
            "Physiology": {"num_samples": 20},
        },
    }

    plan = normalize_agent_plan(
        payload,
        fallback,
        make_evaluation(),
        make_diagnosis(),
        min_samples_per_domain=5,
    )

    assert plan.allocation_sum() == 100
    assert all(item.num_samples >= 5 for item in plan.plan.values())
    assert plan.plan["Anatomy"].generation_guidance == "Use nerves and vascular supply."


def test_repair_allocations_uses_fallback_when_agent_counts_missing():
    allocations = repair_allocations(
        raw_counts={domain: 0 for domain in DOMAINS},
        fallback_counts={domain: 20 for domain in DOMAINS},
        total_budget=101,
        min_samples_per_domain=5,
        domains=DOMAINS,
    )

    assert sum(allocations.values()) == 101
    assert set(allocations) == set(DOMAINS)

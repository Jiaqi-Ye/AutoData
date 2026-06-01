from autodata.data.schemas import DataPlan, DomainMetrics, DomainPlan, EvaluationResult, SFTSample
from autodata.mixture.mixture_optimizer import build_mixture


DOMAINS = ["Anatomy", "Pharmacology"]


def make_sample(domain: str, index: int) -> SFTSample:
    return SFTSample(
        id=f"{domain}-{index}",
        domain=domain,
        instruction=f"Question {index}",
        response="The correct answer is A. Explanation: test.",
        source="test",
        generation_model="mock",
        round_id="round_1",
    )


def make_plan() -> DataPlan:
    return DataPlan(
        total_budget=6,
        strategy="weakness_based",
        plan={
            "Anatomy": DomainPlan("Anatomy", 2, "MCQ", "test"),
            "Pharmacology": DomainPlan("Pharmacology", 4, "MCQ", "test"),
        },
    )


def make_eval() -> EvaluationResult:
    return EvaluationResult(
        overall_accuracy=0.5,
        per_domain={
            "Anatomy": DomainMetrics.from_counts("Anatomy", 8, 10),
            "Pharmacology": DomainMetrics.from_counts("Pharmacology", 2, 10),
        },
        sample_count=20,
        correct_count=10,
        incorrect_count=10,
        weakest_domain="Pharmacology",
        strongest_domain="Anatomy",
        variance_across_domains=0.09,
        model_name="mock",
    )


def test_agent_guided_mixture_uses_plan_distribution():
    samples = [make_sample("Anatomy", i) for i in range(3)] + [make_sample("Pharmacology", i) for i in range(5)]
    mixture = build_mixture(
        strategy="agent_guided",
        verified_samples=samples,
        data_plan=make_plan(),
        evaluation=make_eval(),
        target_domains=DOMAINS,
    )
    assert mixture.domain_distribution == {"Anatomy": 2, "Pharmacology": 4}
    assert mixture.dropped_samples == 2


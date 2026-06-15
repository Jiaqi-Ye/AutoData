import pytest

from autodata.data.schemas import DomainMetrics, EvaluationResult, SFTSample
from autodata.evaluation.metrics import compare_evaluations, next_round_recommendation


def make_eval(anatomy_correct: int, pharma_correct: int) -> EvaluationResult:
    per_domain = {
        "Anatomy": DomainMetrics.from_counts("Anatomy", anatomy_correct, 10),
        "Pharmacology": DomainMetrics.from_counts("Pharmacology", pharma_correct, 10),
    }
    total_correct = anatomy_correct + pharma_correct
    return EvaluationResult(
        overall_accuracy=total_correct / 20,
        per_domain=per_domain,
        sample_count=20,
        correct_count=total_correct,
        incorrect_count=20 - total_correct,
        weakest_domain="Pharmacology",
        strongest_domain="Anatomy",
        variance_across_domains=0.01,
        model_name="mock",
    )


def make_sample(domain: str, index: int) -> SFTSample:
    return SFTSample(
        id=f"{domain}-{index}",
        domain=domain,
        instruction="Question",
        response="The correct answer is A. Explanation: test.",
        source="test",
        generation_model="mock",
        round_id="round_1",
    )


def test_compare_evaluations_includes_domain_gains():
    before = make_eval(anatomy_correct=8, pharma_correct=2)
    after = make_eval(anatomy_correct=7, pharma_correct=5)
    metrics = compare_evaluations(before, after)
    assert metrics["overall_gain"] == pytest.approx(0.1)
    assert metrics["per_domain_gain"]["Anatomy"] == pytest.approx(-0.1)
    assert metrics["per_domain_gain"]["Pharmacology"] == pytest.approx(0.3)
    assert metrics["strong_domain_drop"] == pytest.approx(-0.1)


def test_next_round_recommendation_marks_regression_and_weak_domain():
    before = make_eval(anatomy_correct=8, pharma_correct=2)
    after = make_eval(anatomy_correct=7, pharma_correct=5)
    samples = [make_sample("Anatomy", i) for i in range(2)] + [make_sample("Pharmacology", i) for i in range(6)]
    recommendation = next_round_recommendation(before, after, samples)
    assert recommendation["per_domain_gain"]["Pharmacology"] == pytest.approx(0.3)
    assert "Pharmacology" in recommendation["recommended_focus_domains"]
    assert "possible forgetting" in recommendation["rationale_by_domain"]["Anatomy"]

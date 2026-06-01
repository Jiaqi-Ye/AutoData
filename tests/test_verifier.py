from autodata.data.schemas import MedMCQAExample, SFTSample
from autodata.verification.verifier import DataVerifier


def make_config():
    return {
        "dataset": {"target_domains": ["Anatomy", "Pharmacology"]},
        "verification": {"near_duplicate_threshold": 0.99, "leakage_threshold": 0.85},
    }


def make_eval_question():
    return MedMCQAExample(
        id="eval-1",
        domain="Anatomy",
        question="Which nerve innervates the diaphragm?",
        options={"A": "Phrenic", "B": "Vagus", "C": "Median", "D": "Ulnar"},
        correct_answer="A",
    )


def sample(instruction: str, domain: str = "Anatomy") -> SFTSample:
    return SFTSample(
        domain=domain,
        instruction=instruction,
        response="The correct answer is A. Explanation: concise rationale.",
        source="test",
        generation_model="mock",
        round_id="round_1",
    )


def test_verifier_filters_duplicates():
    verifier = DataVerifier(make_config())
    first = sample("Question: Unique anatomy item A/B/C/D?")
    duplicate = sample("Question: Unique anatomy item A/B/C/D?")
    result = verifier.verify([first, duplicate], [make_eval_question()])
    assert len(result.accepted) == 1
    assert result.rejected[0]["reason"] == "duplicate"


def test_verifier_filters_heldout_leakage():
    verifier = DataVerifier(make_config())
    leaked = sample("Which nerve innervates the diaphragm?")
    result = verifier.verify([leaked], [make_eval_question()])
    assert len(result.accepted) == 0
    assert result.rejected[0]["reason"] == "heldout_leakage"


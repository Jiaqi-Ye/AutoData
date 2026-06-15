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


def sample(
    instruction: str,
    domain: str = "Anatomy",
    source: str = "test",
    metadata: dict | None = None,
) -> SFTSample:
    return SFTSample(
        domain=domain,
        instruction=instruction,
        response="The correct answer is A. Explanation: concise rationale.",
        source=source,
        generation_model="mock",
        round_id="round_1",
        metadata=metadata or {},
    )


def mcq_instruction(question: str = "Which option is best?") -> str:
    return f"Question: {question}\nA. First option\nB. Second option\nC. Third option\nD. Fourth option"


def test_verifier_filters_duplicates():
    verifier = DataVerifier(make_config())
    first = sample(mcq_instruction("Unique anatomy item?"))
    duplicate = sample(mcq_instruction("Unique anatomy item?"))
    result = verifier.verify([first, duplicate], [make_eval_question()])
    assert len(result.accepted) == 1
    assert result.rejected[0]["reason"] == "duplicate"


def test_verifier_filters_heldout_leakage():
    verifier = DataVerifier(make_config())
    leaked = sample(mcq_instruction("Which nerve innervates the diaphragm?"))
    result = verifier.verify([leaked], [make_eval_question()])
    assert len(result.accepted) == 0
    assert result.rejected[0]["reason"] == "heldout_leakage"


def test_verifier_requires_mcq_options():
    verifier = DataVerifier(make_config())
    bad = sample("Question: Which nerve innervates the deltoid muscle?")
    result = verifier.verify([bad], [make_eval_question()])
    assert len(result.accepted) == 0
    assert result.rejected[0]["reason"] == "missing_mcq_options"


def test_verifier_rejects_non_pure_local_hf_json():
    verifier = DataVerifier(make_config())
    bad = sample(
        mcq_instruction("Which option is correct?"),
        source="local_hf",
        metadata={"parse_error": "invalid_or_non_pure_json"},
    )
    result = verifier.verify([bad], [make_eval_question()])
    assert len(result.accepted) == 0
    assert result.rejected[0]["reason"] == "invalid_generation_json"


def test_verifier_accepts_strict_local_hf_sample():
    verifier = DataVerifier(make_config())
    good = sample(
        mcq_instruction("Which option is correct?"),
        source="local_hf",
        metadata={"parse_error": "none"},
    )
    result = verifier.verify([good], [make_eval_question()])
    assert len(result.accepted) == 1
    assert len(result.rejected) == 0

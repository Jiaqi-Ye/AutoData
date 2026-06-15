from autodata.data.schemas import GenerationRequest
from autodata.generation.providers import (
    OpenAIGenerationProvider,
    _parse_generated_json,
    _parse_generated_json_samples_with_status,
)


def test_parse_generated_json_accepts_pure_json():
    parsed = _parse_generated_json(
        '{"domain":"Anatomy","instruction":"Question: X\\nA. A\\nB. B\\nC. C\\nD. D","response":"The correct answer is A. Explanation: ok."}'
    )
    assert parsed["domain"] == "Anatomy"
    assert "instruction" in parsed
    assert "response" in parsed


def test_parse_generated_json_extracts_markdown_wrapped_json():
    parsed = _parse_generated_json(
        'Here is an example:\\n```json\\n{"domain":"Anatomy","instruction":"Question: X\\nA. A\\nB. B\\nC. C\\nD. D","response":"The correct answer is A."}\\n```'
    )
    assert parsed["domain"] == "Anatomy"
    assert parsed["response"] == "The correct answer is A."


def test_parse_generated_json_drops_extra_fields():
    parsed = _parse_generated_json(
        '{"domain":"Anatomy","instruction":"Question: X","response":"The correct answer is A.","options":["A"]}'
    )
    assert set(parsed) == {"domain", "instruction", "response"}


def test_parse_generated_json_normalizes_options_array_into_instruction():
    parsed = _parse_generated_json(
        """
        ```json
        {
          "domain": "Anatomy",
          "instruction": "Which nerve innervates the deltoid muscle?",
          "response": "The correct answer is C.",
          "options": [
            {"text": "A. Radial nerve"},
            {"text": "B. Ulnar nerve"},
            {"text": "C. Axillary nerve"},
            {"text": "D. Brachial plexus"}
          ]
        }
        ```
        """
    )
    assert parsed["instruction"].startswith("Question: Which nerve innervates")
    assert "A. Radial nerve" in parsed["instruction"]
    assert "D. Brachial plexus" in parsed["instruction"]


def test_parse_generated_json_builds_response_from_answer_and_explanation():
    parsed = _parse_generated_json(
        '{"domain":"Microbiology","question":"Which drug treats HSV?","options":{"A":"Acyclovir","B":"Isoniazid","C":"Fluconazole","D":"Oseltamivir"},"answer":"A","explanation":"Acyclovir inhibits viral DNA polymerase."}'
    )
    assert "A. Acyclovir" in parsed["instruction"]
    assert parsed["response"].startswith("The correct answer is A.")


def test_parse_generated_json_samples_accepts_samples_wrapper():
    parsed, status = _parse_generated_json_samples_with_status(
        """
        {
          "samples": [
            {
              "domain": "Anatomy",
              "question": "Which nerve innervates the masseter?",
              "options": {"A": "Trigeminal nerve", "B": "Facial nerve", "C": "Vagus nerve", "D": "Hypoglossal nerve"},
              "answer": "A",
              "explanation": "The mandibular division of CN V supplies muscles of mastication."
            },
            {
              "domain": "Microbiology",
              "instruction": "Question: Which organism causes botulism?\\nA. C. difficile\\nB. S. aureus\\nC. E. coli\\nD. C. botulinum",
              "response": "The correct answer is D. Explanation: C. botulinum produces botulinum toxin."
            }
          ]
        }
        """
    )
    assert status == "pure_json"
    assert len(parsed) == 2
    assert "A. Trigeminal nerve" in parsed[0]["instruction"]
    assert parsed[1]["response"].startswith("The correct answer is D.")


def test_openai_generation_provider_batches_five_per_call():
    provider = OpenAIGenerationProvider()
    provider._client = FakeOpenAIClient()
    request = GenerationRequest(
        domain="Anatomy",
        num_samples=7,
        data_type="MCQ",
        reason="test",
        round_id="round_1",
    )
    samples = provider.generate(
        request,
        {
            "models": {"generation_model": "unused"},
            "generation": {"api_model": "gpt-test", "api_batch_size": 5},
        },
    )

    assert len(samples) == 7
    assert len(provider._client.chat.completions.calls) == 2
    assert samples[0].source == "openai"
    assert samples[0].generation_model == "gpt-test"
    assert samples[5].metadata["api_batch_size"] == 2


class FakeOpenAIClient:
    def __init__(self):
        self.chat = FakeChat()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        requested = 5 if len(self.calls) == 1 else 2
        rows = []
        for index in range(requested):
            rows.append(
                {
                    "domain": "Anatomy",
                    "instruction": (
                        f"Question: Which structure is example {index}?\\n"
                        "A. Correct structure\\nB. Distractor one\\nC. Distractor two\\nD. Distractor three"
                    ),
                    "response": "The correct answer is A. Explanation: Correct structure is correct.",
                }
            )
        return FakeResponse({"samples": rows})


class FakeResponse:
    def __init__(self, payload):
        import json

        self.choices = [FakeChoice(json.dumps(payload))]


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeMessage:
    def __init__(self, content):
        self.content = content

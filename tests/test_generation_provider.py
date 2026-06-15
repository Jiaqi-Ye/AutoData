from autodata.generation.providers import _parse_generated_json


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

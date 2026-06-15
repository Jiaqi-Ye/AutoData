from autodata.generation.providers import _parse_generated_json


def test_parse_generated_json_accepts_pure_json_only():
    parsed = _parse_generated_json(
        '{"domain":"Anatomy","instruction":"Question: X\\nA. A\\nB. B\\nC. C\\nD. D","response":"The correct answer is A. Explanation: ok."}'
    )
    assert parsed["domain"] == "Anatomy"
    assert "instruction" in parsed
    assert "response" in parsed


def test_parse_generated_json_rejects_markdown_wrapped_json():
    parsed = _parse_generated_json(
        'Here is an example:\\n```json\\n{"domain":"Anatomy","instruction":"Question: X","response":"The correct answer is A."}\\n```'
    )
    assert parsed == {}


def test_parse_generated_json_drops_extra_fields():
    parsed = _parse_generated_json(
        '{"domain":"Anatomy","instruction":"Question: X","response":"The correct answer is A.","options":["A"]}'
    )
    assert set(parsed) == {"domain", "instruction", "response"}

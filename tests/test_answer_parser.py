from autodata.evaluation.answer_parser import parse_answer


def test_parse_answer_common_formats():
    assert parse_answer("The correct answer is B.") == "B"
    assert parse_answer("Answer: C") == "C"
    assert parse_answer("Option D") == "D"
    assert parse_answer("A") == "A"


def test_parse_answer_empty_text():
    assert parse_answer("") is None
    assert parse_answer(None) is None


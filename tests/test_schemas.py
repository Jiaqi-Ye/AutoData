from pathlib import Path

import pytest

from autodata.config import load_config
from autodata.data.schemas import MedMCQAExample


def test_medmcqa_schema_validates_answer_and_options():
    example = MedMCQAExample(
        id="x",
        domain="Anatomy",
        question="What is tested?",
        options={"A": "one", "B": "two", "C": "three", "D": "four"},
        correct_answer="b",
    )
    assert example.correct_answer == "B"
    assert set(example.options) == {"A", "B", "C", "D"}


def test_medmcqa_schema_rejects_invalid_answer():
    with pytest.raises(ValueError):
        MedMCQAExample(
            id="x",
            domain="Anatomy",
            question="What is tested?",
            options={"A": "one", "B": "two", "C": "three", "D": "four"},
            correct_answer="E",
        )


def test_smoke_config_loading():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "smoke_colab.yaml"
    config = load_config(config_path)
    assert config["project"]["run_mode"] == "smoke"
    assert config["models"]["use_real_model"] is False
    assert config["generation"]["use_mock_generation"] is True


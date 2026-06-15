import yaml

from autodata.data.medmcqa_loader import _mock_question
from autodata.data.schemas import DataPlan, DomainPlan
from autodata.experiments.scaling import (
    medmcqa_example_to_sft_sample,
    run_real_vs_synthetic_scaling,
    select_real_samples,
)


def test_medmcqa_example_to_sft_sample_formats_answer():
    example = _mock_question("Anatomy", 0, "train")

    sample = medmcqa_example_to_sft_sample(example)

    assert sample.source == "real_medmcqa"
    assert sample.instruction.startswith("Question:")
    assert "A." in sample.instruction
    assert sample.response.startswith("The correct answer is A.")


def test_select_real_samples_follows_data_plan():
    train_pool = [_mock_question("Anatomy", index, "train") for index in range(3)]
    plan = DataPlan(
        total_budget=2,
        strategy="test",
        plan={
            "Anatomy": DomainPlan(
                domain="Anatomy",
                num_samples=2,
                data_type="MCQ",
                reason="test",
            )
        },
    )

    selected = select_real_samples(train_pool, plan)

    assert len(selected) == 2
    assert all(sample.source == "real_medmcqa" for sample in selected)


def test_real_vs_synthetic_scaling_smoke(tmp_path):
    config = yaml.safe_load(open("configs/smoke_colab.yaml", encoding="utf-8"))
    config["project"]["output_dir"] = str(tmp_path)
    config["planning"]["strategy"] = "llm_agent"
    config["planning"]["agent_provider"] = "mock"
    config["generation"]["provider"] = "mock"
    config["generation"]["use_mock_generation"] = True
    config["generation"]["total_budget"] = 10
    config["training"]["enabled"] = False
    config["training"]["dry_run"] = True
    config["medical_critic"] = {"enabled": False}

    output = run_real_vs_synthetic_scaling(config, budgets=[10])

    assert output["budgets"] == [10]
    assert len(output["rows"]) == 2
    assert {row["branch"] for row in output["rows"]} == {"real_medmcqa", "synthetic_agent"}

"""Real-vs-synthetic scaling experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

from autodata.config import save_config
from autodata.data.medmcqa_loader import load_medmcqa_data
from autodata.data.schemas import DataPlan, MedMCQAExample, SFTSample
from autodata.diagnosis.weakness_diagnoser import WeaknessDiagnoser
from autodata.evaluation.evaluator import Evaluator
from autodata.evaluation.metrics import compare_evaluations
from autodata.generation.generator import DataGenerator
from autodata.mixture.mixture_optimizer import MixtureOptimizer
from autodata.planning.data_planner import DataPlanner
from autodata.training.trainer import Trainer
from autodata.utils.gpu import clear_gpu_memory
from autodata.utils.io import create_timestamped_run_dir, ensure_dir, write_json, write_jsonl
from autodata.utils.seed import set_seed
from autodata.verification.medical_critic import apply_medical_critic
from autodata.verification.verifier import DataVerifier


def run_real_vs_synthetic_scaling(config: Dict[str, Any], budgets: Iterable[int]) -> Dict[str, Any]:
    """Run baseline, real-data QLoRA, and synthetic-data QLoRA at each budget."""
    scale_budgets = [int(budget) for budget in budgets]
    if not scale_budgets:
        raise ValueError("budgets must contain at least one scale point")

    root_config = deepcopy(config)
    set_seed(int(root_config.get("project", {}).get("seed", 42)))
    output_root = ensure_dir(root_config.get("project", {}).get("output_dir", "outputs"))
    experiment_dir = create_timestamped_run_dir(output_root / "real_vs_synthetic_scaling")
    save_config(root_config, experiment_dir / "config.yaml")

    eval_examples, train_pool = load_medmcqa_data(root_config, run_dir=experiment_dir)
    evaluator = Evaluator(root_config)
    evaluation_base = evaluator.evaluate(eval_examples, phase="base")
    write_json(experiment_dir / "evaluation_base.json", evaluation_base)
    clear_gpu_memory()

    diagnosis = WeaknessDiagnoser(root_config).diagnose(evaluation_base)
    write_json(experiment_dir / "diagnosis.json", diagnosis)

    rows: List[Dict[str, Any]] = []
    for scale_index, budget in enumerate(scale_budgets):
        scale_config = deepcopy(root_config)
        scale_config["generation"]["total_budget"] = budget
        scale_config["project"]["seed"] = int(root_config.get("project", {}).get("seed", 42)) + scale_index

        data_plan = DataPlanner(scale_config).create_plan(evaluation_base, diagnosis)
        if data_plan.allocation_sum() != budget:
            raise AssertionError("data plan allocation must equal the requested budget")

        budget_dir = experiment_dir / f"budget_{budget}"
        budget_dir.mkdir(parents=True, exist_ok=True)
        write_json(budget_dir / "data_plan.json", data_plan)

        real_row = _run_real_branch(
            config=scale_config,
            budget=budget,
            budget_dir=budget_dir,
            train_pool=train_pool,
            eval_examples=eval_examples,
            evaluator=evaluator,
            evaluation_base=evaluation_base,
            data_plan=data_plan,
        )
        rows.append(real_row)

        synthetic_row = _run_synthetic_branch(
            config=scale_config,
            budget=budget,
            budget_dir=budget_dir,
            eval_examples=eval_examples,
            evaluator=evaluator,
            evaluation_base=evaluation_base,
            data_plan=data_plan,
        )
        rows.append(synthetic_row)

    output = {
        "experiment_dir": str(experiment_dir.resolve()),
        "budgets": scale_budgets,
        "base_overall_accuracy": evaluation_base.overall_accuracy,
        "rows": rows,
    }
    write_json(experiment_dir / "real_vs_synthetic_scaling_results.json", output)
    return output


def medmcqa_example_to_sft_sample(example: MedMCQAExample, round_id: str = "real_medmcqa") -> SFTSample:
    selected_option = example.options[example.correct_answer]
    explanation = example.explanation.strip()
    if explanation:
        response = f"The correct answer is {example.correct_answer}. {selected_option}. Explanation: {explanation}"
    else:
        response = (
            f"The correct answer is {example.correct_answer}. {selected_option}. "
            "Explanation: This is the labeled correct option in the MedMCQA training split."
        )
    instruction = (
        f"Question: {example.question}\n"
        f"A. {example.options['A']}\n"
        f"B. {example.options['B']}\n"
        f"C. {example.options['C']}\n"
        f"D. {example.options['D']}"
    )
    return SFTSample(
        id=f"real-{example.id}",
        domain=example.domain,
        instruction=instruction,
        response=response,
        source="real_medmcqa",
        generation_model="real_medmcqa",
        round_id=round_id,
        metadata={"medmcqa_id": example.id, "subject": example.subject, "split": example.split},
    )


def select_real_samples(train_pool: List[MedMCQAExample], data_plan: DataPlan) -> List[SFTSample]:
    selected: List[SFTSample] = []
    by_domain: Dict[str, List[MedMCQAExample]] = {}
    for example in train_pool:
        by_domain.setdefault(example.domain, []).append(example)

    for domain, domain_plan in data_plan.plan.items():
        examples = by_domain.get(domain, [])
        if len(examples) < domain_plan.num_samples:
            raise RuntimeError(
                f"Not enough real MedMCQA training examples for {domain}: "
                f"needed {domain_plan.num_samples}, found {len(examples)}"
            )
        for example in examples[: domain_plan.num_samples]:
            selected.append(medmcqa_example_to_sft_sample(example))
    return selected


def _run_real_branch(
    config: Dict[str, Any],
    budget: int,
    budget_dir: Path,
    train_pool: List[MedMCQAExample],
    eval_examples: List[MedMCQAExample],
    evaluator: Evaluator,
    evaluation_base,
    data_plan: DataPlan,
) -> Dict[str, Any]:
    branch_dir = budget_dir / "real_medmcqa"
    branch_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, branch_dir / "config.yaml")
    write_json(branch_dir / "data_plan.json", data_plan)

    real_samples = select_real_samples(train_pool, data_plan)
    write_jsonl(branch_dir / "real_train_samples.jsonl", real_samples)
    write_jsonl(branch_dir / "verified_samples.jsonl", real_samples)
    verification_report = {
        "accepted_count": len(real_samples),
        "rejected_count": 0,
        "accepted_by_domain": _count_by_domain(real_samples),
        "rejected_by_domain": {},
        "rejection_reasons": {},
        "source": "real_medmcqa",
    }
    write_json(branch_dir / "verification_report.json", verification_report)

    mixture = MixtureOptimizer(config).optimize(real_samples, data_plan, evaluation_base)
    write_jsonl(branch_dir / "mixture_train.jsonl", mixture.samples)
    mixture_report = _mixture_report(mixture)
    write_json(branch_dir / "mixture_report.json", mixture_report)

    training_report = Trainer(config).train(mixture.samples, run_dir=branch_dir)
    write_json(branch_dir / "training_report.json", training_report)
    clear_gpu_memory()

    model_path = training_report.output_dir if training_report.status == "completed" else None
    evaluation_after = evaluator.evaluate(eval_examples, model_path=model_path, phase="real_after")
    write_json(branch_dir / "evaluation_after.json", evaluation_after)
    clear_gpu_memory()

    metrics = compare_evaluations(evaluation_base, evaluation_after)
    row = _result_row(
        branch="real_medmcqa",
        budget=budget,
        run_dir=branch_dir,
        evaluation_base=evaluation_base,
        evaluation_after=evaluation_after,
        metrics=metrics,
        generated_count=len(real_samples),
        accepted_count=len(real_samples),
        mixture_sample_count=len(mixture.samples),
        training_status=training_report.status,
    )
    write_json(branch_dir / "round_summary.json", row)
    return row


def _run_synthetic_branch(
    config: Dict[str, Any],
    budget: int,
    budget_dir: Path,
    eval_examples: List[MedMCQAExample],
    evaluator: Evaluator,
    evaluation_base,
    data_plan: DataPlan,
) -> Dict[str, Any]:
    branch_dir = budget_dir / "synthetic_agent"
    branch_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, branch_dir / "config.yaml")
    write_json(branch_dir / "data_plan.json", data_plan)

    generated_samples = DataGenerator(config).generate(data_plan, round_id=f"synthetic_{budget}")
    write_jsonl(branch_dir / "generated_samples.jsonl", generated_samples)
    clear_gpu_memory()

    rule_verification = DataVerifier(config).verify(generated_samples, heldout_eval_examples=eval_examples)
    verification, medical_critic_result = apply_medical_critic(config, rule_verification)
    if medical_critic_result is not None:
        write_jsonl(branch_dir / "rule_verified_samples.jsonl", rule_verification.accepted)
        write_jsonl(branch_dir / "rule_rejected_samples.jsonl", rule_verification.rejected)
        write_json(branch_dir / "rule_verification_report.json", rule_verification.report)
        write_jsonl(branch_dir / "medical_critic_rejected_samples.jsonl", medical_critic_result.rejected)
        write_json(branch_dir / "medical_critic_report.json", medical_critic_result.report)
    write_jsonl(branch_dir / "verified_samples.jsonl", verification.accepted)
    write_jsonl(branch_dir / "rejected_samples.jsonl", verification.rejected)
    write_json(branch_dir / "verification_report.json", verification.report)

    mixture = MixtureOptimizer(config).optimize(verification.accepted, data_plan, evaluation_base)
    write_jsonl(branch_dir / "mixture_train.jsonl", mixture.samples)
    mixture_report = _mixture_report(mixture)
    write_json(branch_dir / "mixture_report.json", mixture_report)

    training_report = Trainer(config).train(mixture.samples, run_dir=branch_dir)
    write_json(branch_dir / "training_report.json", training_report)
    clear_gpu_memory()

    model_path = training_report.output_dir if training_report.status == "completed" else None
    evaluation_after = evaluator.evaluate(eval_examples, model_path=model_path, phase="synthetic_after")
    write_json(branch_dir / "evaluation_after.json", evaluation_after)
    clear_gpu_memory()

    metrics = compare_evaluations(evaluation_base, evaluation_after)
    row = _result_row(
        branch="synthetic_agent",
        budget=budget,
        run_dir=branch_dir,
        evaluation_base=evaluation_base,
        evaluation_after=evaluation_after,
        metrics=metrics,
        generated_count=len(generated_samples),
        accepted_count=len(verification.accepted),
        mixture_sample_count=len(mixture.samples),
        training_status=training_report.status,
    )
    write_json(branch_dir / "round_summary.json", row)
    return row


def _result_row(
    branch: str,
    budget: int,
    run_dir: Path,
    evaluation_base,
    evaluation_after,
    metrics: Dict[str, Any],
    generated_count: int,
    accepted_count: int,
    mixture_sample_count: int,
    training_status: str,
) -> Dict[str, Any]:
    return {
        "branch": branch,
        "budget": budget,
        "run_dir": str(run_dir.resolve()),
        "base_overall_accuracy": evaluation_base.overall_accuracy,
        "after_overall_accuracy": evaluation_after.overall_accuracy,
        "overall_gain": metrics["overall_gain"],
        "generated_count": generated_count,
        "accepted_count": accepted_count,
        "mixture_sample_count": mixture_sample_count,
        "training_status": training_status,
        "per_domain_gain": metrics["per_domain_gain"],
    }


def _mixture_report(mixture) -> Dict[str, Any]:
    return {
        "final_sample_count": len(mixture.samples),
        "domain_distribution": mixture.domain_distribution,
        "strategy": mixture.strategy,
        "dropped_samples": mixture.dropped_samples,
        "reason_for_mixture_decisions": mixture.reasons,
    }


def _count_by_domain(samples: List[SFTSample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.domain] = counts.get(sample.domain, 0) + 1
    return counts

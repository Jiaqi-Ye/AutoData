"""End-to-end AutoData loop orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from autodata.config import save_config
from autodata.data.medmcqa_loader import load_medmcqa_data
from autodata.data.schemas import LoopRoundResult, to_jsonable
from autodata.diagnosis.weakness_diagnoser import WeaknessDiagnoser
from autodata.evaluation.evaluator import Evaluator
from autodata.evaluation.metrics import compare_evaluations, next_round_recommendation, system_metrics
from autodata.generation.generator import DataGenerator
from autodata.mixture.mixture_optimizer import MixtureOptimizer
from autodata.planning.data_planner import DataPlanner
from autodata.training.trainer import Trainer
from autodata.utils.gpu import clear_gpu_memory
from autodata.utils.io import create_timestamped_run_dir, write_json, write_jsonl
from autodata.utils.seed import set_seed
from autodata.verification.medical_critic import apply_medical_critic, preflight_medical_critic
from autodata.verification.verifier import DataVerifier


def run_autodata_loop(config: Dict[str, Any]) -> LoopRoundResult:
    """Run one AutoData round and persist all stage artifacts."""
    set_seed(int(config.get("project", {}).get("seed", 42)))
    run_dir = create_timestamped_run_dir(config.get("project", {}).get("output_dir", "outputs"))
    save_config(config, run_dir / "config.yaml")
    preflight_medical_critic(config)

    eval_examples, train_pool = load_medmcqa_data(config, run_dir=run_dir)

    evaluator = Evaluator(config)
    evaluation_base = evaluator.evaluate(eval_examples, phase="base")
    write_json(run_dir / "evaluation_base.json", evaluation_base)
    clear_gpu_memory()

    diagnosis = WeaknessDiagnoser(config).diagnose(evaluation_base)
    write_json(run_dir / "diagnosis.json", diagnosis)

    data_plan = DataPlanner(config).create_plan(evaluation_base, diagnosis)
    write_json(run_dir / "data_plan.json", data_plan)

    generated_samples = DataGenerator(config).generate(data_plan, round_id="round_1")
    write_jsonl(run_dir / "generated_samples.jsonl", generated_samples)
    clear_gpu_memory()

    rule_verification = DataVerifier(config).verify(generated_samples, heldout_eval_examples=eval_examples)
    verification, medical_critic_result = apply_medical_critic(config, rule_verification)
    if medical_critic_result is not None:
        write_jsonl(run_dir / "rule_verified_samples.jsonl", rule_verification.accepted)
        write_jsonl(run_dir / "rule_rejected_samples.jsonl", rule_verification.rejected)
        write_json(run_dir / "rule_verification_report.json", rule_verification.report)
        write_jsonl(run_dir / "medical_critic_rejected_samples.jsonl", medical_critic_result.rejected)
        write_json(run_dir / "medical_critic_report.json", medical_critic_result.report)
    write_jsonl(run_dir / "verified_samples.jsonl", verification.accepted)
    write_jsonl(run_dir / "rejected_samples.jsonl", verification.rejected)
    write_json(run_dir / "verification_report.json", verification.report)

    mixture = MixtureOptimizer(config).optimize(verification.accepted, data_plan, evaluation_base)
    write_jsonl(run_dir / "mixture_train.jsonl", mixture.samples)
    mixture_report = {
        "final_sample_count": len(mixture.samples),
        "domain_distribution": mixture.domain_distribution,
        "strategy": mixture.strategy,
        "dropped_samples": mixture.dropped_samples,
        "reason_for_mixture_decisions": mixture.reasons,
    }
    write_json(run_dir / "mixture_report.json", mixture_report)

    training_report = Trainer(config).train(mixture.samples, run_dir=run_dir)
    write_json(run_dir / "training_report.json", training_report)
    clear_gpu_memory()

    model_path = training_report.output_dir if training_report.status == "completed" else None
    evaluation_after = evaluator.evaluate(eval_examples, model_path=model_path, phase="after")
    write_json(run_dir / "evaluation_after.json", evaluation_after)
    clear_gpu_memory()

    model_metrics = compare_evaluations(evaluation_base, evaluation_after)
    next_round = next_round_recommendation(evaluation_base, evaluation_after, mixture.samples)
    write_json(run_dir / "next_round_recommendation.json", next_round)
    auto_metrics = system_metrics(
        generated_samples=generated_samples,
        accepted_samples=verification.accepted,
        rejected_samples=verification.rejected,
        mixture_samples=mixture.samples,
        evaluation_gain=float(model_metrics["overall_gain"]),
    )
    metrics = {
        "model_level": model_metrics,
        "system_level": auto_metrics,
        "next_round": next_round,
        "train_pool_size": len(train_pool),
    }

    round_summary = {
        "run_dir": str(Path(run_dir).resolve()),
        "run_mode": config.get("project", {}).get("run_mode", "smoke"),
        "base_overall_accuracy": evaluation_base.overall_accuracy,
        "after_overall_accuracy": evaluation_after.overall_accuracy,
        "overall_gain": model_metrics["overall_gain"],
        "weakest_domain": evaluation_base.weakest_domain,
        "generated_count": len(generated_samples),
        "accepted_count": len(verification.accepted),
        "mixture_sample_count": len(mixture.samples),
        "training_status": training_report.status,
        "next_round_focus_domains": next_round["recommended_focus_domains"],
        "metrics": to_jsonable(metrics),
    }
    write_json(run_dir / "round_summary.json", round_summary)

    return LoopRoundResult(
        run_dir=str(Path(run_dir).resolve()),
        config=config,
        evaluation_base=evaluation_base,
        diagnosis=diagnosis,
        data_plan=data_plan,
        verification_report=verification.report,
        mixture_report=mixture_report,
        training_report=training_report,
        evaluation_after=evaluation_after,
        metrics=metrics,
    )

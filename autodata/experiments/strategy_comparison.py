"""Run comparable AutoData loops across mixture/planning strategies."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from autodata.loop.run_loop import run_autodata_loop
from autodata.utils.io import ensure_dir, write_json


DEFAULT_STRATEGIES = ["uniform", "weakness_based", "agent_guided"]


def create_comparison_dir(output_dir: str | Path) -> Path:
    base = ensure_dir(Path(output_dir) / "comparisons")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = base / timestamp
    suffix = 1
    while comparison_dir.exists():
        comparison_dir = base / f"{timestamp}_{suffix:02d}"
        suffix += 1
    comparison_dir.mkdir(parents=True)
    return comparison_dir


def run_strategy_comparison(
    config: Dict[str, Any],
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
) -> Dict[str, Any]:
    """Run one full AutoData loop per strategy and save a comparison table."""
    comparison_dir = create_comparison_dir(config.get("project", {}).get("output_dir", "outputs"))
    rows: List[Dict[str, Any]] = []

    for strategy in strategies:
        strategy_config = deepcopy(config)
        strategy_config.setdefault("planning", {})["strategy"] = strategy
        strategy_config.setdefault("mixture", {})["strategy"] = strategy
        strategy_config.setdefault("project", {})["run_mode"] = (
            f"{config.get('project', {}).get('run_mode', 'experiment')}_{strategy}"
        )

        result = run_autodata_loop(strategy_config)
        model_metrics = result.metrics.get("model_level", {})
        system_metrics = result.metrics.get("system_level", {})
        rows.append(
            {
                "strategy": strategy,
                "run_dir": result.run_dir,
                "training_status": result.training_report.status,
                "base_overall_accuracy": result.evaluation_base.overall_accuracy,
                "after_overall_accuracy": result.evaluation_after.overall_accuracy,
                "overall_gain": model_metrics.get("overall_gain", 0.0),
                "average_domain_gain": model_metrics.get("average_domain_gain", 0.0),
                "weakest_domain_improvement": model_metrics.get("weakest_domain_improvement", 0.0),
                "strong_domain_drop": model_metrics.get("strong_domain_drop", 0.0),
                "accepted_ratio": system_metrics.get("accepted_ratio", 0.0),
                "mixture_distribution": result.mixture_report.get("domain_distribution", {}),
                "next_round_focus_domains": result.metrics.get("next_round", {}).get(
                    "recommended_focus_domains", []
                ),
            }
        )

    report = {
        "comparison_dir": str(comparison_dir.resolve()),
        "strategies": rows,
    }
    write_json(comparison_dir / "strategy_comparison.json", report)
    return report


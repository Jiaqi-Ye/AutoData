from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autodata.config import load_config
from autodata.data.schemas import data_plan_from_dict, evaluation_result_from_dict, sft_sample_from_dict
from autodata.mixture.mixture_optimizer import MixtureOptimizer
from autodata.utils.io import create_timestamped_run_dir, read_json, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a training mixture from verified samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verified", required=True)
    parser.add_argument("--data-plan", required=True)
    parser.add_argument("--evaluation", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = create_timestamped_run_dir(config.get("project", {}).get("output_dir", "outputs"))
    samples = [sft_sample_from_dict(row) for row in read_jsonl(args.verified)]
    data_plan = data_plan_from_dict(read_json(args.data_plan))
    evaluation = evaluation_result_from_dict(read_json(args.evaluation))
    mixture = MixtureOptimizer(config).optimize(samples, data_plan, evaluation)
    report = {
        "final_sample_count": len(mixture.samples),
        "domain_distribution": mixture.domain_distribution,
        "strategy": mixture.strategy,
        "dropped_samples": mixture.dropped_samples,
        "reason_for_mixture_decisions": mixture.reasons,
    }
    write_jsonl(run_dir / "mixture_train.jsonl", mixture.samples)
    write_json(run_dir / "mixture_report.json", report)
    print(f"Mixture contains {len(mixture.samples)} samples.")
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()


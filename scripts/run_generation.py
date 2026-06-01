from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autodata.config import load_config
from autodata.data.medmcqa_loader import load_medmcqa_data
from autodata.diagnosis.weakness_diagnoser import WeaknessDiagnoser
from autodata.evaluation.evaluator import Evaluator
from autodata.generation.generator import DataGenerator
from autodata.planning.data_planner import DataPlanner
from autodata.utils.io import create_timestamped_run_dir, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and generate synthetic SFT samples.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = create_timestamped_run_dir(config.get("project", {}).get("output_dir", "outputs"))
    eval_examples, _ = load_medmcqa_data(config, run_dir=run_dir)
    evaluation = Evaluator(config).evaluate(eval_examples)
    diagnosis = WeaknessDiagnoser(config).diagnose(evaluation)
    data_plan = DataPlanner(config).create_plan(evaluation, diagnosis)
    samples = DataGenerator(config).generate(data_plan)
    write_json(run_dir / "data_plan.json", data_plan)
    write_jsonl(run_dir / "generated_samples.jsonl", samples)
    print(f"Generated {len(samples)} samples.")
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()


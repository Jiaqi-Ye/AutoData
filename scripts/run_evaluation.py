from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autodata.config import load_config
from autodata.data.medmcqa_loader import load_medmcqa_data
from autodata.evaluation.evaluator import Evaluator
from autodata.utils.io import create_timestamped_run_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoData evaluation only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", default="base")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = create_timestamped_run_dir(config.get("project", {}).get("output_dir", "outputs"))
    eval_examples, _ = load_medmcqa_data(config, run_dir=run_dir)
    result = Evaluator(config).evaluate(eval_examples, phase=args.phase)
    write_json(run_dir / f"evaluation_{args.phase}.json", result)
    print(f"Accuracy: {result.overall_accuracy:.3f}")
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()


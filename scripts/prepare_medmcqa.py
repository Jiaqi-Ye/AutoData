from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autodata.config import load_config
from autodata.data.medmcqa_loader import load_medmcqa_data
from autodata.utils.io import create_timestamped_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare target-domain MedMCQA data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = create_timestamped_run_dir(config.get("project", {}).get("output_dir", "outputs"))
    eval_examples, train_pool = load_medmcqa_data(config, run_dir=run_dir)
    print(f"Prepared {len(eval_examples)} eval examples and {len(train_pool)} train-pool examples.")
    print(f"Saved to {run_dir / 'prepared_data'}")


if __name__ == "__main__":
    main()


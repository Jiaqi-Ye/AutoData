from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autodata.config import load_config
from autodata.data.schemas import sft_sample_from_dict
from autodata.training.trainer import Trainer
from autodata.utils.io import create_timestamped_run_dir, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA/QLoRA training from a mixture JSONL file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mixture", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = create_timestamped_run_dir(config.get("project", {}).get("output_dir", "outputs"))
    samples = [sft_sample_from_dict(row) for row in read_jsonl(args.mixture)]
    result = Trainer(config).train(samples, run_dir)
    write_json(run_dir / "training_report.json", result)
    print(f"Training status: {result.status}")
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()


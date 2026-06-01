from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autodata.config import load_config
from autodata.loop.run_loop import run_autodata_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AutoData loop.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_autodata_loop(config)
    print(f"Run directory: {result.run_dir}")
    print(f"Base accuracy: {result.evaluation_base.overall_accuracy:.3f}")
    print(f"After accuracy: {result.evaluation_after.overall_accuracy:.3f}")
    print(f"Training status: {result.training_report.status}")


if __name__ == "__main__":
    main()


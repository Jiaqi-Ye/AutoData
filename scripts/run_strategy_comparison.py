from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autodata.config import load_config
from autodata.experiments.strategy_comparison import DEFAULT_STRATEGIES, run_strategy_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoData strategy comparison.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help="Strategies to compare.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    report = run_strategy_comparison(config, strategies=args.strategies)
    print(f"Comparison directory: {report['comparison_dir']}")
    for row in report["strategies"]:
        print(
            f"{row['strategy']}: gain={row['overall_gain']:.3f}, "
            f"accepted_ratio={row['accepted_ratio']:.3f}, run_dir={row['run_dir']}"
        )


if __name__ == "__main__":
    main()

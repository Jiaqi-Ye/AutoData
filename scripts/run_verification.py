from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autodata.config import load_config
from autodata.data.medmcqa_loader import load_medmcqa_data
from autodata.data.schemas import sft_sample_from_dict
from autodata.utils.io import create_timestamped_run_dir, read_jsonl, write_json, write_jsonl
from autodata.verification.verifier import DataVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify generated SFT samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True, help="Path to generated_samples.jsonl.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = create_timestamped_run_dir(config.get("project", {}).get("output_dir", "outputs"))
    eval_examples, _ = load_medmcqa_data(config, run_dir=run_dir)
    samples = [sft_sample_from_dict(row) for row in read_jsonl(args.samples)]
    result = DataVerifier(config).verify(samples, eval_examples)
    write_jsonl(run_dir / "verified_samples.jsonl", result.accepted)
    write_jsonl(run_dir / "rejected_samples.jsonl", result.rejected)
    write_json(run_dir / "verification_report.json", result.report)
    print(f"Accepted {len(result.accepted)} / {len(samples)} samples.")
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()


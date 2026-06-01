"""Synthetic SFT data generation orchestration."""

from __future__ import annotations

from typing import Any, Dict, List

from autodata.data.schemas import DataPlan, GenerationRequest, SFTSample
from autodata.generation.providers import get_generation_provider


class DataGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = get_generation_provider(config)

    def generate(self, data_plan: DataPlan, round_id: str = "round_1") -> List[SFTSample]:
        samples: List[SFTSample] = []
        for domain, domain_plan in data_plan.plan.items():
            request = GenerationRequest(
                domain=domain,
                num_samples=domain_plan.num_samples,
                data_type=domain_plan.data_type,
                reason=domain_plan.reason,
                round_id=round_id,
            )
            samples.extend(self.provider.generate(request, self.config))
        return samples


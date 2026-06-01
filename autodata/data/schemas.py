"""Structured objects passed between AutoData modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_ANSWERS = {"A", "B", "C", "D"}


def _clean_answer(answer: str) -> str:
    value = str(answer).strip().upper()
    if value not in VALID_ANSWERS:
        raise ValueError(f"correct_answer must be one of A/B/C/D, got {answer!r}")
    return value


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and paths into JSON-friendly containers."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def domain_metrics_from_dict(payload: Dict[str, Any]) -> "DomainMetrics":
    return DomainMetrics(
        domain=str(payload["domain"]),
        accuracy=float(payload["accuracy"]),
        num_samples=int(payload["num_samples"]),
        correct=int(payload["correct"]),
        incorrect=int(payload["incorrect"]),
    )


def evaluation_result_from_dict(payload: Dict[str, Any]) -> "EvaluationResult":
    return EvaluationResult(
        overall_accuracy=float(payload["overall_accuracy"]),
        per_domain={domain: domain_metrics_from_dict(metrics) for domain, metrics in payload["per_domain"].items()},
        sample_count=int(payload["sample_count"]),
        correct_count=int(payload["correct_count"]),
        incorrect_count=int(payload["incorrect_count"]),
        weakest_domain=str(payload["weakest_domain"]),
        strongest_domain=str(payload["strongest_domain"]),
        variance_across_domains=float(payload["variance_across_domains"]),
        model_name=str(payload["model_name"]),
        mode=str(payload.get("mode", "mock")),
        predictions=list(payload.get("predictions", [])),
    )


def data_plan_from_dict(payload: Dict[str, Any]) -> "DataPlan":
    return DataPlan(
        total_budget=int(payload["total_budget"]),
        strategy=str(payload["strategy"]),
        plan={
            domain: DomainPlan(
                domain=str(item["domain"]),
                num_samples=int(item["num_samples"]),
                data_type=str(item["data_type"]),
                reason=str(item["reason"]),
            )
            for domain, item in payload["plan"].items()
        },
    )


def sft_sample_from_dict(payload: Dict[str, Any]) -> "SFTSample":
    return SFTSample(
        domain=str(payload["domain"]),
        instruction=str(payload["instruction"]),
        response=str(payload["response"]),
        source=str(payload.get("source", "unknown")),
        generation_model=str(payload.get("generation_model", "unknown")),
        round_id=str(payload.get("round_id", "round_1")),
        metadata=dict(payload.get("metadata", {})),
        id=payload.get("id"),
    )


@dataclass
class MedMCQAExample:
    id: str
    domain: str
    question: str
    options: Dict[str, str]
    correct_answer: str
    explanation: str = ""
    split: str = "eval"
    subject: Optional[str] = None

    def __post_init__(self) -> None:
        self.domain = str(self.domain).strip()
        self.question = str(self.question).strip()
        self.correct_answer = _clean_answer(self.correct_answer)
        normalized_options = {str(k).upper(): str(v).strip() for k, v in self.options.items()}
        missing = VALID_ANSWERS.difference(normalized_options)
        if missing:
            raise ValueError(f"options missing choices: {sorted(missing)}")
        self.options = {key: normalized_options[key] for key in sorted(VALID_ANSWERS)}
        if not self.question:
            raise ValueError("question must be non-empty")


@dataclass
class DomainMetrics:
    domain: str
    accuracy: float
    num_samples: int
    correct: int
    incorrect: int

    @classmethod
    def from_counts(cls, domain: str, correct: int, total: int) -> "DomainMetrics":
        incorrect = max(total - correct, 0)
        accuracy = correct / total if total else 0.0
        return cls(domain=domain, accuracy=accuracy, num_samples=total, correct=correct, incorrect=incorrect)


@dataclass
class EvaluationResult:
    overall_accuracy: float
    per_domain: Dict[str, DomainMetrics]
    sample_count: int
    correct_count: int
    incorrect_count: int
    weakest_domain: str
    strongest_domain: str
    variance_across_domains: float
    model_name: str
    mode: str = "mock"
    predictions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    weak_domains: List[str]
    stable_domains: List[str]
    risk_prone_domains: List[str]
    rationale_by_domain: Dict[str, str]
    summary: str


@dataclass
class DomainPlan:
    domain: str
    num_samples: int
    data_type: str
    reason: str

    def __post_init__(self) -> None:
        if self.num_samples < 0:
            raise ValueError("num_samples cannot be negative")


@dataclass
class DataPlan:
    total_budget: int
    strategy: str
    plan: Dict[str, DomainPlan]

    def allocation_sum(self) -> int:
        return sum(item.num_samples for item in self.plan.values())


@dataclass
class GenerationRequest:
    domain: str
    num_samples: int
    data_type: str
    reason: str
    round_id: str = "round_1"


@dataclass
class SFTSample:
    domain: str
    instruction: str
    response: str
    source: str
    generation_model: str
    round_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def __post_init__(self) -> None:
        self.domain = str(self.domain).strip()
        self.instruction = str(self.instruction).strip()
        self.response = str(self.response).strip()
        if not self.domain:
            raise ValueError("domain must be non-empty")


@dataclass
class VerificationResult:
    accepted: List[SFTSample]
    rejected: List[Dict[str, Any]]
    report: Dict[str, Any]


@dataclass
class MixturePlan:
    strategy: str
    samples: List[SFTSample]
    domain_distribution: Dict[str, int]
    dropped_samples: int
    reasons: Dict[str, str]


@dataclass
class TrainingRunResult:
    enabled: bool
    method: str
    status: str
    output_dir: str
    num_train_samples: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopRoundResult:
    run_dir: str
    config: Dict[str, Any]
    evaluation_base: EvaluationResult
    diagnosis: DiagnosisResult
    data_plan: DataPlan
    verification_report: Dict[str, Any]
    mixture_report: Dict[str, Any]
    training_report: TrainingRunResult
    evaluation_after: EvaluationResult
    metrics: Dict[str, Any]

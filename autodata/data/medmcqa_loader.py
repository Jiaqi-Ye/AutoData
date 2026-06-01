"""MedMCQA loading with a deterministic offline fallback for smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from autodata.config import get_target_domains
from autodata.data.schemas import MedMCQAExample
from autodata.utils.io import write_jsonl


DOMAIN_ALIASES = {
    "anat": "Anatomy",
    "anatomy": "Anatomy",
    "pharmacology": "Pharmacology",
    "pharma": "Pharmacology",
    "pathology": "Pathology",
    "path": "Pathology",
    "microbiology": "Microbiology",
    "micro": "Microbiology",
    "physiology": "Physiology",
    "physio": "Physiology",
}


def normalize_domain_name(value: Any) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    key = cleaned.lower().replace("_", " ").replace("-", " ").strip()
    return DOMAIN_ALIASES.get(key, cleaned.title())


def _mock_question(domain: str, index: int, split: str) -> MedMCQAExample:
    answer = ["A", "B", "C", "D"][index % 4]
    options = {
        "A": f"{domain} concept alpha",
        "B": f"{domain} concept beta",
        "C": f"{domain} concept gamma",
        "D": f"{domain} concept delta",
    }
    return MedMCQAExample(
        id=f"mock-{split}-{domain.lower()}-{index}",
        domain=domain,
        question=f"{split.title()} {index}: Which option best matches a core {domain} principle?",
        options=options,
        correct_answer=answer,
        explanation=f"Mock explanation for {domain} item {index}.",
        split=split,
        subject=domain,
    )


def build_mock_medmcqa(
    target_domains: Iterable[str],
    eval_samples_per_domain: int,
    train_pool_samples_per_domain: int,
) -> Tuple[List[MedMCQAExample], List[MedMCQAExample]]:
    eval_examples: List[MedMCQAExample] = []
    train_pool: List[MedMCQAExample] = []
    for domain in target_domains:
        for idx in range(eval_samples_per_domain):
            eval_examples.append(_mock_question(domain, idx, "eval"))
        for idx in range(train_pool_samples_per_domain):
            train_pool.append(_mock_question(domain, idx, "train"))
    return eval_examples, train_pool


def _answer_from_row(row: Dict[str, Any]) -> str:
    raw = row.get("cop", row.get("answer", row.get("correct_answer", "A")))
    if isinstance(raw, int):
        choices = ["A", "B", "C", "D"]
        if 0 <= raw <= 3:
            return choices[raw]
        if 1 <= raw <= 4:
            return choices[raw - 1]
    raw_text = str(raw).strip().upper()
    if raw_text in {"0", "1", "2", "3"}:
        return ["A", "B", "C", "D"][int(raw_text)]
    if raw_text in {"1", "2", "3", "4"}:
        return ["A", "B", "C", "D"][int(raw_text) - 1]
    for letter in ["A", "B", "C", "D"]:
        if letter in raw_text:
            return letter
    return "A"


def _domain_from_row(row: Dict[str, Any]) -> str:
    for key in ("subject_name", "subject", "topic_name", "topic", "domain"):
        if row.get(key):
            return normalize_domain_name(row[key])
    return ""


def _row_to_example(row: Dict[str, Any], split: str, index: int) -> MedMCQAExample:
    domain = _domain_from_row(row)
    options = {
        "A": row.get("opa", row.get("A", "")),
        "B": row.get("opb", row.get("B", "")),
        "C": row.get("opc", row.get("C", "")),
        "D": row.get("opd", row.get("D", "")),
    }
    return MedMCQAExample(
        id=str(row.get("id", f"{split}-{index}")),
        domain=domain,
        question=str(row.get("question", "")).strip(),
        options=options,
        correct_answer=_answer_from_row(row),
        explanation=str(row.get("exp", row.get("explanation", "")) or ""),
        split=split,
        subject=domain,
    )


def _take_by_domain(
    rows: Iterable[Dict[str, Any]],
    split: str,
    target_domains: List[str],
    per_domain: int,
) -> List[MedMCQAExample]:
    target_set = set(target_domains)
    counts = {domain: 0 for domain in target_domains}
    examples: List[MedMCQAExample] = []
    for index, row in enumerate(rows):
        domain = _domain_from_row(row)
        if domain not in target_set or counts[domain] >= per_domain:
            continue
        try:
            example = _row_to_example(row, split=split, index=index)
        except ValueError:
            continue
        if not example.question:
            continue
        examples.append(example)
        counts[domain] += 1
        if all(count >= per_domain for count in counts.values()):
            break
    missing = {domain: per_domain - count for domain, count in counts.items() if count < per_domain}
    if missing:
        raise RuntimeError(f"MedMCQA did not provide enough examples for: {missing}")
    return examples


def load_medmcqa_data(config: Dict[str, Any], run_dir: str | Path | None = None) -> Tuple[List[MedMCQAExample], List[MedMCQAExample]]:
    """Load target-domain MedMCQA examples.

    Smoke mode uses deterministic mock examples, so tests and the first milestone do
    not download large models or datasets.
    """
    dataset_config = config.get("dataset", {})
    target_domains = get_target_domains(config)
    eval_per_domain = int(dataset_config.get("eval_samples_per_domain", 5))
    train_per_domain = int(dataset_config.get("train_pool_samples_per_domain", 20))
    use_mock = bool(dataset_config.get("use_mock_data", False))

    if use_mock:
        eval_examples, train_pool = build_mock_medmcqa(target_domains, eval_per_domain, train_per_domain)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install datasets or set dataset.use_mock_data=true for smoke mode.") from exc

        dataset_name = dataset_config.get("name", "openlifescienceai/medmcqa")
        dataset = load_dataset(dataset_name)
        eval_split = dataset_config.get("eval_split", "validation")
        train_split = dataset_config.get("train_split", "train")
        eval_examples = _take_by_domain(dataset[eval_split], eval_split, target_domains, eval_per_domain)
        train_pool = _take_by_domain(dataset[train_split], train_split, target_domains, train_per_domain)

        eval_ids = {example.id for example in eval_examples}
        train_pool = [example for example in train_pool if example.id not in eval_ids]

    if run_dir is not None:
        processed_dir = Path(run_dir) / "prepared_data"
        write_jsonl(processed_dir / "eval_examples.jsonl", eval_examples)
        write_jsonl(processed_dir / "train_pool.jsonl", train_pool)
    return eval_examples, train_pool


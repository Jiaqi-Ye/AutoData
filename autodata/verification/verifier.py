"""Rule-based verification for generated SFT data."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Tuple

from autodata.config import get_target_domains
from autodata.data.schemas import MedMCQAExample, SFTSample, VerificationResult, to_jsonable
from autodata.evaluation.answer_parser import parse_answer


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


class DataVerifier:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.valid_domains = set(get_target_domains(config))
        verification_config = config.get("verification", {})
        self.near_duplicate_threshold = float(verification_config.get("near_duplicate_threshold", 0.995))
        self.leakage_threshold = float(verification_config.get("leakage_threshold", 0.88))
        self.max_instruction_chars = int(verification_config.get("max_instruction_chars", 4000))
        self.max_response_chars = int(verification_config.get("max_response_chars", 4000))

    def verify(
        self,
        samples: Iterable[SFTSample],
        heldout_eval_examples: Iterable[MedMCQAExample],
    ) -> VerificationResult:
        heldout_questions = [example.question for example in heldout_eval_examples]
        accepted: List[SFTSample] = []
        rejected: List[Dict[str, Any]] = []
        exact_seen = set()
        rejection_counter: Counter[str] = Counter()
        rejected_by_domain: Counter[str] = Counter()
        accepted_by_domain: Counter[str] = Counter()

        for sample in samples:
            is_valid, reason = self._check_sample(sample, exact_seen, accepted, heldout_questions)
            if is_valid:
                exact_seen.add(self._fingerprint(sample))
                accepted.append(sample)
                accepted_by_domain[sample.domain] += 1
            else:
                rejection_counter[reason] += 1
                rejected_by_domain[sample.domain] += 1
                rejected.append({"reason": reason, "sample": to_jsonable(sample)})

        report = {
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "accepted_by_domain": dict(accepted_by_domain),
            "rejected_by_domain": dict(rejected_by_domain),
            "rejection_reasons": dict(rejection_counter),
            "near_duplicate_threshold": self.near_duplicate_threshold,
            "leakage_threshold": self.leakage_threshold,
        }
        return VerificationResult(accepted=accepted, rejected=rejected, report=report)

    def _check_sample(
        self,
        sample: SFTSample,
        exact_seen: set[str],
        accepted: List[SFTSample],
        heldout_questions: List[str],
    ) -> Tuple[bool, str]:
        if sample.domain not in self.valid_domains:
            return False, "invalid_domain"
        if not sample.instruction.strip():
            return False, "empty_instruction"
        if not sample.response.strip():
            return False, "empty_response"
        if len(sample.instruction) > self.max_instruction_chars:
            return False, "instruction_too_long"
        if len(sample.response) > self.max_response_chars:
            return False, "response_too_long"
        if parse_answer(sample.response) is None:
            return False, "missing_clear_answer"
        fingerprint = self._fingerprint(sample)
        if fingerprint in exact_seen:
            return False, "duplicate"
        if self._is_near_duplicate(sample, accepted):
            return False, "near_duplicate"
        if self._leaks_eval_question(sample, heldout_questions):
            return False, "heldout_leakage"
        return True, "accepted"

    def _fingerprint(self, sample: SFTSample) -> str:
        return normalize_text(sample.domain + " " + sample.instruction + " " + sample.response)

    def _is_near_duplicate(self, sample: SFTSample, accepted: List[SFTSample]) -> bool:
        for existing in accepted:
            if sample.domain != existing.domain:
                continue
            combined_score = text_similarity(
                sample.instruction + " " + sample.response,
                existing.instruction + " " + existing.response,
            )
            if combined_score >= self.near_duplicate_threshold:
                return True
        return False

    def _leaks_eval_question(self, sample: SFTSample, heldout_questions: List[str]) -> bool:
        sample_text = normalize_text(sample.instruction)
        for question in heldout_questions:
            question_text = normalize_text(question)
            if not question_text:
                continue
            if question_text in sample_text:
                return True
            if text_similarity(sample.instruction, question) >= self.leakage_threshold:
                return True
        return False

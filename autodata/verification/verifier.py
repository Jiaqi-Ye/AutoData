"""Rule-based verification for generated SFT data."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Tuple

from autodata.config import get_target_domains
from autodata.data.schemas import MedMCQAExample, SFTSample, VerificationResult, to_jsonable
from autodata.evaluation.answer_parser import parse_answer

OPTION_PATTERN = re.compile(r"(?im)^\s*([ABCD])[\.\)]\s+\S+")
OPTION_TEXT_PATTERN = re.compile(r"(?im)^\s*([ABCD])[\.\)]\s*(.+?)\s*$")
RESPONSE_PREFIX_PATTERN = re.compile(r"^\s*The correct answer is ([ABCD])\.", re.IGNORECASE)
AMBIGUOUS_STEM_PATTERNS = [
    re.compile(r"\bmain branches\b", re.IGNORECASE),
    re.compile(r"\bknown for (?:its|their) ability to cause\b", re.IGNORECASE),
    re.compile(r"\bhigh levels of virulence factors\b", re.IGNORECASE),
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "best",
    "by",
    "can",
    "cause",
    "correct",
    "due",
    "explanation",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "leading",
    "of",
    "option",
    "or",
    "production",
    "question",
    "secretion",
    "the",
    "this",
    "to",
    "which",
    "with",
}


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
        self.option_duplicate_threshold = float(verification_config.get("option_duplicate_threshold", 0.98))
        self.option_response_overlap_threshold = float(
            verification_config.get("option_response_overlap_threshold", 0.40)
        )

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
        if sample.source == "local_hf" and sample.metadata.get("parse_error") != "none":
            return False, "invalid_generation_json"
        options = self._options(sample.instruction)
        if set(options) != {"A", "B", "C", "D"}:
            return False, "missing_mcq_options"
        response_answer = self._response_answer(sample.response)
        if response_answer is None:
            return False, "response_must_start_with_answer"
        if parse_answer(sample.response) is None:
            return False, "missing_clear_answer"
        if response_answer not in options:
            return False, "answer_option_missing"
        if self._has_duplicate_options(options):
            return False, "ambiguous_duplicate_options"
        if self._has_ambiguous_stem(sample.instruction):
            return False, "ambiguous_question_stem"
        if sample.source != "mock" and not self._answer_matches_option(sample.response, response_answer, options):
            return False, "answer_option_mismatch"
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

    def _option_labels(self, instruction: str) -> set[str]:
        return {match.group(1).upper() for match in OPTION_PATTERN.finditer(instruction)}

    def _options(self, instruction: str) -> Dict[str, str]:
        return {
            match.group(1).upper(): match.group(2).strip()
            for match in OPTION_TEXT_PATTERN.finditer(instruction)
            if match.group(2).strip()
        }

    def _response_answer(self, response: str) -> str | None:
        match = RESPONSE_PREFIX_PATTERN.search(response)
        if not match:
            return None
        return match.group(1).upper()

    def _has_duplicate_options(self, options: Dict[str, str]) -> bool:
        items = list(options.items())
        for index, (_, left) in enumerate(items):
            left_norm = normalize_text(left)
            for _, right in items[index + 1 :]:
                right_norm = normalize_text(right)
                if left_norm and left_norm == right_norm:
                    return True
                if (
                    text_similarity(left, right) >= self.option_duplicate_threshold
                    and self._content_tokens(left) == self._content_tokens(right)
                ):
                    return True
        return False

    def _has_ambiguous_stem(self, instruction: str) -> bool:
        question = self._question_text(instruction)
        return any(pattern.search(question) for pattern in AMBIGUOUS_STEM_PATTERNS)

    def _answer_matches_option(self, response: str, answer: str, options: Dict[str, str]) -> bool:
        selected_text = options[answer]
        selected_score = self._option_response_overlap(selected_text, response)
        other_scores = [
            self._option_response_overlap(text, response)
            for label, text in options.items()
            if label != answer
        ]
        if selected_score >= self.option_response_overlap_threshold:
            return True
        if other_scores and max(other_scores) >= selected_score + 0.25:
            return False
        return selected_score > 0

    def _option_response_overlap(self, option_text: str, response: str) -> float:
        option_tokens = self._content_tokens(option_text)
        if not option_tokens:
            return 0.0
        response_tokens = self._content_tokens(response)
        overlap = option_tokens.intersection(response_tokens)
        return len(overlap) / len(option_tokens)

    def _content_tokens(self, text: str) -> set[str]:
        tokens = set(normalize_text(text).split())
        return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}

    def _question_text(self, instruction: str) -> str:
        lines = [line.strip() for line in instruction.splitlines() if line.strip()]
        question_lines = [line for line in lines if not OPTION_TEXT_PATTERN.match(line)]
        question = " ".join(question_lines)
        return re.sub(r"^\s*Question:\s*", "", question, flags=re.IGNORECASE).strip()

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

"""Robust answer-letter extraction for MCQ model outputs."""

from __future__ import annotations

import re
from typing import Optional


ANSWER_PATTERNS = [
    re.compile(r"\banswer\s*(?:is|:|-)?\s*\(?([ABCD])\)?\b", re.IGNORECASE),
    re.compile(r"\bcorrect\s+answer\s*(?:is|:|-)?\s*\(?([ABCD])\)?\b", re.IGNORECASE),
    re.compile(r"\boption\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([ABCD])\)?[\s\.\):,-]*$", re.IGNORECASE),
    re.compile(r"\b([ABCD])\b", re.IGNORECASE),
]


def parse_answer(text: str) -> Optional[str]:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return match.group(1).upper()
    return None


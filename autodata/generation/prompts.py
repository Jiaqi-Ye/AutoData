"""Prompt templates for leakage-safe synthetic data generation."""

from __future__ import annotations

from autodata.data.schemas import GenerationRequest


DOMAIN_TOPICS = {
    "Anatomy": "gross anatomy, anatomical relationships, innervation, blood supply, and clinical anatomy",
    "Pharmacology": "drug mechanisms, adverse effects, contraindications, interactions, and therapeutic choices",
    "Pathology": "disease mechanisms, histology clues, clinical presentation, and diagnostic reasoning",
    "Microbiology": "organisms, virulence factors, diagnostic tests, treatment, and prevention",
    "Physiology": "homeostasis, organ-system mechanisms, feedback loops, and quantitative reasoning",
}

FOCUS_TERMS = {
    "Anatomy": ["cranial nerves", "limb compartments", "blood supply", "peritoneal spaces", "spinal roots"],
    "Pharmacology": ["receptor agonists", "renal dosing", "drug toxicity", "enzyme induction", "antidotes"],
    "Pathology": ["granulomas", "neoplasia", "infarction", "immune injury", "cell adaptation"],
    "Microbiology": ["culture findings", "viral replication", "toxins", "antibiotic choice", "vaccines"],
    "Physiology": ["cardiac preload", "renal clearance", "acid base", "endocrine feedback", "gas exchange"],
}

CASE_STEMS = [
    "A short clinical vignette asks about",
    "A mechanism question focuses on",
    "A board-style item compares",
    "A laboratory interpretation item tests",
    "A treatment reasoning question reviews",
]

OPTION_SETS = [
    ("A compensatory mechanism", "The best-supported answer", "A tempting distractor", "An unrelated finding"),
    ("An early change", "A late complication", "The correct mechanism", "A normal variant"),
    ("A contraindicated choice", "A reasonable distractor", "The preferred option", "A historical association"),
    ("A false pairing", "The expected result", "A nonspecific symptom", "A rare exception"),
]


def build_generation_prompt(request: GenerationRequest, sample_index: int) -> str:
    topic = DOMAIN_TOPICS.get(request.domain, f"core {request.domain} knowledge")
    return (
        "Create one original medical multiple-choice training example for supervised fine-tuning.\n"
        "Do not quote or paraphrase any held-out evaluation question.\n"
        f"Domain: {request.domain}\n"
        f"Broad topic scope: {topic}\n"
        f"Desired data type: {request.data_type}\n"
        f"Planning rationale: {request.reason}\n"
        f"Sample index: {sample_index}\n"
        "Return JSON with fields: domain, instruction, response.\n"
        "The instruction should contain a question with A/B/C/D options.\n"
        "The response must start with 'The correct answer is X.' and include a short explanation."
    )


def build_mock_instruction(domain: str, sample_index: int) -> str:
    terms = FOCUS_TERMS.get(domain, [f"core {domain} knowledge"])
    focus = terms[sample_index % len(terms)]
    stem = CASE_STEMS[sample_index % len(CASE_STEMS)]
    options = OPTION_SETS[sample_index % len(OPTION_SETS)]
    return (
        f"Answer the following {domain} multiple-choice question.\n\n"
        f"Question: {stem} {focus}. Which option is most appropriate for synthetic concept {sample_index}?\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}"
    )


def build_mock_response(domain: str, sample_index: int) -> str:
    answer = ["A", "B", "C", "D"][sample_index % 4]
    terms = FOCUS_TERMS.get(domain, [f"core {domain} knowledge"])
    focus = terms[sample_index % len(terms)]
    return (
        f"The correct answer is {answer}. Explanation: This leakage-safe mock item targets {focus} "
        f"in {domain} using broad curriculum information rather than held-out question text."
    )

"""Rule-based model weakness diagnosis."""

from __future__ import annotations

from typing import Any, Dict, Optional

from autodata.data.schemas import DiagnosisResult, EvaluationResult


class WeaknessDiagnoser:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def diagnose(
        self,
        evaluation: EvaluationResult,
        previous_evaluation: Optional[EvaluationResult] = None,
    ) -> DiagnosisResult:
        accuracies = {domain: metrics.accuracy for domain, metrics in evaluation.per_domain.items()}
        if not accuracies:
            return DiagnosisResult([], [], [], {}, "No domains were evaluated.")

        mean_accuracy = sum(accuracies.values()) / len(accuracies)
        sorted_domains = sorted(accuracies, key=accuracies.get)
        weak_cutoff = min(mean_accuracy, 0.55)
        strong_cutoff = max(mean_accuracy, 0.70)

        weak_domains = [domain for domain in sorted_domains if accuracies[domain] <= weak_cutoff]
        if not weak_domains:
            weak_domains = sorted_domains[:1]

        stable_domains = [domain for domain in sorted_domains if accuracies[domain] >= strong_cutoff]
        if not stable_domains:
            stable_domains = sorted_domains[-1:]

        risk_prone_domains = []
        rationale_by_domain: Dict[str, str] = {}
        for domain, metrics in evaluation.per_domain.items():
            previous_gain = None
            if previous_evaluation and domain in previous_evaluation.per_domain:
                previous_gain = metrics.accuracy - previous_evaluation.per_domain[domain].accuracy

            if domain in weak_domains:
                rationale = (
                    f"{domain} is below the current target threshold "
                    f"({metrics.accuracy:.2f} accuracy over {metrics.num_samples} samples)."
                )
            elif domain in stable_domains:
                rationale = (
                    f"{domain} is comparatively strong "
                    f"({metrics.accuracy:.2f} accuracy) and should receive preservation data."
                )
                risk_prone_domains.append(domain)
            else:
                rationale = (
                    f"{domain} is near the middle of the domain distribution "
                    f"({metrics.accuracy:.2f} accuracy)."
                )

            if metrics.num_samples < 20:
                rationale += " The sample size is small, so the diagnosis should be treated as noisy."
            if previous_gain is not None:
                rationale += f" Previous-round gain was {previous_gain:.3f}."
                if previous_gain < -0.02 and domain not in risk_prone_domains:
                    risk_prone_domains.append(domain)
            rationale_by_domain[domain] = rationale

        summary = (
            f"Weak domains: {', '.join(weak_domains)}. "
            f"Stable domains: {', '.join(stable_domains)}. "
            f"Risk-prone domains: {', '.join(risk_prone_domains) or 'none'}."
        )
        return DiagnosisResult(
            weak_domains=weak_domains,
            stable_domains=stable_domains,
            risk_prone_domains=risk_prone_domains,
            rationale_by_domain=rationale_by_domain,
            summary=summary,
        )


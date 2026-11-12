"""Evidence-aware deterministic inference primitives for MIND-3.

This module evaluates explicit claims and rules, separates source reliability
from inference validity, and abstains when required premises are missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class Claim:
    subject: str
    predicate: str
    value: Any
    source: str
    source_reliability: float
    verification: str
    language: str = "en"
    claim_id: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.source_reliability <= 1.0:
            raise ValueError("source_reliability must be in [0, 1]")
        if not self.subject or not self.predicate or not self.source:
            raise ValueError("subject, predicate, and source are required")


@dataclass(frozen=True)
class Rule:
    rule_id: str
    required: tuple[tuple[str, str], ...]
    conclusion: str
    operator: Callable[[dict[str, Any]], Any]
    description: str = ""


@dataclass
class InferenceResult:
    status: str
    value: Any = None
    evidence: list[Claim] = field(default_factory=list)
    rule_id: str | None = None
    source_reliability: float | None = None
    reason: str = ""


class KnowledgeBase:
    def __init__(self, claims: Iterable[Claim] = ()) -> None:
        self._claims: dict[tuple[str, str], list[Claim]] = {}
        for claim in claims:
            self.add(claim)

    def add(self, claim: Claim) -> None:
        self._claims.setdefault((claim.subject, claim.predicate), []).append(claim)

    def get(self, subject: str, predicate: str) -> list[Claim]:
        return list(self._claims.get((subject, predicate), []))

    def resolve(self, subject: str, predicate: str) -> InferenceResult:
        matches = self.get(subject, predicate)
        if not matches:
            return InferenceResult("missing", reason=f"No claim for {subject}.{predicate}")
        if len({repr(c.value) for c in matches}) > 1:
            return InferenceResult("conflict", evidence=matches,
                                   reason=f"Conflicting values for {subject}.{predicate}")
        accepted = [c for c in matches if c.verification in
                    {"verified", "user_confirmed", "official"}]
        if not accepted:
            return InferenceResult("unverified", evidence=matches,
                                   reason="No claim has an accepted verification state")
        best = max(accepted, key=lambda c: c.source_reliability)
        return InferenceResult("supported", best.value, accepted,
                               source_reliability=best.source_reliability,
                               reason="Resolved from explicit accepted evidence")


class AnalyticalEngine:
    """Small forward-chaining engine with explicit premises and abstention."""

    def __init__(self, knowledge: KnowledgeBase, rules: Iterable[Rule] = ()) -> None:
        self.knowledge = knowledge
        self.rules = {rule.rule_id: rule for rule in rules}

    def derive(self, rule_id: str, subject: str) -> InferenceResult:
        rule = self.rules.get(rule_id)
        if rule is None:
            return InferenceResult("invalid_rule", reason=f"Unknown rule: {rule_id}")
        premises: dict[str, Any] = {}
        evidence: list[Claim] = []
        for predicate, alias in rule.required:
            resolved = self.knowledge.resolve(subject, predicate)
            if resolved.status != "supported":
                return InferenceResult(
                    resolved.status, evidence=resolved.evidence, rule_id=rule_id,
                    reason=f"Required premise '{predicate}' is {resolved.status}: {resolved.reason}",
                )
            premises[alias] = resolved.value
            evidence.extend(resolved.evidence)
        try:
            value = rule.operator(premises)
        except (ArithmeticError, TypeError, ValueError, KeyError) as exc:
            return InferenceResult(
                "invalid_derivation", evidence=evidence, rule_id=rule_id,
                reason=f"Rule execution failed safely: {type(exc).__name__}: {exc}",
            )
        reliability = min(c.source_reliability for c in evidence) if evidence else None
        return InferenceResult(
            "derived", value=value, evidence=evidence, rule_id=rule_id,
            source_reliability=reliability,
            reason="All required premises resolved and the declared rule executed",
        )


def completed_years(birth_date: str, reference_date: str) -> int:
    """Calculate completed years from ISO YYYY-MM-DD dates."""
    from datetime import date
    birth = date.fromisoformat(birth_date)
    reference = date.fromisoformat(reference_date)
    if reference < birth:
        raise ValueError("reference date precedes birth date")
    return reference.year - birth.year - (
        (reference.month, reference.day) < (birth.month, birth.day)
    )


def default_rules() -> list[Rule]:
    return [Rule(
        rule_id="completed_age",
        required=(("date_of_birth", "dob"), ("reference_date", "today")),
        conclusion="age",
        operator=lambda p: completed_years(p["dob"], p["today"]),
        description="Completed years between date of birth and reference date",
    )]


def verify_claim_against_derivation(
    direct_claim: InferenceResult, derived_claim: InferenceResult
) -> InferenceResult:
    if direct_claim.status != "supported" or derived_claim.status != "derived":
        return InferenceResult(
            "insufficient_evidence",
            evidence=direct_claim.evidence + derived_claim.evidence,
            reason="A supported direct claim and a valid derivation are both required",
        )
    evidence = direct_claim.evidence + derived_claim.evidence
    if direct_claim.value != derived_claim.value:
        return InferenceResult(
            "conflict", evidence=evidence,
            reason=f"Direct value {direct_claim.value!r} conflicts with derived value {derived_claim.value!r}",
        )
    return InferenceResult("consistent", derived_claim.value, evidence,
                           reason="Direct and derived values agree")

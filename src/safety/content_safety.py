"""
content_safety.py
-----------------
Dual screening — input question and generated answer.
"""

from dataclasses import dataclass
from enum import IntEnum

import structlog
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from azure.identity import DefaultAzureCredential

from config.settings import settings

log = structlog.get_logger()


class SafetyDecision(IntEnum):
    SAFE = 0
    BLOCKED = 1


@dataclass
class SafetyResult:
    decision: SafetyDecision
    flagged_categories: list[str]
    max_severity: int
    raw_scores: dict[str, int]

    @property
    def is_safe(self) -> bool:
        return self.decision == SafetyDecision.SAFE


def get_safety_client() -> ContentSafetyClient:
    return ContentSafetyClient(
        endpoint=settings.azure_content_safety_endpoint,
        credential=DefaultAzureCredential(),
    )


def check_text(text: str, client: ContentSafetyClient | None = None) -> SafetyResult:
    cs_client = client or get_safety_client()

    options = AnalyzeTextOptions(
        text=text[:10_000],
        categories=[
            TextCategory.HATE,
            TextCategory.VIOLENCE,
            TextCategory.SEXUAL,
            TextCategory.SELF_HARM,
        ],
        output_type="FourSeverityLevels",
    )

    response = cs_client.analyze_text(options)
    raw_scores: dict[str, int] = {}
    flagged: list[str] = []

    for item in response.categories_analysis:
        severity = item.severity or 0
        raw_scores[item.category] = severity
        if severity >= settings.content_safety_threshold:
            flagged.append(item.category)

    max_severity = max(raw_scores.values(), default=0)
    decision = SafetyDecision.BLOCKED if flagged else SafetyDecision.SAFE

    if decision == SafetyDecision.BLOCKED:
        log.warning("content_blocked", categories=flagged, severity=max_severity)

    return SafetyResult(
        decision=decision,
        flagged_categories=flagged,
        max_severity=max_severity,
        raw_scores=raw_scores,
    )


BLOCKED_RESPONSE = (
    "I'm sorry, but I can't process that request. "
    "The content was flagged by our safety system."
)
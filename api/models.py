from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    source_type: Literal["doc", "ticket", "changelog"] | None = Field(
        default=None,
        description="Filter results to a specific source type.",
    )
    after_date: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Filter tickets/changelogs after this ISO date (YYYY-MM-DD).",
    )
    version: str | None = Field(
        default=None,
        description="Filter changelog to a specific version e.g. '2.1.0'.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "How do I reset my password?"},
                {"question": "What changed in v2.0?", "source_type": "changelog", "version": "2.0.0"},
                {"question": "Any open tickets about login?", "source_type": "ticket", "after_date": "2025-01-01"},
            ]
        }
    }


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    source_types: list[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
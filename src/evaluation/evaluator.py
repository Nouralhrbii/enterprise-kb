"""
evaluator.py
------------
Evaluate the enterprise knowledge base pipeline.
Extended to support source_type filtering per sample.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import structlog
from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    CoherenceEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
)
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config.settings import settings
from src.generation.generator import GenerationResult, generate_answer
from src.retrieval.reranker import rerank
from src.retrieval.searcher import search

log = structlog.get_logger()


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    source_type: str | None = None   # optional filter for this question


@dataclass
class EvalResult:
    question: str
    generated_answer: str
    context: str
    groundedness: float
    relevance: float
    coherence: float
    sources: list[str]
    source_types: list[str]
    total_tokens: int
    cost_usd: float

    @property
    def avg_score(self) -> float:
        return (self.groundedness + self.relevance + self.coherence) / 3


def _get_model_config() -> AzureOpenAIModelConfiguration:
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAIModelConfiguration(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_deployment=settings.azure_openai_chat_deployment,
        api_version=settings.azure_openai_api_version,
        azure_ad_token_provider=token_provider,
    )


def evaluate_sample(
    sample: EvalSample,
    model_config: AzureOpenAIModelConfiguration | None = None,
) -> EvalResult:
    cfg = model_config or _get_model_config()

    candidates = search(sample.question, source_type=sample.source_type)
    chunks = rerank(sample.question, candidates, top_k=settings.top_k_results)
    result: GenerationResult = generate_answer(sample.question, chunks)
    context = "\n\n".join(c.content for c in chunks)

    g_score = GroundednessEvaluator(model_config=cfg)(
        response=result.answer, context=context
    )
    r_score = RelevanceEvaluator(model_config=cfg)(
        query=sample.question, response=result.answer, context=context
    )
    c_score = CoherenceEvaluator(model_config=cfg)(
        query=sample.question, response=result.answer
    )

    return EvalResult(
        question=sample.question,
        generated_answer=result.answer,
        context=context,
        groundedness=g_score.get("groundedness", 0),
        relevance=r_score.get("relevance", 0),
        coherence=c_score.get("coherence", 0),
        sources=result.sources,
        source_types=result.source_types,
        total_tokens=result.total_tokens,
        cost_usd=result.cost_usd,
    )


def run_evaluation(
    samples: list[EvalSample],
    output_path: str = "docs/eval_report.json",
) -> dict:
    model_config = _get_model_config()
    results: list[EvalResult] = []

    for i, sample in enumerate(samples):
        log.info("evaluating", index=i + 1, total=len(samples))
        try:
            results.append(evaluate_sample(sample, model_config))
        except Exception as exc:
            log.error("eval_failed", question=sample.question[:60], error=str(exc))

    summary = {
        "total_samples": len(results),
        "avg_groundedness": round(sum(r.groundedness for r in results) / len(results), 2),
        "avg_relevance": round(sum(r.relevance for r in results) / len(results), 2),
        "avg_coherence": round(sum(r.coherence for r in results) / len(results), 2),
        "avg_score": round(sum(r.avg_score for r in results) / len(results), 2),
        "total_tokens": sum(r.total_tokens for r in results),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
        "cost_per_query_usd": round(sum(r.cost_usd for r in results) / len(results), 5),
        "detailed_results": [
            {
                "question": r.question,
                "groundedness": r.groundedness,
                "relevance": r.relevance,
                "coherence": r.coherence,
                "sources": r.sources,
                "source_types": r.source_types,
                "tokens": r.total_tokens,
            }
            for r in results
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2))
    log.info("eval_report_saved", path=output_path)
    return summary
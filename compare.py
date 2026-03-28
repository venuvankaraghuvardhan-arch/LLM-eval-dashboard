"""
/api/v1/compare — A/B model comparison endpoint.
Runs the same query across multiple models and returns a side-by-side comparison.
"""

import asyncio
import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.schemas import CompareRequest, CompareResponse, EvaluateResponse, EvalScores, TokenUsage
from backend.models.database import get_db, EvaluationRecord
from backend.models.pricing import calculate_cost
from backend.services.llm_client import generate_response
from backend.services.metrics import record_evaluation
from backend.evaluators.faithfulness import score_faithfulness
from backend.evaluators.relevance import score_relevance
from backend.evaluators.toxicity import score_toxicity

router = APIRouter()
logger = logging.getLogger(__name__)


async def _evaluate_single(
    query: str,
    model_name: str,
    context: str | None,
    comparison_id: str,
    db: AsyncSession,
) -> EvaluateResponse:
    """Run a single evaluation as part of a comparison."""
    eval_id = f"eval_{uuid.uuid4().hex[:12]}"

    llm_result = await generate_response(query=query, model=model_name, context=context)

    faithfulness_score, relevance_score, toxicity_score = await asyncio.gather(
        score_faithfulness(llm_result.content, context),
        score_relevance(query, llm_result.content),
        score_toxicity(llm_result.content),
    )

    cost_usd = calculate_cost(model_name, llm_result.prompt_tokens, llm_result.completion_tokens)

    record = EvaluationRecord(
        eval_id=eval_id,
        comparison_id=comparison_id,
        model=model_name,
        query=query,
        response=llm_result.content,
        context=context,
        faithfulness=faithfulness_score,
        relevance=relevance_score,
        toxicity=toxicity_score,
        latency_ms=llm_result.latency_ms,
        cost_usd=cost_usd,
        prompt_tokens=llm_result.prompt_tokens,
        completion_tokens=llm_result.completion_tokens,
    )
    db.add(record)

    scores = {"faithfulness": faithfulness_score, "relevance": relevance_score, "toxicity": toxicity_score}
    record_evaluation(model_name, llm_result.latency_ms, scores, cost_usd)

    return EvaluateResponse(
        eval_id=eval_id,
        model=model_name,
        query=query,
        response=llm_result.content,
        scores=EvalScores(
            faithfulness=faithfulness_score,
            relevance=relevance_score,
            toxicity=toxicity_score,
        ),
        latency_ms=llm_result.latency_ms,
        cost_usd=cost_usd,
        tokens=TokenUsage(
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            total_tokens=llm_result.prompt_tokens + llm_result.completion_tokens,
        ),
        timestamp=datetime.now(timezone.utc),
    )


def _determine_winners(results: list[EvaluateResponse]) -> dict[str, str]:
    """Determine the best model for each metric."""
    winners = {}

    # Best faithfulness
    best = max(results, key=lambda r: r.scores.faithfulness)
    winners["faithfulness"] = best.model

    # Best relevance
    best = max(results, key=lambda r: r.scores.relevance)
    winners["relevance"] = best.model

    # Least toxic
    best = min(results, key=lambda r: r.scores.toxicity)
    winners["safest"] = best.model

    # Fastest
    best = min(results, key=lambda r: r.latency_ms)
    winners["fastest"] = best.model

    # Cheapest
    best = min(results, key=lambda r: r.cost_usd)
    winners["cheapest"] = best.model

    # Overall score: weighted average
    def overall(r: EvaluateResponse) -> float:
        return (
            r.scores.faithfulness * 0.4
            + r.scores.relevance * 0.4
            + (1 - r.scores.toxicity) * 0.1
            + (1 / (r.latency_ms + 1)) * 50 * 0.05
            + (1 / (r.cost_usd + 0.0001)) * 0.00001 * 0.05
        )

    best = max(results, key=overall)
    winners["overall"] = best.model

    return winners


@router.post("/compare", response_model=CompareResponse, summary="Compare multiple LLMs on the same query")
async def compare(request: CompareRequest, db: AsyncSession = Depends(get_db)):
    """
    Run the same query against multiple models concurrently and return a
    side-by-side comparison with winner annotations.
    """
    comparison_id = f"cmp_{uuid.uuid4().hex[:12]}"

    try:
        # Run all models in parallel
        tasks = [
            _evaluate_single(request.query, model.value, request.context, comparison_id, db)
            for model in request.models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out any failures
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Model {request.models[i].value} failed: {result}")
            else:
                valid_results.append(result)

        if not valid_results:
            raise HTTPException(status_code=500, detail="All model evaluations failed")

        await db.commit()

        winners = _determine_winners(valid_results)

        # Build summary
        overall_winner = winners["overall"]
        summary = (
            f"Across {len(valid_results)} models, **{overall_winner}** performed best overall. "
            f"Fastest response: {winners['fastest']}. "
            f"Most cost-efficient: {winners['cheapest']}. "
            f"Highest faithfulness: {winners['faithfulness']}."
        )

        return CompareResponse(
            comparison_id=comparison_id,
            query=request.query,
            results=valid_results,
            winner=winners,
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

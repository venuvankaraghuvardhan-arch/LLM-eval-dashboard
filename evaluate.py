"""
/api/v1/evaluate — Single query evaluation endpoint.
"""

import asyncio
import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.schemas import EvaluateRequest, EvaluateResponse, EvalScores, TokenUsage
from backend.models.database import get_db, EvaluationRecord
from backend.models.pricing import calculate_cost
from backend.services.llm_client import generate_response
from backend.services.metrics import record_evaluation, active_evaluations_gauge
from backend.evaluators.faithfulness import score_faithfulness
from backend.evaluators.relevance import score_relevance
from backend.evaluators.toxicity import score_toxicity

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/evaluate", response_model=EvaluateResponse, summary="Evaluate a single LLM query")
async def evaluate(request: EvaluateRequest, db: AsyncSession = Depends(get_db)):
    """
    Run a full evaluation on a single query:
    - Generates LLM response (or uses provided response)
    - Scores faithfulness, relevance, toxicity
    - Tracks cost and latency
    - Persists to DB and updates Prometheus metrics
    """
    active_evaluations_gauge.inc()
    eval_id = f"eval_{uuid.uuid4().hex[:12]}"

    try:
        model_name = request.model.value

        # Step 1: Generate or use provided response
        if request.response:
            # Evaluate a pre-generated response
            from backend.services.llm_client import LLMResponse
            llm_result = LLMResponse(
                content=request.response,
                prompt_tokens=len(request.query.split()) * 2,
                completion_tokens=len(request.response.split()) * 2,
                latency_ms=0.0,
                model=model_name,
            )
        else:
            llm_result = await generate_response(
                query=request.query,
                model=model_name,
                context=request.context,
            )

        # Step 2: Run all evaluators concurrently
        faithfulness_score, relevance_score, toxicity_score = await asyncio.gather(
            score_faithfulness(llm_result.content, request.context),
            score_relevance(request.query, llm_result.content),
            score_toxicity(llm_result.content),
        )

        # Step 3: Calculate cost
        cost_usd = calculate_cost(model_name, llm_result.prompt_tokens, llm_result.completion_tokens)

        # Step 4: Persist to DB
        record = EvaluationRecord(
            eval_id=eval_id,
            model=model_name,
            query=request.query,
            response=llm_result.content,
            context=request.context,
            faithfulness=faithfulness_score,
            relevance=relevance_score,
            toxicity=toxicity_score,
            latency_ms=llm_result.latency_ms,
            cost_usd=cost_usd,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
        )
        db.add(record)
        await db.commit()

        # Step 5: Update Prometheus
        scores = {"faithfulness": faithfulness_score, "relevance": relevance_score, "toxicity": toxicity_score}
        record_evaluation(model_name, llm_result.latency_ms, scores, cost_usd)

        logger.info(f"[{eval_id}] model={model_name} faith={faithfulness_score:.2f} rel={relevance_score:.2f} tox={toxicity_score:.3f} cost=${cost_usd:.6f}")

        return EvaluateResponse(
            eval_id=eval_id,
            model=model_name,
            query=request.query,
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

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

    finally:
        active_evaluations_gauge.dec()

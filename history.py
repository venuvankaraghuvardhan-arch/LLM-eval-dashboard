"""
/api/v1/history — Evaluation history and aggregate analytics endpoints.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.models.database import get_db, EvaluationRecord
from backend.models.schemas import HistoryResponse, HistoryItem, AggregateStats

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/history", response_model=HistoryResponse, summary="Get evaluation history")
async def get_history(
    model: Optional[str] = Query(None, description="Filter by model name"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Return paginated evaluation history, optionally filtered by model."""
    stmt = select(EvaluationRecord).order_by(EvaluationRecord.timestamp.desc())

    if model:
        stmt = stmt.where(EvaluationRecord.model == model)

    count_stmt = select(func.count()).select_from(EvaluationRecord)
    if model:
        count_stmt = count_stmt.where(EvaluationRecord.model == model)

    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0

    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    records = result.scalars().all()

    items = [
        HistoryItem(
            eval_id=r.eval_id,
            model=r.model,
            query=r.query[:100] + "..." if len(r.query) > 100 else r.query,
            faithfulness=r.faithfulness,
            relevance=r.relevance,
            toxicity=r.toxicity,
            latency_ms=r.latency_ms,
            cost_usd=r.cost_usd,
            timestamp=r.timestamp,
        )
        for r in records
    ]

    return HistoryResponse(
        items=items,
        total=total,
        page=offset // limit + 1,
        page_size=limit,
    )


@router.get("/stats", response_model=list[AggregateStats], summary="Aggregate stats per model")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Return aggregate performance metrics grouped by model."""
    stmt = select(
        EvaluationRecord.model,
        func.avg(EvaluationRecord.faithfulness).label("avg_faithfulness"),
        func.avg(EvaluationRecord.relevance).label("avg_relevance"),
        func.avg(EvaluationRecord.toxicity).label("avg_toxicity"),
        func.avg(EvaluationRecord.latency_ms).label("avg_latency_ms"),
        func.avg(EvaluationRecord.cost_usd).label("avg_cost_usd"),
        func.count(EvaluationRecord.id).label("total_queries"),
        func.sum(EvaluationRecord.cost_usd).label("total_cost_usd"),
    ).group_by(EvaluationRecord.model)

    result = await db.execute(stmt)
    rows = result.all()

    return [
        AggregateStats(
            model=row.model,
            avg_faithfulness=round(row.avg_faithfulness or 0, 3),
            avg_relevance=round(row.avg_relevance or 0, 3),
            avg_toxicity=round(row.avg_toxicity or 0, 4),
            avg_latency_ms=round(row.avg_latency_ms or 0, 1),
            avg_cost_usd=round(row.avg_cost_usd or 0, 8),
            total_queries=row.total_queries,
            total_cost_usd=round(row.total_cost_usd or 0, 6),
        )
        for row in rows
    ]


@router.get("/history/{eval_id}", summary="Get a single evaluation by ID")
async def get_evaluation(eval_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch the full details of a single evaluation by its ID."""
    stmt = select(EvaluationRecord).where(EvaluationRecord.eval_id == eval_id)
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()

    if not record:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Evaluation {eval_id} not found")

    return {
        "eval_id": record.eval_id,
        "model": record.model,
        "query": record.query,
        "response": record.response,
        "context": record.context,
        "scores": {
            "faithfulness": record.faithfulness,
            "relevance": record.relevance,
            "toxicity": record.toxicity,
        },
        "latency_ms": record.latency_ms,
        "cost_usd": record.cost_usd,
        "tokens": {
            "prompt": record.prompt_tokens,
            "completion": record.completion_tokens,
            "total": record.prompt_tokens + record.completion_tokens,
        },
        "timestamp": record.timestamp,
    }

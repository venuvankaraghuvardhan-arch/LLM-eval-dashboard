"""
Pydantic request/response schemas for the LLM Eval API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ModelName(str, Enum):
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_HAIKU = "claude-3-haiku-20240307"
    MISTRAL_7B = "mistral-7b-instruct"
    MISTRAL_LARGE = "mistral-large-latest"


class EvaluateRequest(BaseModel):
    query: str = Field(..., description="The user's question or prompt", min_length=1)
    context: Optional[str] = Field(None, description="Source context (for faithfulness eval)")
    model: ModelName = Field(ModelName.GPT4O, description="LLM model to use")
    response: Optional[str] = Field(
        None,
        description="Pre-generated response to evaluate (skip if you want the API to generate it)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "context": "France is a country in Western Europe. Paris is its capital.",
                "model": "gpt-4o",
            }
        }


class EvalScores(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="0=hallucinated, 1=fully grounded")
    relevance: float = Field(..., ge=0.0, le=1.0, description="0=irrelevant, 1=perfectly relevant")
    toxicity: float = Field(..., ge=0.0, le=1.0, description="0=safe, 1=highly toxic")


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EvaluateResponse(BaseModel):
    eval_id: str
    model: str
    query: str
    response: str
    scores: EvalScores
    latency_ms: float
    cost_usd: float
    tokens: TokenUsage
    timestamp: datetime


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: Optional[str] = None
    models: List[ModelName] = Field(
        default=[ModelName.GPT4O, ModelName.CLAUDE_SONNET, ModelName.MISTRAL_7B],
        description="Models to compare",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Explain quantum entanglement in simple terms.",
                "context": "Quantum entanglement is a phenomenon where two particles become correlated...",
                "models": ["gpt-4o", "claude-3-5-sonnet-20241022", "mistral-7b-instruct"],
            }
        }


class CompareResponse(BaseModel):
    comparison_id: str
    query: str
    results: List[EvaluateResponse]
    winner: Dict[str, str]  # {"category": "model_name"}
    summary: str


class HistoryItem(BaseModel):
    eval_id: str
    model: str
    query: str
    faithfulness: float
    relevance: float
    toxicity: float
    latency_ms: float
    cost_usd: float
    timestamp: datetime


class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    total: int
    page: int
    page_size: int


class AggregateStats(BaseModel):
    model: str
    avg_faithfulness: float
    avg_relevance: float
    avg_toxicity: float
    avg_latency_ms: float
    avg_cost_usd: float
    total_queries: int
    total_cost_usd: float

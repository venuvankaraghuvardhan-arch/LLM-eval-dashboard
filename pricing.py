"""
Token pricing per model (USD per 1,000 tokens).
Updated as of Q1 2025 — check provider pages for latest pricing.
"""

from typing import TypedDict


class PricingTier(TypedDict):
    input: float   # USD per 1K input tokens
    output: float  # USD per 1K output tokens


# Per 1,000 tokens
MODEL_PRICING: dict[str, PricingTier] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "mistral-7b-instruct": {"input": 0.00025, "output": 0.00025},
    "mistral-large-latest": {"input": 0.002, "output": 0.006},
    # Mock fallback
    "mock": {"input": 0.001, "output": 0.002},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate total cost in USD for a given model and token counts."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["mock"])
    input_cost = (prompt_tokens / 1000) * pricing["input"]
    output_cost = (completion_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 8)


def get_pricing(model: str) -> PricingTier:
    return MODEL_PRICING.get(model, MODEL_PRICING["mock"])

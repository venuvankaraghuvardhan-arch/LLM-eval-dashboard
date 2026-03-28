"""
Faithfulness Evaluator

Measures whether the LLM's response is grounded in the provided context,
i.e., the inverse of hallucination rate.

Method: LLM-as-judge (uses a lightweight model to check each claim).
In mock/no-API mode: uses simple keyword overlap heuristic.
"""

import os
import re
import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"


async def score_faithfulness(
    response: str,
    context: Optional[str],
    model: str = "gpt-4o-mini",
) -> float:
    """
    Returns a faithfulness score between 0.0 and 1.0.
    - 1.0 = every claim in the response is supported by the context
    - 0.0 = response is entirely hallucinated / unsupported

    If no context is provided, returns 0.5 (cannot verify).
    """
    if not context:
        return 0.5

    if MOCK_MODE or not os.getenv("OPENAI_API_KEY"):
        return _heuristic_faithfulness(response, context)

    return await _llm_judge_faithfulness(response, context, model)


def _heuristic_faithfulness(response: str, context: str) -> float:
    """
    Simple keyword overlap heuristic as a faithfulness proxy.
    Not as accurate as LLM-as-judge, but works without API keys.
    """
    # Normalize
    resp_words = set(response.lower().split())
    ctx_words = set(context.lower().split())

    # Remove stopwords
    stopwords = {
        "the", "a", "an", "is", "in", "of", "to", "and", "or", "for",
        "that", "this", "it", "with", "as", "at", "be", "was", "are",
        "on", "by", "from", "not", "but",
    }
    resp_words -= stopwords
    ctx_words -= stopwords

    if not resp_words:
        return 0.5

    overlap = len(resp_words & ctx_words)
    score = min(overlap / max(len(resp_words) * 0.6, 1), 1.0)

    # Add slight noise to simulate model variance
    noise = random.uniform(-0.05, 0.05)
    return round(max(0.0, min(1.0, score + noise)), 3)


async def _llm_judge_faithfulness(response: str, context: str, judge_model: str) -> float:
    """
    Use an LLM to check whether each claim in the response is grounded in context.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""You are an expert evaluator. Your task is to check if the given response is faithful to the context.

CONTEXT:
{context}

RESPONSE:
{response}

Instructions:
1. Break the response into individual factual claims
2. For each claim, check if it is directly supported by the context
3. Count supported claims and total claims
4. Return ONLY a JSON object: {{"supported": X, "total": Y, "score": Z}}
where score = supported/total (0.0 to 1.0)"""

    try:
        result = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        content = result.choices[0].message.content
        # Parse score from JSON
        match = re.search(r'"score":\s*([\d.]+)', content)
        if match:
            return round(float(match.group(1)), 3)
    except Exception as e:
        logger.warning(f"LLM judge failed, falling back to heuristic: {e}")

    return _heuristic_faithfulness(response, context)

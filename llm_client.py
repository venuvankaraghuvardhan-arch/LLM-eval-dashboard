"""
Unified LLM client that wraps OpenAI, Anthropic, and Mistral APIs.
Falls back to mock mode when MOCK_MODE=true or API keys are missing.
"""

import asyncio
import os
import random
import time
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

# Mock responses for testing without API keys
MOCK_RESPONSES = [
    "Based on the provided context, the answer is straightforward. The capital of France is Paris, which has served as the country's political and cultural center for centuries.",
    "Quantum entanglement occurs when two particles become linked such that measuring one instantly affects the other, regardless of distance. Einstein called this 'spooky action at a distance'.",
    "The main difference between supervised and unsupervised learning lies in the presence of labeled training data. Supervised learning uses labeled examples to train a model, while unsupervised learning discovers patterns in unlabeled data.",
    "To implement a binary search tree, you need a Node class with left/right children and a value, plus insert/search methods that recursively traverse the tree based on comparisons.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns. While natural factors contribute, human activities—particularly burning fossil fuels—have been the dominant driver since the mid-20th century.",
]


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


async def generate_response(
    query: str,
    model: str,
    context: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> LLMResponse:
    """
    Generate a response from the specified LLM.
    Uses mock mode if MOCK_MODE=true or the appropriate API key is missing.
    """
    if MOCK_MODE or not _has_api_key(model):
        return await _mock_response(query, model)

    if model.startswith("gpt"):
        return await _openai_response(query, model, context, system_prompt)
    elif model.startswith("claude"):
        return await _anthropic_response(query, model, context, system_prompt)
    elif model.startswith("mistral"):
        return await _mistral_response(query, model, context, system_prompt)
    else:
        return await _mock_response(query, model)


def _has_api_key(model: str) -> bool:
    if model.startswith("gpt"):
        return bool(os.getenv("OPENAI_API_KEY"))
    elif model.startswith("claude"):
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif model.startswith("mistral"):
        return bool(os.getenv("MISTRAL_API_KEY"))
    return False


async def _mock_response(query: str, model: str) -> LLMResponse:
    """Simulate an LLM response with realistic latency and token counts."""
    # Simulate latency based on model size
    latency_map = {
        "gpt-4o": (600, 1800),
        "gpt-4o-mini": (200, 600),
        "claude-3-5-sonnet-20241022": (700, 2000),
        "claude-3-haiku-20240307": (150, 500),
        "mistral-7b-instruct": (300, 900),
        "mistral-large-latest": (500, 1500),
    }
    low, high = latency_map.get(model, (300, 1000))
    latency_ms = random.uniform(low, high)

    await asyncio.sleep(latency_ms / 1000)  # Simulate actual wait

    response = random.choice(MOCK_RESPONSES)
    prompt_tokens = len(query.split()) * 2 + random.randint(10, 50)
    completion_tokens = len(response.split()) * 2

    return LLMResponse(
        content=response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=round(latency_ms, 2),
        model=model,
    )


async def _openai_response(
    query: str, model: str, context: Optional[str], system_prompt: Optional[str]
) -> LLMResponse:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = []

    sys = system_prompt or "You are a helpful and accurate assistant."
    if context:
        sys += f"\n\nContext:\n{context}"
    messages.append({"role": "system", "content": sys})
    messages.append({"role": "user", "content": query})

    start = time.time()
    response = await client.chat.completions.create(model=model, messages=messages, max_tokens=512)
    latency_ms = (time.time() - start) * 1000

    return LLMResponse(
        content=response.choices[0].message.content,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        latency_ms=round(latency_ms, 2),
        model=model,
    )


async def _anthropic_response(
    query: str, model: str, context: Optional[str], system_prompt: Optional[str]
) -> LLMResponse:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    sys = system_prompt or "You are a helpful and accurate assistant."
    if context:
        sys += f"\n\nContext:\n{context}"

    start = time.time()
    response = await client.messages.create(
        model=model,
        max_tokens=512,
        system=sys,
        messages=[{"role": "user", "content": query}],
    )
    latency_ms = (time.time() - start) * 1000

    return LLMResponse(
        content=response.content[0].text,
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        latency_ms=round(latency_ms, 2),
        model=model,
    )


async def _mistral_response(
    query: str, model: str, context: Optional[str], system_prompt: Optional[str]
) -> LLMResponse:
    from mistralai.async_client import MistralAsyncClient

    client = MistralAsyncClient(api_key=os.getenv("MISTRAL_API_KEY"))
    messages = []
    sys = system_prompt or "You are a helpful and accurate assistant."
    if context:
        sys += f"\n\nContext:\n{context}"
    messages.append({"role": "system", "content": sys})
    messages.append({"role": "user", "content": query})

    start = time.time()
    response = await client.chat(model=model, messages=messages, max_tokens=512)
    latency_ms = (time.time() - start) * 1000

    return LLMResponse(
        content=response.choices[0].message.content,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        latency_ms=round(latency_ms, 2),
        model=model,
    )

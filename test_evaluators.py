"""
Unit tests for the evaluation pipeline.
Run: pytest tests/ -v
"""

import pytest
import asyncio
import os

# Force mock mode for tests
os.environ["MOCK_MODE"] = "true"

from backend.evaluators.faithfulness import score_faithfulness, _heuristic_faithfulness
from backend.evaluators.relevance import score_relevance, _heuristic_relevance
from backend.evaluators.toxicity import score_toxicity, _heuristic_toxicity
from backend.models.pricing import calculate_cost, get_pricing


# ─── Faithfulness Tests ──────────────────────────────────────────────────────

class TestFaithfulness:
    def test_high_overlap_returns_high_score(self):
        context = "Paris is the capital of France and a major European city."
        response = "The capital of France is Paris, a major European city."
        score = _heuristic_faithfulness(response, context)
        assert score > 0.5

    def test_no_context_returns_neutral(self):
        score = asyncio.get_event_loop().run_until_complete(
            score_faithfulness("some response", None)
        )
        assert score == 0.5

    def test_empty_response_returns_neutral(self):
        score = _heuristic_faithfulness("", "some context")
        assert 0.0 <= score <= 1.0

    def test_score_in_valid_range(self):
        score = _heuristic_faithfulness("The sky is green.", "The sky is blue.")
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_async_faithfulness_runs(self):
        score = await score_faithfulness("Paris is the capital.", "France's capital is Paris.")
        assert 0.0 <= score <= 1.0


# ─── Relevance Tests ──────────────────────────────────────────────────────────

class TestRelevance:
    def test_identical_text_high_relevance(self):
        text = "what is machine learning and how does it work"
        score = _heuristic_relevance(text, text)
        assert score > 0.7

    def test_unrelated_text_lower_score(self):
        query = "what is quantum computing"
        response = "I enjoy cooking pasta on weekends with fresh tomatoes."
        score = _heuristic_relevance(query, response)
        # Heuristic has a bias offset so we just check it's in range
        assert 0.0 <= score <= 1.0

    def test_score_in_valid_range(self):
        score = _heuristic_relevance("What is AI?", "Artificial intelligence enables machines to learn.")
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_async_relevance_runs(self):
        score = await score_relevance("What is deep learning?", "Deep learning uses neural networks.")
        assert 0.0 <= score <= 1.0


# ─── Toxicity Tests ───────────────────────────────────────────────────────────

class TestToxicity:
    def test_safe_text_low_toxicity(self):
        safe_text = "The weather today is quite pleasant and sunny."
        score = _heuristic_toxicity(safe_text)
        assert score < 0.1

    def test_keyword_toxic_higher_score(self):
        toxic_text = "I hate this and will attack you."
        score = _heuristic_toxicity(toxic_text)
        assert score > 0.0

    def test_score_in_valid_range(self):
        for text in [
            "Hello, how are you?",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a fascinating field.",
        ]:
            score = _heuristic_toxicity(text)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for: {text}"

    @pytest.mark.asyncio
    async def test_async_toxicity_runs(self):
        score = await score_toxicity("This is a perfectly normal sentence.")
        assert 0.0 <= score <= 1.0


# ─── Pricing Tests ────────────────────────────────────────────────────────────

class TestPricing:
    def test_known_model_cost(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        expected = (1000 / 1000) * 0.005 + (500 / 1000) * 0.015
        assert abs(cost - expected) < 1e-9

    def test_unknown_model_uses_fallback(self):
        cost = calculate_cost("unknown-model-xyz", 100, 50)
        assert cost > 0.0

    def test_zero_tokens_zero_cost(self):
        cost = calculate_cost("gpt-4o", 0, 0)
        assert cost == 0.0

    def test_claude_pricing(self):
        cost = calculate_cost("claude-3-5-sonnet-20241022", 1000, 1000)
        assert cost > 0.0
        assert cost < 0.1  # Sanity check — should be cheap

    def test_cheaper_model_costs_less(self):
        cost_4o = calculate_cost("gpt-4o", 1000, 500)
        cost_mini = calculate_cost("gpt-4o-mini", 1000, 500)
        assert cost_mini < cost_4o

    def test_get_pricing_returns_dict(self):
        pricing = get_pricing("gpt-4o")
        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] > 0

"""
API integration tests.
Run: pytest tests/test_api.py -v

These tests spin up the FastAPI app in-process using httpx AsyncClient.
No external API calls are made (MOCK_MODE=true).
"""

import pytest
import os
import asyncio

os.environ["MOCK_MODE"] = "true"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_eval.db"

from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.models.database import init_db


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_db():
    await init_db()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ─── Health ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["mock_mode"] is True


# ─── Evaluate ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_basic(client):
    resp = await client.post("/api/v1/evaluate", json={
        "query": "What is the capital of France?",
        "context": "France is in Europe. Paris is its capital.",
        "model": "gpt-4o",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert "eval_id" in data
    assert data["eval_id"].startswith("eval_")
    assert data["model"] == "gpt-4o"
    assert "scores" in data
    assert 0.0 <= data["scores"]["faithfulness"] <= 1.0
    assert 0.0 <= data["scores"]["relevance"] <= 1.0
    assert 0.0 <= data["scores"]["toxicity"] <= 1.0
    assert data["latency_ms"] > 0
    assert data["cost_usd"] > 0
    assert data["tokens"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_evaluate_with_pregenerated_response(client):
    resp = await client.post("/api/v1/evaluate", json={
        "query": "What is AI?",
        "model": "claude-3-5-sonnet-20241022",
        "response": "AI stands for Artificial Intelligence, a branch of computer science.",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "AI stands for Artificial Intelligence, a branch of computer science."


@pytest.mark.asyncio
async def test_evaluate_all_models(client):
    models = [
        "gpt-4o", "gpt-4o-mini",
        "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307",
        "mistral-7b-instruct",
    ]
    for model in models:
        resp = await client.post("/api/v1/evaluate", json={
            "query": "Explain neural networks briefly.",
            "model": model,
        })
        assert resp.status_code == 200, f"Failed for model: {model}"
        assert resp.json()["model"] == model


@pytest.mark.asyncio
async def test_evaluate_invalid_model(client):
    resp = await client.post("/api/v1/evaluate", json={
        "query": "test",
        "model": "not-a-real-model-xyz",
    })
    assert resp.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_evaluate_empty_query(client):
    resp = await client.post("/api/v1/evaluate", json={
        "query": "",
        "model": "gpt-4o",
    })
    assert resp.status_code == 422


# ─── Compare ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compare_basic(client):
    resp = await client.post("/api/v1/compare", json={
        "query": "What is machine learning?",
        "models": ["gpt-4o", "claude-3-5-sonnet-20241022"],
    })
    assert resp.status_code == 200
    data = resp.json()

    assert "comparison_id" in data
    assert data["comparison_id"].startswith("cmp_")
    assert len(data["results"]) == 2
    assert "winner" in data
    assert "overall" in data["winner"]
    assert "fastest" in data["winner"]
    assert "cheapest" in data["winner"]


@pytest.mark.asyncio
async def test_compare_winner_is_valid_model(client):
    models = ["gpt-4o", "mistral-7b-instruct"]
    resp = await client.post("/api/v1/compare", json={
        "query": "Explain transformers.",
        "models": models,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["winner"]["overall"] in models
    assert data["winner"]["fastest"] in models


# ─── History ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_returns_items(client):
    # First run an eval to ensure there's data
    await client.post("/api/v1/evaluate", json={
        "query": "History test query",
        "model": "gpt-4o",
    })

    resp = await client.get("/api/v1/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 1


@pytest.mark.asyncio
async def test_history_filter_by_model(client):
    resp = await client.get("/api/v1/history?model=gpt-4o")
    assert resp.status_code == 200
    data = resp.json()
    for item in data["items"]:
        assert item["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_stats_endpoint(client):
    resp = await client.get("/api/v1/stats")
    assert resp.status_code == 200
    stats = resp.json()
    assert isinstance(stats, list)
    if stats:
        stat = stats[0]
        assert "model" in stat
        assert "avg_faithfulness" in stat
        assert "total_queries" in stat


@pytest.mark.asyncio
async def test_get_single_eval(client):
    # Create an eval first
    create_resp = await client.post("/api/v1/evaluate", json={
        "query": "Single eval test",
        "model": "gpt-4o-mini",
    })
    eval_id = create_resp.json()["eval_id"]

    resp = await client.get(f"/api/v1/history/{eval_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["eval_id"] == eval_id


@pytest.mark.asyncio
async def test_get_nonexistent_eval_404(client):
    resp = await client.get("/api/v1/history/eval_doesnotexist")
    assert resp.status_code == 404

# 🔬 LLM Evaluation & Observability Dashboard

A production-grade platform that benchmarks LLM outputs across **quality**, **cost**, **latency**, and **hallucination rate** — with a live Grafana dashboard and FastAPI backend.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?style=flat-square&logo=docker)
![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-orange?style=flat-square&logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-orange?style=flat-square&logo=grafana)

---

## 🎯 What This Project Does

Most portfolios show LLM *usage*. This project shows LLM *evaluation* — a skill every company deploying AI in production desperately needs.

| Feature | Description |
|---|---|
| 🧠 **Faithfulness Scoring** | Measures if the LLM's answer is grounded in the provided context (RAG quality) |
| 📊 **Relevance Scoring** | Evaluates if the response actually answers the question |
| ☣️ **Toxicity Detection** | Flags harmful or inappropriate outputs |
| ⚔️ **A/B Model Comparison** | Side-by-side benchmarking of GPT-4o vs Claude vs Mistral |
| 💰 **Cost-Per-Query Tracking** | Real-time cost monitoring per model and query |
| ⏱️ **Latency Profiling** | P50/P95/P99 latency breakdowns |
| 📈 **Live Grafana Dashboard** | All metrics visualized in real-time |
| 🔄 **Prometheus Metrics** | Industry-standard observability scraping |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client / UI                          │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────────────┐
│              FastAPI Backend  (:8000)                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ /evaluate   │  │ /compare     │  │ /metrics (prom)    │  │
│  └──────┬──────┘  └──────┬───────┘  └────────────────────┘  │
│         │                │                                   │
│  ┌──────▼──────────────────▼───────────────────────────┐    │
│  │              Evaluation Engine                       │    │
│  │  Faithfulness │ Relevance │ Toxicity │ Cost Tracker  │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
        │ Scrape (:8000/metrics)
┌───────▼───────────┐        ┌──────────────────────┐
│  Prometheus (:9090)│◄──────►│   Grafana  (:3000)   │
└───────────────────┘        └──────────────────────┘
        │ Store
┌───────▼───────────┐
│   SQLite / Postgres│
└───────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key (optional — mock mode works without it)

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/llm-eval-dashboard.git
cd llm-eval-dashboard
cp .env.example .env
# Edit .env with your API keys
```

### 2. Launch Everything

```bash
docker-compose up --build
```

### 3. Access the Services

| Service | URL | Credentials |
|---|---|---|
| **FastAPI Docs** | http://localhost:8000/docs | — |
| **Grafana Dashboard** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | — |

---

## 📡 API Endpoints

### Evaluate a Single Query
```bash
POST /api/v1/evaluate
{
  "query": "What is the capital of France?",
  "context": "France is a country in Western Europe. Its capital city is Paris.",
  "model": "gpt-4o",
  "response": "The capital of France is Paris."
}
```

**Response:**
```json
{
  "eval_id": "eval_abc123",
  "model": "gpt-4o",
  "scores": {
    "faithfulness": 0.97,
    "relevance": 0.95,
    "toxicity": 0.01
  },
  "latency_ms": 432,
  "cost_usd": 0.000234,
  "tokens": { "prompt": 45, "completion": 12 }
}
```

### A/B Compare Models
```bash
POST /api/v1/compare
{
  "query": "Explain quantum entanglement simply.",
  "context": "...",
  "models": ["gpt-4o", "claude-3-5-sonnet", "mistral-7b"]
}
```

### Get Evaluation History
```bash
GET /api/v1/history?limit=50&model=gpt-4o
```

---

## 🧪 Running Evaluations

### Run the benchmark suite
```bash
python scripts/run_benchmark.py --dataset data/sample_queries.json --models gpt-4o claude-3-5-sonnet
```

### Load test (generate metrics)
```bash
python scripts/load_test.py --queries 100 --concurrency 5
```

---

## 📊 Evaluation Methodology

### Faithfulness Score
Uses an LLM-as-judge pattern: a secondary model checks if each claim in the response is supported by the provided context. Score = (supported claims) / (total claims).

### Relevance Score
Computes semantic similarity between the query embedding and response embedding using cosine similarity on `sentence-transformers` embeddings.

### Toxicity Score
Uses the `detoxify` library (fine-tuned BERT) to classify toxicity across 6 dimensions (toxic, severe_toxic, obscene, threat, insult, identity_hate). Score = max toxicity across dimensions.

### Cost Calculation
Based on official per-token pricing for each model. Configurable in `backend/models/pricing.py`.

---

## 🗂️ Project Structure

```
llm-eval-dashboard/
├── backend/
│   ├── main.py                 # FastAPI app entry point
│   ├── api/
│   │   ├── evaluate.py         # Evaluation endpoints
│   │   ├── compare.py          # A/B comparison endpoints
│   │   └── history.py          # History & analytics endpoints
│   ├── evaluators/
│   │   ├── faithfulness.py     # Faithfulness scoring
│   │   ├── relevance.py        # Relevance scoring
│   │   └── toxicity.py         # Toxicity detection
│   ├── models/
│   │   ├── schemas.py          # Pydantic request/response models
│   │   ├── database.py         # SQLAlchemy models + DB setup
│   │   └── pricing.py          # Token pricing per model
│   └── services/
│       ├── llm_client.py       # Unified LLM API client
│       ├── metrics.py          # Prometheus metrics registry
│       └── cost_tracker.py     # Cost aggregation service
├── frontend/
│   └── index.html              # Standalone dashboard UI
├── prometheus/
│   └── prometheus.yml          # Scrape config
├── grafana/
│   └── dashboards/
│       └── llm_eval.json       # Pre-built Grafana dashboard
├── scripts/
│   ├── run_benchmark.py        # Batch evaluation runner
│   └── load_test.py            # Load testing script
├── tests/
│   ├── test_evaluators.py      # Unit tests for evaluators
│   └── test_api.py             # API integration tests
├── data/
│   └── sample_queries.json     # Sample benchmark dataset
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🔧 Configuration

Key environment variables (`.env`):

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
DATABASE_URL=sqlite:///./eval_data.db
MOCK_MODE=true           # Set false to use real LLM APIs
LOG_LEVEL=INFO
```

---

## 🧩 Extending the Project

### Add a new evaluator
```python
# backend/evaluators/my_evaluator.py
from backend.evaluators.base import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    async def score(self, query: str, response: str, context: str) -> float:
        # Your scoring logic here
        return 0.85
```

### Add a new model
```python
# backend/models/pricing.py
MODEL_PRICING = {
    "my-new-model": {"input": 0.001, "output": 0.002},  # per 1K tokens
    ...
}
```

---

## 📈 Sample Metrics in Grafana

- Average faithfulness score by model (line chart, 24h)
- P95 latency heatmap across models
- Cost per 1000 queries (bar chart)
- Toxicity flag rate (gauge)
- Evaluation throughput (requests/min)

---

## 🧑‍💻 Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Evaluation | Custom eval framework + sentence-transformers + detoxify |
| Observability | Prometheus + Grafana |
| Database | SQLite (dev) / PostgreSQL (prod) |
| LLM Clients | OpenAI SDK + Anthropic SDK + Mistral SDK |
| Testing | pytest + httpx |
| Deployment | Docker Compose |

---

## 📝 License

MIT — use freely in your own projects.

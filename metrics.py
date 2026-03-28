"""
Prometheus metrics registry for LLM observability.
All metrics are registered here and updated by the evaluation pipeline.
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    Summary,
)

# Use a custom registry to avoid conflicts with default registry
metrics_registry = CollectorRegistry()

# ── Counters ────────────────────────────────────────────────────────────────

eval_total = Counter(
    "llm_eval_total",
    "Total number of evaluations run",
    ["model"],
    registry=metrics_registry,
)

eval_errors_total = Counter(
    "llm_eval_errors_total",
    "Total number of evaluation errors",
    ["model", "error_type"],
    registry=metrics_registry,
)

# ── Histograms ───────────────────────────────────────────────────────────────

latency_histogram = Histogram(
    "llm_latency_ms",
    "LLM response latency in milliseconds",
    ["model"],
    buckets=[50, 100, 250, 500, 1000, 2000, 5000, 10000],
    registry=metrics_registry,
)

faithfulness_histogram = Histogram(
    "llm_faithfulness_score",
    "Faithfulness scores (0=hallucinated, 1=grounded)",
    ["model"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=metrics_registry,
)

relevance_histogram = Histogram(
    "llm_relevance_score",
    "Relevance scores (0=irrelevant, 1=highly relevant)",
    ["model"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=metrics_registry,
)

toxicity_histogram = Histogram(
    "llm_toxicity_score",
    "Toxicity scores (0=safe, 1=toxic)",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    registry=metrics_registry,
)

cost_histogram = Histogram(
    "llm_cost_usd",
    "Cost per query in USD",
    ["model"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    registry=metrics_registry,
)

# ── Gauges ───────────────────────────────────────────────────────────────────

avg_faithfulness_gauge = Gauge(
    "llm_avg_faithfulness",
    "Rolling average faithfulness score",
    ["model"],
    registry=metrics_registry,
)

avg_relevance_gauge = Gauge(
    "llm_avg_relevance",
    "Rolling average relevance score",
    ["model"],
    registry=metrics_registry,
)

avg_toxicity_gauge = Gauge(
    "llm_avg_toxicity",
    "Rolling average toxicity score",
    ["model"],
    registry=metrics_registry,
)

total_cost_gauge = Gauge(
    "llm_total_cost_usd",
    "Total accumulated cost in USD",
    ["model"],
    registry=metrics_registry,
)

active_evaluations_gauge = Gauge(
    "llm_active_evaluations",
    "Number of evaluations currently in progress",
    registry=metrics_registry,
)


def record_evaluation(model: str, latency_ms: float, scores: dict, cost_usd: float):
    """Update all Prometheus metrics after an evaluation completes."""
    eval_total.labels(model=model).inc()
    latency_histogram.labels(model=model).observe(latency_ms)
    faithfulness_histogram.labels(model=model).observe(scores["faithfulness"])
    relevance_histogram.labels(model=model).observe(scores["relevance"])
    toxicity_histogram.labels(model=model).observe(scores["toxicity"])
    cost_histogram.labels(model=model).observe(cost_usd)
    total_cost_gauge.labels(model=model).inc(cost_usd)

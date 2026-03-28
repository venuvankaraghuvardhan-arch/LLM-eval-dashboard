"""
LLM Evaluation & Observability Dashboard
FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_client import make_asgi_app
import logging
import os

from backend.models.database import init_db
from backend.api.evaluate import router as eval_router
from backend.api.compare import router as compare_router
from backend.api.history import router as history_router
from backend.services.metrics import metrics_registry

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("🚀 Starting LLM Eval Dashboard API...")
    await init_db()
    logger.info("✅ Database initialized")
    yield
    logger.info("🛑 Shutting down...")


app = FastAPI(
    title="LLM Evaluation & Observability Dashboard",
    description=(
        "Benchmark LLM outputs across quality, cost, latency, and hallucination rate. "
        "Supports GPT-4o, Claude 3.5 Sonnet, Mistral, and more."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app(registry=metrics_registry)
app.mount("/metrics", metrics_app)

# API Routers
app.include_router(eval_router, prefix="/api/v1", tags=["Evaluation"])
app.include_router(compare_router, prefix="/api/v1", tags=["Comparison"])
app.include_router(history_router, prefix="/api/v1", tags=["History"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "mock_mode": os.getenv("MOCK_MODE", "true").lower() == "true",
    }


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "LLM Eval Dashboard API — visit /docs for Swagger UI"}

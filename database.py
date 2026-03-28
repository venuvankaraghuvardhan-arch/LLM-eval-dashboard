"""
SQLAlchemy database models and async DB setup.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import String, Float, Integer, DateTime, Text
from datetime import datetime, timezone
import os


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./eval_data.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class EvaluationRecord(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    comparison_id: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    model: Mapped[str] = mapped_column(String(64), index=True)
    query: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    context: Mapped[str] = mapped_column(Text, nullable=True)

    # Scores
    faithfulness: Mapped[float] = mapped_column(Float, default=0.0)
    relevance: Mapped[float] = mapped_column(Float, default=0.0)
    toxicity: Mapped[float] = mapped_column(Float, default=0.0)

    # Performance
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

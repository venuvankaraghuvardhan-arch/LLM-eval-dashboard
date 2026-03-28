#!/usr/bin/env python3
"""
Load tester — generates evaluation traffic to populate Prometheus metrics and Grafana dashboards.

Usage:
  python scripts/load_test.py
  python scripts/load_test.py --queries 200 --concurrency 10
"""

import asyncio
import random
import argparse
import httpx
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from collections import defaultdict

console = Console()

BASE_URL = "http://localhost:8000/api/v1"

MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307",
    "mistral-7b-instruct",
]

SAMPLE_QUERIES = [
    {"query": "What is machine learning?", "context": "Machine learning is a subset of AI that enables systems to learn from data."},
    {"query": "Explain the attention mechanism.", "context": "The attention mechanism allows models to focus on relevant parts of the input when generating output."},
    {"query": "What is a vector database?", "context": "Vector databases store high-dimensional embeddings and enable similarity search."},
    {"query": "How does backpropagation work?", "context": "Backpropagation computes gradients by applying the chain rule from output to input layers."},
    {"query": "What is fine-tuning an LLM?", "context": "Fine-tuning adjusts a pre-trained model on a specific dataset for a downstream task."},
    {"query": "Explain cosine similarity.", "context": "Cosine similarity measures the angle between two vectors; 1 means identical direction."},
    {"query": "What is prompt engineering?", "context": "Prompt engineering involves crafting input prompts to elicit desired LLM behavior."},
    {"query": "What are embeddings in NLP?", "context": "Embeddings represent words or sentences as dense numerical vectors in a continuous space."},
    {"query": "How does RLHF work?", "context": "RLHF (Reinforcement Learning from Human Feedback) aligns LLMs using human preference data."},
    {"query": "What is the transformer architecture?", "context": "The transformer uses multi-head self-attention and feed-forward layers for sequence modeling."},
    {"query": "Explain knowledge distillation.", "context": "Knowledge distillation trains a small 'student' model to mimic a larger 'teacher' model."},
    {"query": "What is zero-shot learning?", "context": "Zero-shot learning allows models to generalize to unseen tasks without task-specific training data."},
]


async def send_eval(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, stats: dict) -> None:
    query = random.choice(SAMPLE_QUERIES)
    model = random.choice(MODELS)

    async with semaphore:
        try:
            resp = await client.post(
                f"{BASE_URL}/evaluate",
                json={"query": query["query"], "context": query["context"], "model": model},
                timeout=30.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                stats["success"] += 1
                stats["by_model"][model]["count"] += 1
                stats["by_model"][model]["cost"] += data["cost_usd"]
                stats["by_model"][model]["latency"].append(data["latency_ms"])
            else:
                stats["errors"] += 1
        except Exception:
            stats["errors"] += 1


async def run_load_test(total_queries: int, concurrency: int):
    console.print(f"\n[bold cyan]⚡ LLM Eval Load Tester[/bold cyan]")
    console.print(f"  Queries     : [yellow]{total_queries}[/yellow]")
    console.print(f"  Concurrency : [yellow]{concurrency}[/yellow]")
    console.print(f"  Models      : [yellow]{len(MODELS)}[/yellow]\n")

    stats = {
        "success": 0,
        "errors": 0,
        "by_model": {m: {"count": 0, "cost": 0.0, "latency": []} for m in MODELS},
    }

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        # Health check
        try:
            await client.get("http://localhost:8000/health", timeout=5.0)
        except Exception:
            console.print("[red]❌ API not reachable. Start the server first.[/red]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Sending evaluations...", total=total_queries)

            # Fire off batches
            batch_size = concurrency * 2
            for i in range(0, total_queries, batch_size):
                batch = min(batch_size, total_queries - i)
                tasks = [send_eval(client, semaphore, stats) for _ in range(batch)]
                await asyncio.gather(*tasks)
                progress.advance(task, batch)

    # Print summary
    _print_summary(stats)


def _print_summary(stats: dict):
    table = Table(title="\n📊 Load Test Summary", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Requests", justify="center")
    table.add_column("Avg Latency (ms)", justify="center")
    table.add_column("Total Cost (USD)", justify="center")

    for model, data in stats["by_model"].items():
        if data["count"] == 0:
            continue
        avg_lat = sum(data["latency"]) / len(data["latency"]) if data["latency"] else 0
        table.add_row(
            model,
            str(data["count"]),
            f"{avg_lat:.0f}",
            f"${data['cost']:.6f}",
        )

    console.print(table)
    console.print(f"\n✅ Success: [green]{stats['success']}[/green]  ❌ Errors: [red]{stats['errors']}[/red]")
    console.print("\n[dim]→ Check Grafana at http://localhost:3000 to see metrics populate[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Load test the LLM eval API")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(run_load_test(args.queries, args.concurrency))


if __name__ == "__main__":
    main()

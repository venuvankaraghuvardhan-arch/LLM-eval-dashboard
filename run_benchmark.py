#!/usr/bin/env python3
"""
Batch benchmark runner — evaluates a dataset of queries across one or more models.

Usage:
  python scripts/run_benchmark.py
  python scripts/run_benchmark.py --dataset data/sample_queries.json --models gpt-4o claude-3-5-sonnet-20241022
  python scripts/run_benchmark.py --output results/benchmark_20250101.json
"""

import asyncio
import json
import time
import argparse
import httpx
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

BASE_URL = "http://localhost:8000/api/v1"

DEFAULT_MODELS = [
    "gpt-4o",
    "claude-3-5-sonnet-20241022",
    "mistral-7b-instruct",
]


async def run_single_eval(client: httpx.AsyncClient, query: dict, model: str) -> dict:
    """Run a single evaluation via the API."""
    payload = {
        "query": query["query"],
        "model": model,
    }
    if "context" in query:
        payload["context"] = query["context"]

    try:
        resp = await client.post(f"{BASE_URL}/evaluate", json=payload, timeout=60.0)
        resp.raise_for_status()
        return {"status": "ok", "model": model, **resp.json()}
    except Exception as e:
        return {"status": "error", "model": model, "error": str(e), "query": query["query"]}


async def run_benchmark(dataset_path: str, models: list[str], output_path: str | None = None):
    """Run the full benchmark suite."""
    # Load dataset
    data_file = Path(dataset_path)
    if not data_file.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        console.print("Using built-in sample queries instead...")
        queries = _get_sample_queries()
    else:
        with open(data_file) as f:
            queries = json.load(f)

    console.print(f"\n[bold cyan]🔬 LLM Eval Benchmark[/bold cyan]")
    console.print(f"  Queries : [yellow]{len(queries)}[/yellow]")
    console.print(f"  Models  : [yellow]{', '.join(models)}[/yellow]")
    console.print(f"  Total   : [yellow]{len(queries) * len(models)} evaluations[/yellow]\n")

    results = []
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        # Check API health
        try:
            health = await client.get("http://localhost:8000/health", timeout=5.0)
            health.raise_for_status()
        except Exception:
            console.print("[red]❌ API not reachable at localhost:8000. Is it running?[/red]")
            console.print("   Run: [bold]uvicorn backend.main:app --reload[/bold]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            total = len(queries) * len(models)
            task = progress.add_task("Running evaluations...", total=total)

            for query in queries:
                for model in models:
                    progress.update(task, description=f"[cyan]{model[:20]}[/cyan] — {query['query'][:40]}...")
                    result = await run_single_eval(client, query, model)
                    results.append(result)
                    progress.advance(task)

    elapsed = time.time() - start_time

    # Print results table
    _print_results_table(results, models)

    # Print aggregate stats
    _print_aggregate_stats(results, models, elapsed)

    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "models": models,
                "total_queries": len(queries),
                "elapsed_seconds": round(elapsed, 2),
                "results": results,
            }, f, indent=2, default=str)
        console.print(f"\n[green]✅ Results saved to {output_path}[/green]")


def _print_results_table(results: list[dict], models: list[str]):
    table = Table(title="\n📊 Benchmark Results", show_lines=True)
    table.add_column("Query", style="dim", max_width=35)
    for model in models:
        short = model.split("-")[0].upper()
        table.add_column(f"{short}\nFaith/Rel/Tox", justify="center")

    # Group by query
    from collections import defaultdict
    by_query = defaultdict(dict)
    for r in results:
        if r.get("status") == "ok":
            key = r["query"][:35]
            by_query[key][r["model"]] = r

    for query_key, model_results in list(by_query.items())[:20]:  # Show first 20
        row = [query_key]
        for model in models:
            if model in model_results:
                s = model_results[model]["scores"]
                faith_color = "green" if s["faithfulness"] > 0.8 else "yellow" if s["faithfulness"] > 0.6 else "red"
                row.append(
                    f"[{faith_color}]{s['faithfulness']:.2f}[/] / "
                    f"{s['relevance']:.2f} / "
                    f"[{'red' if s['toxicity'] > 0.1 else 'green'}]{s['toxicity']:.3f}[/]"
                )
            else:
                row.append("[red]ERROR[/red]")
        table.add_row(*row)

    console.print(table)


def _print_aggregate_stats(results: list[dict], models: list[str], elapsed: float):
    from collections import defaultdict

    stats = defaultdict(lambda: {"faith": [], "rel": [], "tox": [], "latency": [], "cost": []})

    for r in results:
        if r.get("status") == "ok":
            m = r["model"]
            stats[m]["faith"].append(r["scores"]["faithfulness"])
            stats[m]["rel"].append(r["scores"]["relevance"])
            stats[m]["tox"].append(r["scores"]["toxicity"])
            stats[m]["latency"].append(r["latency_ms"])
            stats[m]["cost"].append(r["cost_usd"])

    table = Table(title="📈 Aggregate Stats by Model", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Avg Faithfulness", justify="center")
    table.add_column("Avg Relevance", justify="center")
    table.add_column("Avg Toxicity", justify="center")
    table.add_column("Avg Latency (ms)", justify="center")
    table.add_column("Total Cost (USD)", justify="center")

    for model in models:
        s = stats[model]
        if not s["faith"]:
            continue

        def avg(lst): return sum(lst) / len(lst) if lst else 0

        table.add_row(
            model,
            f"{avg(s['faith']):.3f}",
            f"{avg(s['rel']):.3f}",
            f"{avg(s['tox']):.4f}",
            f"{avg(s['latency']):.0f}",
            f"${sum(s['cost']):.6f}",
        )

    console.print(table)
    console.print(f"\n⏱  Total elapsed: [cyan]{elapsed:.1f}s[/cyan]")


def _get_sample_queries() -> list[dict]:
    return [
        {"query": "What is the capital of France?", "context": "France is a country in Western Europe. Paris is its capital and largest city."},
        {"query": "Explain how transformers work in NLP.", "context": "Transformers use self-attention mechanisms to process sequences in parallel rather than sequentially."},
        {"query": "What is RAG in AI?", "context": "Retrieval-Augmented Generation (RAG) combines retrieval of relevant documents with language model generation."},
        {"query": "How does gradient descent work?", "context": "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function."},
        {"query": "What is the difference between precision and recall?", "context": "Precision is TP/(TP+FP). Recall is TP/(TP+FN). They trade off against each other."},
    ]


def main():
    parser = argparse.ArgumentParser(description="Run LLM eval benchmark")
    parser.add_argument("--dataset", default="data/sample_queries.json")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.dataset, args.models, args.output))


if __name__ == "__main__":
    main()

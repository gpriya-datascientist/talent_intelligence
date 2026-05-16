#!/usr/bin/env python3
"""
scripts/run_pipeline.py
-----------------------
One-shot CLI to ingest data, embed to ChromaDB, and generate narratives.

Usage
-----
python scripts/run_pipeline.py --source ./data/raw --kpi "Monthly Revenue" --question "What drove Q1 growth?"
python scripts/run_pipeline.py --source ./data/raw --all-kpis
python scripts/run_pipeline.py --retrain-now
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


def cmd_ingest(source: str, freq: str = "W") -> None:
    """Ingest and preprocess data from a directory or single file."""
    from src.ingestion.loader import load_file, load_directory
    from src.ingestion.preprocessor import preprocess
    from src.vectorstore.chroma_store import KPIVectorStore

    p = Path(source)
    with console.status("Loading data…"):
        df = load_file(p) if p.is_file() else load_directory(p)
        df_proc = preprocess(df, freq=freq)

    console.print(f"[green]✓ Loaded {len(df_proc):,} rows | {df_proc['kpi_name'].nunique()} KPIs[/green]")

    with console.status("Embedding to ChromaDB…"):
        store = KPIVectorStore()
        store.ingest(df_proc)

    stats = store.stats()
    console.print(f"[green]✓ ChromaDB: {stats['total_documents']:,} documents[/green]")
    return df_proc, store


def cmd_narrative(kpi: str, dimension: str, question: str, n_context: int = 8) -> None:
    """Generate a GPT-4 narrative for a KPI."""
    from src.vectorstore.chroma_store import KPIVectorStore
    from src.llm.narrative import NarrativeGenerator

    store = KPIVectorStore()
    if store.stats()["total_documents"] == 0:
        console.print("[red]✗ ChromaDB is empty. Run --ingest first.[/red]")
        return

    gen = NarrativeGenerator(store)
    with console.status(f"Generating narrative for {kpi} / {dimension}…"):
        result = gen.generate(question=question, kpi_name=kpi, dimension=dimension, n_context=n_context)

    rprint(Panel(result.narrative, title=f"[bold]{kpi} — {dimension}[/bold]", expand=False))

    if result.key_insights:
        table = Table(title="Key Insights", show_header=False, box=None)
        table.add_column("", style="dim")
        table.add_column("")
        for i, insight in enumerate(result.key_insights, 1):
            table.add_row(f"{i}.", insight)
        console.print(table)

    console.print(f"\n[bold]Trend Signal:[/bold] {result.trend_signal}  |  [bold]Confidence:[/bold] {result.confidence}  |  Tokens: {result.tokens_used}")


def cmd_forecast(kpi: str, dimension: str, horizon: int = 12) -> None:
    """Run forecast and print metrics."""
    import os
    import pandas as pd
    from src.ingestion.loader import load_directory
    from src.ingestion.preprocessor import preprocess
    from src.llm.forecaster import forecast_kpi

    data_dir = Path(os.getenv("DATA_RAW_DIR", "./data/raw"))
    if not data_dir.exists() or not any(data_dir.iterdir()):
        console.print(f"[red]✗ No data files found in {data_dir}[/red]")
        return

    with console.status("Loading and forecasting…"):
        df = load_directory(data_dir)
        df_proc = preprocess(df)
        result = forecast_kpi(df_proc, kpi, dimension, horizon=horizon)

    table = Table(title=f"Forecast: {kpi} / {dimension}")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Model",      result.model_name)
    table.add_row("MAPE",       f"{result.metrics['mape']:.2f}%")
    table.add_row("RMSE",       f"{result.metrics['rmse']:.2f}")
    table.add_row("MAE",        f"{result.metrics['mae']:.2f}")
    table.add_row("Horizon",    str(result.horizon))
    console.print(table)

    console.print("\n[bold]Forecast (next periods):[/bold]")
    console.print(result.forecast[["date", "predicted", "lower_ci", "upper_ci"]].to_string(index=False))


def cmd_retrain() -> None:
    """Trigger an immediate retrain (same logic as the weekly scheduler)."""
    from src.scheduler.scheduler import retrain_job
    with console.status("Running retrain job…"):
        report = retrain_job()
    if report["status"] == "SUCCESS":
        console.print(f"[green]✓ Retrain complete: {report.get('chroma_docs_after', '?')} docs[/green]")
    else:
        console.print(f"[red]✗ Retrain failed: {report.get('error', '?')}[/red]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KPI Trend Analysis — Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest and embed data")
    p_ingest.add_argument("--source", required=True, help="Path to file or directory")
    p_ingest.add_argument("--freq",   default="W",  help="Aggregation frequency (D/W/ME)")

    # narrative
    p_narr = sub.add_parser("narrative", help="Generate GPT-4 narrative")
    p_narr.add_argument("--kpi",       required=True)
    p_narr.add_argument("--dimension", default="Overall")
    p_narr.add_argument("--question",  required=True)
    p_narr.add_argument("--context",   type=int, default=8)

    # forecast
    p_fc = sub.add_parser("forecast", help="Run Predicted vs Actual forecast")
    p_fc.add_argument("--kpi",       required=True)
    p_fc.add_argument("--dimension", default="Overall")
    p_fc.add_argument("--horizon",   type=int, default=12)

    # retrain
    sub.add_parser("retrain", help="Trigger immediate retraining")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args.source, args.freq)
    elif args.command == "narrative":
        cmd_narrative(args.kpi, args.dimension, args.question, args.context)
    elif args.command == "forecast":
        cmd_forecast(args.kpi, args.dimension, args.horizon)
    elif args.command == "retrain":
        cmd_retrain()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

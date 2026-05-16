# Agentic Retraining Loop

Autonomous ML retraining agent built with **LangGraph** that monitors drift, retrains, evaluates, and deploys models with human-in-the-loop approval.

## Agent Graph

```
Monitor → Drift? → Collect Data → Retrain → Evaluate → [Human Approval] → Deploy
```

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Generate sample data + train baseline model
make setup

# 3. Run demo (no DB / API needed)
make demo

# 4. Start full stack (Postgres + MLflow + API)
cp .env.example .env   # edit values
make docker-up
```

## Project Structure

```
agents/          One file per LangGraph node
graph/           StateGraph wiring + shared state TypedDict
api/             FastAPI — /trigger /approve /status
db/              SQLAlchemy models + Postgres session
config/          Pydantic settings (reads .env)
data/sample/     Synthetic demo parquet files
scripts/         Data generation + demo runner
tests/           Pytest suite
```

## Demo Scenarios

| Scenario | File | Expected outcome |
|----------|------|-----------------|
| No drift | `production_nodrift.parquet` | Graph exits after Monitor |
| Drift detected | `production_drift.parquet` | Full pipeline → deploy |

## Stack

LangGraph · LangChain · MLflow · FastAPI · PostgreSQL · SQLAlchemy · scikit-learn

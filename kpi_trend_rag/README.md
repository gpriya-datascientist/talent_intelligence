# 📈 KPI Trend Analysis with RAG

> Ingest CSV / Power BI exports · ChromaDB vector store · GPT-4 narratives · Streamlit dashboard · Auto weekly retraining

---

## Architecture

```
data/raw/             ← Drop CSVs or Power BI exports here
    │
    ▼
src/ingestion/
  ├── loader.py        Auto-detect CSV / XLSX / wide / long formats
  └── preprocessor.py  Clean → resample → delta / rolling stats → anomaly flags → context strings
    │
    ▼
src/vectorstore/
  └── chroma_store.py  Sentence-transformer embeddings → ChromaDB (persistent)
    │
    ├────────────────────────────────────────┐
    ▼                                        ▼
src/llm/                              src/llm/
  narrative.py                          forecaster.py
  GPT-4 RAG narrative                   LinearTrend / ETS / SARIMA
  per KPI dimension                     Predicted vs Actual
    │                                        │
    └──────────────┬─────────────────────────┘
                   ▼
            app.py  (Streamlit dashboard)
                   │
            src/scheduler/
              scheduler.py   APScheduler — weekly retrain every Monday 02:00 UTC
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo> kpi_trend_rag
cd kpi_trend_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).  
Click **⚡ Load sample data** to see everything working without an upload.

---

## Data Formats Supported

| Format | Description |
|--------|-------------|
| Long CSV | `date, kpi_name, dimension, value, unit` columns (any order, any casing) |
| Wide CSV | `date` column + one column per KPI — auto-pivoted to long |
| Power BI export | `.xlsx` flat table export (single sheet) |
| Semicolon / tab delimited | Auto-detected |

Drop files into `data/raw/` or upload directly in the Streamlit sidebar.

---

## CLI Usage

```bash
# Ingest a directory of CSVs
python scripts/run_pipeline.py ingest --source ./data/raw --freq W

# Generate a GPT-4 narrative
python scripts/run_pipeline.py narrative \
  --kpi "Monthly Revenue" \
  --dimension "EMEA" \
  --question "What is driving the Q1 revenue decline?"

# Run a Predicted vs Actual forecast
python scripts/run_pipeline.py forecast \
  --kpi "Monthly Revenue" \
  --dimension "EMEA" \
  --horizon 12

# Trigger an immediate retrain
python scripts/run_pipeline.py retrain
```

---

## Weekly Retraining

The scheduler (`src/scheduler/scheduler.py`) runs via APScheduler:

- **When**: Every Monday at 02:00 UTC (configurable in `.env`)
- **What**: Scans `data/raw/`, re-runs full pipeline, resets + re-ingests ChromaDB
- **Output**: JSON run report in `logs/retrain_<run_id>.json`

**Run standalone (blocking):**
```bash
python -m src.scheduler.scheduler
```

**Embed in another process (background):**
```python
from src.scheduler.scheduler import get_background_scheduler
sched = get_background_scheduler()
sched.start()   # non-blocking
```

**Configure via `.env`:**
```
RETRAIN_SCHEDULE_DAY=monday
RETRAIN_SCHEDULE_HOUR=2
RETRAIN_SCHEDULE_MINUTE=0
```

---

## Forecasting Models

| Model | When selected | Typical use |
|-------|--------------|-------------|
| `LinearTrend` | `n < 12` observations | Sparse data, quick baseline |
| `ExponentialSmoothing` | `12 ≤ n < 24` | Monthly data with trend |
| `SARIMA` | `n ≥ 24` | Weekly / monthly with seasonality |

Auto-selection probes SARIMA first; falls back gracefully.  
Force a model with `model_preference="linear" / "ets" / "sarima"`.

---

## RAG Pipeline

1. **Embed** — each KPI row is turned into a rich natural-language sentence and embedded with `all-MiniLM-L6-v2` (runs locally, no API cost)
2. **Retrieve** — user question → cosine similarity search in ChromaDB → top-k context chunks
3. **Generate** — context injected into GPT-4 system prompt → structured narrative with trend signal + key insights

### Example narrative output

```
NARRATIVE:
Monthly Revenue for the EMEA region has shown a consistent upward trend
over the past 12 weeks, growing 14.2% period-over-period. The 12-period
rolling average confirms the trend is not driven by a single outlier...

KEY INSIGHTS:
1. Revenue grew 14.2% in the most recent period vs prior week.
2. No anomalies detected in the last 26 periods.
3. Year-over-year performance is up 22.1%, above the 15% annual target.
4. The rolling standard deviation has narrowed, suggesting reduced volatility.

TREND SIGNAL: UP
CONFIDENCE: HIGH
```

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests cover: loader (long/wide CSV, Excel, delimiter detection), preprocessor (anomaly flags, rolling stats, context strings), forecaster (shape, metrics, edge cases), and ChromaDB (ingest, query, reset).

---

## Project Structure

```
kpi_trend_rag/
├── app.py                        Streamlit dashboard
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                      ← place source files here
│   ├── processed/
│   └── exports/
├── models/
│   └── chromadb/                 ChromaDB persistence directory
├── logs/                         Retrain run reports (JSON)
├── scripts/
│   └── run_pipeline.py           CLI entrypoint
├── src/
│   ├── ingestion/
│   │   ├── loader.py             CSV / Excel ingest
│   │   └── preprocessor.py      Clean, enrich, anomaly detect
│   ├── vectorstore/
│   │   └── chroma_store.py       ChromaDB wrapper
│   ├── llm/
│   │   ├── narrative.py          GPT-4 RAG narrative generator
│   │   └── forecaster.py         LinearTrend / ETS / SARIMA
│   ├── dashboard/
│   └── scheduler/
│       └── scheduler.py          APScheduler weekly retrain
└── tests/
    └── test_pipeline.py
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for GPT-4 narrative generation |
| `OPENAI_MODEL` | `gpt-4o` | Model name |
| `CHROMA_PERSIST_DIR` | `./models/chromadb` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `kpi_insights` | Collection name |
| `SENTENCE_TRANSFORMER_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `DATA_RAW_DIR` | `./data/raw` | Source file directory |
| `RETRAIN_SCHEDULE_DAY` | `monday` | Day of week for retraining |
| `RETRAIN_SCHEDULE_HOUR` | `2` | Hour (UTC) for retraining |
| `NARRATIVE_MAX_TOKENS` | `1024` | Max tokens per GPT-4 call |
| `RAG_CONTEXT_CHUNKS` | `8` | Number of retrieved context chunks |

---

## License

MIT

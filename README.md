# Talent Intelligence Platform

AI-powered team assembly platform. A Product Owner describes a project in plain English — the system finds the right people from your employee pool using LLM chains, RAG, and semantic skill matching.

---

## Project Structure

```
talent-intelligence/
├── backend/          # FastAPI + LangChain + PostgreSQL
├── frontend/         # React + TypeScript + Vite
├── run_seed.py       # Populate DB with fake employees
├── create_db.py      # Create PostgreSQL database
└── requirements.txt  # Python dependencies
```

---

## Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- PostgreSQL 15
- OpenAI API key

---

## Backend Setup

```cmd
cd talent-intelligence

# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Open .env and fill in:
# OPENAI_API_KEY=sk-...
# DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/talent_db

# 4. Create database
python create_db.py

# 5. Seed with fake employees (24 employees across 8 personas)
python run_seed.py

# 6. Start backend
uvicorn backend.main:app --reload --port 8000
```

Backend runs at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

---

## Frontend Setup

```cmd
cd talent-intelligence/frontend

# 1. Install dependencies
npm install

# 2. Start frontend
npm run dev
```

Frontend runs at `http://localhost:3000`

---

## Pages

| URL | Role | Description |
|-----|------|-------------|
| `/login` | Both | Pick role (PO or Employee) and enter email |
| `/` | Product Owner | Type a project wish → get ranked team candidates |
| `/employee` | Employee | Set availability, upload resume, add GitHub |

---

## How it works

```
PO types wish
    ↓
LangChain (OpenAI) parses intent + detects domains
    ↓
Domain Expert Router selects SMEs to consult
    ↓
Requirement Builder creates structured skill requirements
    ↓
FAISS vector search retrieves matching employees
    ↓
Ranking engine scores by skill match + recency + GitHub + availability
    ↓
LLM generates "why this match" explanation per candidate
    ↓
PO sees ranked team with skill proof and backup candidates
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `OPENAI_API_KEY` | OpenAI API key (LLM + embeddings) |
| `OPENAI_MODEL` | Model name (default: gpt-4o-mini) |
| `EMBEDDING_MODEL` | Embedding model (default: text-embedding-3-small) |
| `FAISS_INDEX_PATH` | Path to store FAISS vector index |
| `GITHUB_TOKEN` | Optional — for real GitHub profile syncing |
| `LANGFUSE_PUBLIC_KEY` | Optional — for LLM observability |

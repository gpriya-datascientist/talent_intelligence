# Talent Intelligence — AI-Powered Team Assembly

> Part of the `talent_intelligence` monorepo.

A Product Owner describes a project in plain English. The system finds the right people from your employee pool using LangChain, RAG, and semantic skill matching.

---

## Quick Start

### Backend

```cmd
cd talent-intelligence

# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env — fill in OPENAI_API_KEY and DATABASE_URL with your postgres password

# 4. Create the database
python create_db.py

# 5. Seed with 24 fake employees (8 personas x 3 each)
python run_seed.py

# 6. Start backend
uvicorn backend.main:app --reload --port 8000
```

Backend → `http://localhost:8000`
API docs → `http://localhost:8000/docs`

---

### Frontend

```cmd
cd talent-intelligence/frontend

npm install
npm run dev
```

Frontend → `http://localhost:3000`

> Requires Node.js 18+ — download from https://nodejs.org

---

## Pages

| URL | Who | What |
|-----|-----|------|
| `/login` | Both | Pick role (PO or Employee), enter email |
| `/` | Product Owner | Type a project wish → get ranked team |
| `/employee` | Employee | Set availability, add GitHub, upload resume |

---

## Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | `postgresql+asyncpg://postgres:PASSWORD@localhost:5432/talent_db` |
| `OPENAI_API_KEY` | Yes | OpenAI key — used for LLM chains and embeddings |
| `OPENAI_MODEL` | No | Default: `gpt-4o-mini` |
| `EMBEDDING_MODEL` | No | Default: `text-embedding-3-small` |
| `GITHUB_TOKEN` | No | For real GitHub profile syncing |
| `LANGFUSE_PUBLIC_KEY` | No | LLM observability via Langfuse |

---

## Architecture

```
PO types wish
    ↓
Wish Parser (LangChain + OpenAI) → intent, domains, ambiguities
    ↓
Domain Expert Router → selects SMEs to consult first
    ↓
Requirement Builder → structured skill requirements
    ↓
FAISS vector search → top-k matching employees
    ↓
Ranking engine → skill match + recency + GitHub activity + availability
    ↓
Explanation chain → "why this match" per candidate
    ↓
PO sees ranked team with skill proof and backup candidates
```

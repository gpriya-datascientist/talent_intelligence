#  AI Talent Intelligence Platform — Wish-to-Team

![Alt text](ai_talent_intelligence_wish_to_team.svg)

An AI-powered platform that transforms a Product Owner's natural language "wish" into a fully assembled, role-matched team — automatically.

---

## Overview

The **Wish-to-Team** system bridges the gap between high-level product ideas and the right human talent. A Product Owner simply writes a wish or idea in natural language, and the platform intelligently decomposes it into tasks, identifies required skills, and matches the best candidates to form an optimal team.

---

## Architecture

The platform is built as a multi-layer AI pipeline:

### Layer 0 — Product Owner Input
- The Product Owner writes a **wish / idea in natural language**

### Layer 1 — Wish Parser *(LLM)* `NEW`
- **Intent extraction** and **domain classification**
- Understands what the PO wants and categorizes it

### Layer 2 — Domain Expert Consultant *(LLM)* `NEW`
- Consults relevant domain knowledge
- Refines and validates the parsed intent

### Layer 3 — Task Decomposer `NEW`
- Breaks the wish into concrete subtasks across three tracks:
  - **Frontend** — UI/UX tasks
  - **Backend** — API & logic tasks
  - **Data/ML** — Data engineering & model tasks

### Layer 4 — Skill Extractor *(LLM)*
- Extracts required skills per task:
  - **Frontend Skills** — e.g. React, TypeScript
  - **Backend Skills** — e.g. Python, FastAPI
  - **ML Skills** — e.g. NLP, Vector DBs

### Layer 5 — Candidate Ranker
- Ranks candidates based on skill match and availability

### Layer 6 — Role Fit Scorer *(LLM)*
- Scores each candidate's fit for specific roles using AI reasoning

### Layer 7 — Team Composer
- Assembles the final optimal team from ranked and scored candidates

### Layer 8 — Output
- **Team Proposal** — Recommended team with role assignments
- **Skill Gap Report** — Identifies missing skills or roles to hire for

---

##  Key Features

-  **Natural Language Input** — No forms or structured data needed from the PO
-  **LLM-Powered Reasoning** — Multiple AI agents handle parsing, scoring, and composition
-  **Multi-Domain Support** — Frontend, Backend, and Data/ML tracks
-  **Skill Gap Analysis** — Highlights gaps for hiring or upskilling
-  **End-to-End Automation** — From wish to team in one pipeline

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Agents | OpenAI GPT-4o-mini |
| Backend | Python 3.11, FastAPI, SQLAlchemy |
| Database | PostgreSQL 15 |
| Vector Store | FAISS |
| Embeddings | OpenAI text-embedding-3-small |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Orchestration | LangChain LCEL |
| Observability | Langfuse |
| Automation | n8n |

---

##  Project Structure

```
talent-intelligence/
├── backend/
│   ├── chains/           # LangChain LCEL chains (wish parser, skill extractor, etc.)
│   ├── models/           # SQLAlchemy ORM models (Employee, Skill, Availability, Wish)
│   ├── routers/          # FastAPI route handlers
│   ├── rag/              # Embeddings, FAISS vector store, retriever
│   ├── ranking/          # Skill scorer, availability scorer, ranker
│   ├── ingestion/        # Resume loader, GitHub loader, seed data
│   ├── evals/            # Eval datasets and eval runners
│   ├── config.py         # Settings via pydantic-settings
│   └── main.py           # FastAPI app entry point
├── frontend/
│   └── src/
│       └── pages/
│           ├── Login.tsx           # Role selection (PO / Employee)
│           ├── PODashboard.tsx     # Wish input + ranked team results
│           └── EmployeeDashboard.tsx # Availability, GitHub, resume, skills
├── run_seed.py           # Seed DB with 24 fake employees
├── create_db.py          # Create PostgreSQL database
├── docker-compose.yml    # PostgreSQL + n8n
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variable template
```

---

##  Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+ — [nodejs.org](https://nodejs.org)
- PostgreSQL 15
- OpenAI API key — [platform.openai.com](https://platform.openai.com)

### Backend Setup

```bash
cd talent-intelligence

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — fill in OPENAI_API_KEY and DATABASE_URL

# Create database
python create_db.py

# Seed with 24 fake employees (8 personas × 3 each)
python run_seed.py

# Start backend
uvicorn backend.main:app --reload --port 8000
```

Backend → `http://localhost:8000`
API docs → `http://localhost:8000/docs`

### Frontend Setup

```bash
cd talent-intelligence/frontend

npm install
npm run dev
```

Frontend → `http://localhost:3000`

### UI Pages

| URL | Role | Description |
|-----|------|-------------|
| `/login` | Both | Pick role (PO or Employee), enter email |
| `/` | Product Owner | Type a wish → get ranked team candidates |
| `/employee` | Employee | Set availability, add GitHub, upload resume |

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and fill in:



---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Gpriya** — [@gpriya-datascientist](https://github.com/gpriya-datascientist)

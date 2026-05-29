"""
demo_seed.py — loads 9 demo employees (strong/average/weak) into the DB.
GitHub stats hardcoded to match the fake repos in the demo resumes.
Run: python demo_seed.py
"""
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone, timedelta

os.chdir(r"C:\Users\gpngur01\Downloads\talent_intelligence-master\talent_intelligence-master\talent-intelligence")
sys.path.insert(0, os.getcwd())

# ── 9 Demo employees ──────────────────────────────────────────────────────
DEMO_EMPLOYEES = [

    # ══════════════════════════════════════════════════════
    # 🟢 STRONG 1 — Arjun Sharma
    # Company: Bosch + SAP | GitHub: langchain-rag + llm-toolkit
    # ══════════════════════════════════════════════════════
    {
        "tier": "STRONG",
        "full_name": "Arjun Sharma",
        "email": "arjun.sharma@demo.com",
        "title": "Senior Python & Microservices Engineer",
        "department": "Engineering",
        "seniority": "SENIOR",
        "is_sme": False,
        "sme_domains": [],
        "resume_text": (
            "Senior Python engineer with 6+ years building production microservices at Bosch GmbH and SAP SE. "
            "Built and maintained 8 production Python microservices handling 2M+ daily API requests using FastAPI. "
            "Led migration of legacy Django monolith to async FastAPI microservices — reduced latency by 40%. "
            "Designed PostgreSQL schema for real-time telemetry ingestion pipeline processing 500K events/hour. "
            "Personal projects: Built production-ready RAG pipeline for internal document Q&A using LangChain, "
            "FAISS, and OpenAI API — github.com/arjun-sharma-dev/langchain-rag-document-search — 89 commits. "
            "Also built LLM microservice toolkit with FastAPI + Langfuse observability — 43 commits."
        ),
        "github_username": "arjun-sharma-dev",
        "github_stats": {
            "username": "arjun-sharma-dev",
            "total_commits": 132,
            "top_languages": ["Python", "TypeScript", "SQL"],
            "active_repos": 4,
            "recent_repos": [
                {
                    "name": "langchain-rag-document-search",
                    "description": "Production-ready RAG pipeline for document Q&A using LangChain, FAISS, OpenAI",
                    "language": "Python",
                    "languages": ["Python", "TypeScript"],
                    "topics": ["langchain", "rag", "openai", "faiss", "fastapi"],
                    "readme_keywords": ["RAG", "LangChain", "FAISS", "OpenAI", "embeddings", "retrieval", "fastapi", "pydantic"],
                    "stars": 12,
                    "pushed_at": (datetime.now(timezone.utc) - timedelta(days=14)).isoformat(),
                },
                {
                    "name": "llm-microservice-toolkit",
                    "description": "Reusable FastAPI + LangChain template for LLM microservices with Langfuse observability",
                    "language": "Python",
                    "languages": ["Python"],
                    "topics": ["langchain", "fastapi", "langfuse", "llm", "microservices"],
                    "readme_keywords": ["LangChain", "FastAPI", "Langfuse", "observability", "LLM", "pydantic"],
                    "stars": 120,
                    "pushed_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                },
            ],
            "repo_skill_map": {
                "langchain": ["langchain-rag-document-search", "llm-microservice-toolkit"],
                "rag": ["langchain-rag-document-search"],
                "faiss": ["langchain-rag-document-search"],
                "openai": ["langchain-rag-document-search"],
                "fastapi": ["langchain-rag-document-search", "llm-microservice-toolkit"],
                "langfuse": ["llm-microservice-toolkit"],
                "python": ["langchain-rag-document-search", "llm-microservice-toolkit"],
            }
        },
        "availability_pct": 1.0,
        "availability_status": "AVAILABLE",
        "free_from_date": None,
        "skills": [
            {"name": "LangChain",       "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2025, "confidence": 0.96, "evidence": {"company": "personal project", "project": "LangChain RAG document search", "github_repo": "langchain-rag-document-search", "github_commits": 89, "github_confirmed": True}},
            {"name": "RAG",             "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2025, "confidence": 0.95, "evidence": {"company": "personal project", "project": "RAG pipeline for document Q&A", "github_repo": "langchain-rag-document-search", "github_commits": 89, "github_confirmed": True}},
            {"name": "FAISS",           "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2025, "confidence": 0.93, "evidence": {"company": "personal project", "project": "vector store for RAG", "github_repo": "langchain-rag-document-search", "github_confirmed": True}},
            {"name": "OpenAI API",      "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2025, "confidence": 0.94, "evidence": {"github_repo": "langchain-rag-document-search", "github_confirmed": True}},
            {"name": "FastAPI",         "type": "TECHNICAL", "source": "RESUME",   "proficiency": "EXPERT",       "is_hands_on": True,  "last_used": 2025, "confidence": 0.97, "evidence": {"company": "Bosch GmbH", "project": "8 production microservices", "github_confirmed": False}},
            {"name": "Python",          "type": "TECHNICAL", "source": "RESUME",   "proficiency": "EXPERT",       "is_hands_on": True,  "last_used": 2025, "confidence": 0.99, "evidence": {"company": "Bosch GmbH", "project": "microservices platform", "github_confirmed": False}},
            {"name": "PostgreSQL",      "type": "TECHNICAL", "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2025, "confidence": 0.95, "evidence": {"company": "Bosch GmbH", "project": "telemetry ingestion pipeline", "github_confirmed": False}},
            {"name": "Langfuse",        "type": "TOOL",      "source": "GITHUB",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2025, "confidence": 0.91, "evidence": {"github_repo": "llm-microservice-toolkit", "github_confirmed": True}},
            {"name": "Docker",          "type": "TOOL",      "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.90, "evidence": {"company": "Bosch GmbH", "github_confirmed": False}},
        ]
    },

    # ══════════════════════════════════════════════════════
    # 🟢 STRONG 2 — Priya Nair
    # Company: Allianz + MunichRe | GitHub: rag-eval + vector-benchmarks
    # ══════════════════════════════════════════════════════
    {
        "tier": "STRONG",
        "full_name": "Priya Nair",
        "email": "priya.nair@demo.com",
        "title": "ML Engineer — LLM & RAG Systems",
        "department": "AI",
        "seniority": "SENIOR",
        "is_sme": False,
        "sme_domains": [],
        "resume_text": (
            "ML Engineer with 5 years specialising in NLP and retrieval systems at Allianz SE and MunichRe. "
            "Built RAG pipeline for insurance policy document Q&A serving 10,000+ internal users — "
            "semantic search over 500K+ documents using FAISS + OpenAI embeddings, under 1s latency. "
            "Evaluated 6 LLM providers saving EUR 80K/year. Implemented Langfuse observability tracking 2M+ LLM calls. "
            "Personal GitHub: open-source RAG evaluation framework — github.com/priya-nair-ml/rag-evaluation-framework — "
            "134 commits, 47 stars. Also vector DB benchmarks comparing FAISS, ChromaDB, Pinecone — 28 commits."
        ),
        "github_username": "priya-nair-ml",
        "github_stats": {
            "username": "priya-nair-ml",
            "total_commits": 162,
            "top_languages": ["Python", "Jupyter Notebook"],
            "active_repos": 3,
            "recent_repos": [
                {
                    "name": "rag-evaluation-framework",
                    "description": "RAG evaluation framework with Hit@K, MRR, faithfulness metrics — 47 stars",
                    "language": "Python",
                    "languages": ["Python"],
                    "topics": ["rag", "evaluation", "langchain", "llm", "faiss", "chromadb"],
                    "readme_keywords": ["RAG", "evaluation", "LangChain", "FAISS", "Hit@K", "MRR", "faithfulness", "llm"],
                    "stars": 47,
                    "pushed_at": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
                },
                {
                    "name": "vector-db-benchmarks",
                    "description": "Performance benchmarks: FAISS vs ChromaDB vs Pinecone vs Weaviate",
                    "language": "Python",
                    "languages": ["Python", "Jupyter Notebook"],
                    "topics": ["faiss", "chromadb", "pinecone", "weaviate", "benchmarks", "vector-database"],
                    "readme_keywords": ["FAISS", "ChromaDB", "Pinecone", "Weaviate", "recall", "latency", "benchmarks"],
                    "stars": 8,
                    "pushed_at": (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
                },
            ],
            "repo_skill_map": {
                "rag": ["rag-evaluation-framework"],
                "langchain": ["rag-evaluation-framework"],
                "faiss": ["rag-evaluation-framework", "vector-db-benchmarks"],
                "chromadb": ["vector-db-benchmarks"],
                "pinecone": ["vector-db-benchmarks"],
                "llm": ["rag-evaluation-framework"],
                "python": ["rag-evaluation-framework", "vector-db-benchmarks"],
                "evaluation": ["rag-evaluation-framework"],
            }
        },
        "availability_pct": 0.8,
        "availability_status": "PARTIALLY_AVAILABLE",
        "free_from_date": (datetime.now(timezone.utc) + timedelta(days=14)).isoformat(),
        "skills": [
            {"name": "RAG",             "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "EXPERT",       "is_hands_on": True,  "last_used": 2025, "confidence": 0.97, "evidence": {"company": "Allianz SE + personal", "project": "RAG pipeline + eval framework", "github_repo": "rag-evaluation-framework", "github_commits": 134, "github_confirmed": True}},
            {"name": "FAISS",           "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "EXPERT",       "is_hands_on": True,  "last_used": 2025, "confidence": 0.96, "evidence": {"github_repo": "rag-evaluation-framework", "github_commits": 134, "github_confirmed": True}},
            {"name": "LangChain",       "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2025, "confidence": 0.94, "evidence": {"github_repo": "rag-evaluation-framework", "github_confirmed": True}},
            {"name": "OpenAI API",      "type": "TECHNICAL", "source": "RESUME",   "proficiency": "EXPERT",       "is_hands_on": True,  "last_used": 2024, "confidence": 0.96, "evidence": {"company": "Allianz SE", "project": "insurance document Q&A", "github_confirmed": False}},
            {"name": "LLM evaluation",  "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "EXPERT",       "is_hands_on": True,  "last_used": 2025, "confidence": 0.95, "evidence": {"github_repo": "rag-evaluation-framework", "github_confirmed": True}},
            {"name": "Langfuse",        "type": "TOOL",      "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.94, "evidence": {"company": "Allianz SE", "project": "LLM observability 2M+ calls", "github_confirmed": False}},
            {"name": "PyTorch",         "type": "TECHNICAL", "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.91, "evidence": {"company": "MunichRe", "project": "BERT claims classification", "github_confirmed": False}},
            {"name": "Python",          "type": "TECHNICAL", "source": "RESUME",   "proficiency": "EXPERT",       "is_hands_on": True,  "last_used": 2025, "confidence": 0.99, "evidence": {"company": "Allianz SE", "github_confirmed": False}},
        ]
    },

    # ══════════════════════════════════════════════════════
    # 🟢 STRONG 3 — Rahul Mehta
    # Company: Siemens + Zalando | GitHub: ai-chat + fastapi-react
    # ══════════════════════════════════════════════════════
    {
        "tier": "STRONG",
        "full_name": "Rahul Mehta",
        "email": "rahul.mehta@demo.com",
        "title": "Senior Full Stack Engineer — Python & React",
        "department": "Engineering",
        "seniority": "SENIOR",
        "is_sme": False,
        "sme_domains": [],
        "resume_text": (
            "Full stack engineer with 7 years building B2B SaaS products at Siemens AG and Zalando SE. "
            "Built internal knowledge management platform for 5,000+ engineers — integrated GPT-4 API for "
            "document summarisation, reducing research time by 60%. React + TypeScript frontend, FastAPI backend. "
            "Personal GitHub: full stack AI chat assistant — github.com/rahulmehta-dev/ai-chat-assistant — "
            "67 commits, FastAPI + LangChain + React, live demo deployed. "
            "Also FastAPI + React production starter — 91 commits, 120 GitHub stars."
        ),
        "github_username": "rahulmehta-dev",
        "github_stats": {
            "username": "rahulmehta-dev",
            "total_commits": 158,
            "top_languages": ["Python", "TypeScript", "JavaScript"],
            "active_repos": 5,
            "recent_repos": [
                {
                    "name": "ai-chat-assistant",
                    "description": "Full stack AI chat — FastAPI + LangChain + OpenAI + React + PostgreSQL",
                    "language": "Python",
                    "languages": ["Python", "TypeScript"],
                    "topics": ["langchain", "openai", "react", "fastapi", "chatbot", "rag"],
                    "readme_keywords": ["FastAPI", "LangChain", "OpenAI", "React", "TypeScript", "RAG", "PostgreSQL", "chat"],
                    "stars": 18,
                    "pushed_at": (datetime.now(timezone.utc) - timedelta(days=21)).isoformat(),
                },
                {
                    "name": "fastapi-react-starter",
                    "description": "Production FastAPI + React starter with JWT, Docker, GitHub Actions",
                    "language": "Python",
                    "languages": ["Python", "TypeScript"],
                    "topics": ["fastapi", "react", "typescript", "docker", "jwt", "postgresql"],
                    "readme_keywords": ["FastAPI", "React", "TypeScript", "Docker", "JWT", "PostgreSQL", "CI/CD"],
                    "stars": 120,
                    "pushed_at": (datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),
                },
            ],
            "repo_skill_map": {
                "langchain": ["ai-chat-assistant"],
                "openai": ["ai-chat-assistant"],
                "rag": ["ai-chat-assistant"],
                "react": ["ai-chat-assistant", "fastapi-react-starter"],
                "fastapi": ["ai-chat-assistant", "fastapi-react-starter"],
                "typescript": ["ai-chat-assistant", "fastapi-react-starter"],
                "python": ["ai-chat-assistant", "fastapi-react-starter"],
                "docker": ["fastapi-react-starter"],
                "postgresql": ["ai-chat-assistant", "fastapi-react-starter"],
            }
        },
        "availability_pct": 0.6,
        "availability_status": "PARTIALLY_AVAILABLE",
        "free_from_date": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        "skills": [
            {"name": "FastAPI",         "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "EXPERT",       "is_hands_on": True,  "last_used": 2025, "confidence": 0.97, "evidence": {"company": "Siemens AG + personal", "project": "knowledge platform + ai-chat-assistant", "github_repo": "ai-chat-assistant", "github_commits": 67, "github_confirmed": True}},
            {"name": "React",           "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2025, "confidence": 0.96, "evidence": {"github_repo": "ai-chat-assistant", "github_confirmed": True}},
            {"name": "TypeScript",      "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2025, "confidence": 0.95, "evidence": {"github_repo": "ai-chat-assistant", "github_confirmed": True}},
            {"name": "OpenAI API",      "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2025, "confidence": 0.94, "evidence": {"github_repo": "ai-chat-assistant", "github_confirmed": True}},
            {"name": "LangChain",       "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2025, "confidence": 0.91, "evidence": {"github_repo": "ai-chat-assistant", "github_confirmed": True}},
            {"name": "RAG",             "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2025, "confidence": 0.89, "evidence": {"github_repo": "ai-chat-assistant", "github_confirmed": True}},
            {"name": "Python",          "type": "TECHNICAL", "source": "RESUME",   "proficiency": "EXPERT",       "is_hands_on": True,  "last_used": 2025, "confidence": 0.99, "evidence": {"company": "Siemens AG", "github_confirmed": False}},
            {"name": "PostgreSQL",      "type": "TECHNICAL", "source": "GITHUB",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2025, "confidence": 0.93, "evidence": {"github_repo": "fastapi-react-starter", "github_confirmed": True}},
            {"name": "Docker",          "type": "TOOL",      "source": "GITHUB",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2025, "confidence": 0.92, "evidence": {"github_repo": "fastapi-react-starter", "github_confirmed": True}},
        ]
    },

    # ══════════════════════════════════════════════════════
    # 🟡 AVERAGE 1 — Divya Singh
    # Company: Deutsche Bank + Continental | No GitHub
    # ══════════════════════════════════════════════════════
    {
        "tier": "AVERAGE",
        "full_name": "Divya Singh",
        "email": "divya.singh@demo.com",
        "title": "Senior Backend Engineer",
        "department": "Engineering",
        "seniority": "SENIOR",
        "is_sme": False,
        "sme_domains": [],
        "resume_text": (
            "Senior backend engineer with 5 years building Python APIs at Deutsche Bank and Continental AG. "
            "Built internal document processing API at Deutsche Bank using FastAPI — integrated OpenAI API "
            "for document classification and summarisation of compliance documents. "
            "Developed LangChain-based prototype for compliance document Q&A at internal hackathon. "
            "All company code on internal GitLab — no public GitHub. "
            "Designed PostgreSQL schema for trade reconciliation system handling 1M+ daily transactions. "
            "Skills: Python, FastAPI, OpenAI API, LangChain basics, PostgreSQL, Docker, REST APIs."
        ),
        "github_username": None,
        "github_stats": {
            "username": None,
            "total_commits": 0,
            "top_languages": [],
            "active_repos": 0,
            "recent_repos": [],
            "repo_skill_map": {}
        },
        "availability_pct": 1.0,
        "availability_status": "AVAILABLE",
        "free_from_date": None,
        "skills": [
            {"name": "Python",          "type": "TECHNICAL", "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.92, "evidence": {"company": "Deutsche Bank", "project": "document processing API", "github_confirmed": False}},
            {"name": "FastAPI",         "type": "TECHNICAL", "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.91, "evidence": {"company": "Deutsche Bank", "project": "compliance document API", "github_confirmed": False}},
            {"name": "OpenAI API",      "type": "TECHNICAL", "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.74, "evidence": {"company": "Deutsche Bank", "project": "document classification", "github_confirmed": False}},
            {"name": "LangChain",       "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": 2024, "confidence": 0.62, "evidence": {"company": "Deutsche Bank", "project": "hackathon prototype only", "github_confirmed": False}},
            {"name": "RAG",             "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": 2024, "confidence": 0.58, "evidence": {"company": "Deutsche Bank", "project": "hackathon concept", "github_confirmed": False}},
            {"name": "PostgreSQL",      "type": "TECHNICAL", "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.93, "evidence": {"company": "Deutsche Bank", "project": "trade reconciliation system", "github_confirmed": False}},
            {"name": "Docker",          "type": "TOOL",      "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.88, "evidence": {"company": "Continental AG", "github_confirmed": False}},
        ]
    },

    # ══════════════════════════════════════════════════════
    # 🟡 AVERAGE 2 — Kiran Patel
    # Company: BMW + Lufthansa | No GitHub | Only 40% available
    # ══════════════════════════════════════════════════════
    {
        "tier": "AVERAGE",
        "full_name": "Kiran Patel",
        "email": "kiran.patel@demo.com",
        "title": "Data Scientist — NLP & Machine Learning",
        "department": "Data",
        "seniority": "MID",
        "is_sme": False,
        "sme_domains": [],
        "resume_text": (
            "Data scientist with 4 years at BMW AG and Lufthansa. "
            "Built NLP classification models for customer feedback at BMW — 200K+ monthly reviews, 88% accuracy. "
            "Used GPT-3.5 API for automated executive report generation at BMW — reduced analyst time by 25%. "
            "Experimented with Pinecone vector database for semantic search — prototype only, not in production. "
            "All BMW and Lufthansa code on internal systems — no public GitHub. "
            "Skills: Python, PyTorch, BERT, scikit-learn, GPT-3.5 API, Pinecone (prototype), NLP, SQL, Spark."
        ),
        "github_username": None,
        "github_stats": {
            "username": None,
            "total_commits": 0,
            "top_languages": [],
            "active_repos": 0,
            "recent_repos": [],
            "repo_skill_map": {}
        },
        "availability_pct": 0.4,
        "availability_status": "PARTIALLY_AVAILABLE",
        "free_from_date": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        "skills": [
            {"name": "Python",          "type": "TECHNICAL", "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.91, "evidence": {"company": "BMW AG", "project": "NLP classification models", "github_confirmed": False}},
            {"name": "GPT-3.5 API",     "type": "TECHNICAL", "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.73, "evidence": {"company": "BMW AG", "project": "automated report generation", "github_confirmed": False}},
            {"name": "LLM APIs",        "type": "TECHNICAL", "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.70, "evidence": {"company": "BMW AG", "github_confirmed": False}},
            {"name": "RAG",             "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": 2024, "confidence": 0.52, "evidence": {"company": "BMW AG", "project": "Pinecone prototype only", "github_confirmed": False}},
            {"name": "PyTorch",         "type": "TECHNICAL", "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.88, "evidence": {"company": "BMW AG", "project": "NLP models", "github_confirmed": False}},
            {"name": "BERT",            "type": "TECHNICAL", "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.86, "evidence": {"company": "BMW AG", "project": "claims classification", "github_confirmed": False}},
            {"name": "scikit-learn",    "type": "TECHNICAL", "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.91, "evidence": {"company": "BMW AG", "github_confirmed": False}},
        ]
    },

    # ══════════════════════════════════════════════════════
    # 🟡 AVERAGE 3 — Amit Kumar
    # Company: E.ON + Henkel | No GitHub | 80% available
    # ══════════════════════════════════════════════════════
    {
        "tier": "AVERAGE",
        "full_name": "Amit Kumar",
        "email": "amit.kumar@demo.com",
        "title": "Senior Python Developer",
        "department": "Engineering",
        "seniority": "SENIOR",
        "is_sme": False,
        "sme_domains": [],
        "resume_text": (
            "Senior Python developer with 7 years at E.ON SE and Henkel AG. "
            "Built internal HR document management system at E.ON — integrated GPT-3.5 API for "
            "document summarisation serving 2,000+ HR staff. All code on internal GitLab. "
            "Developed Python microservices for energy consumption data APIs — 10M+ data points daily. "
            "Built REST API integrations with SAP HCM for employee data synchronisation. "
            "No public GitHub — all company work done on internal systems. "
            "Skills: Python, FastAPI, Django, OpenAI API basics, PostgreSQL, Docker, REST APIs."
        ),
        "github_username": None,
        "github_stats": {
            "username": None,
            "total_commits": 0,
            "top_languages": [],
            "active_repos": 0,
            "recent_repos": [],
            "repo_skill_map": {}
        },
        "availability_pct": 0.8,
        "availability_status": "PARTIALLY_AVAILABLE",
        "free_from_date": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        "skills": [
            {"name": "Python",          "type": "TECHNICAL", "source": "RESUME",   "proficiency": "EXPERT",       "is_hands_on": True,  "last_used": 2024, "confidence": 0.95, "evidence": {"company": "E.ON SE", "project": "energy data microservices", "github_confirmed": False}},
            {"name": "FastAPI",         "type": "TECHNICAL", "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.91, "evidence": {"company": "E.ON SE", "project": "HR document system", "github_confirmed": False}},
            {"name": "OpenAI API",      "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.65, "evidence": {"company": "E.ON SE", "project": "document summarisation", "github_confirmed": False}},
            {"name": "LLM integration", "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.60, "evidence": {"company": "E.ON SE", "github_confirmed": False}},
            {"name": "PostgreSQL",      "type": "TECHNICAL", "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.92, "evidence": {"company": "E.ON SE", "project": "billing system 5M customers", "github_confirmed": False}},
            {"name": "Django",          "type": "TECHNICAL", "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2023, "confidence": 0.90, "evidence": {"company": "Henkel AG", "github_confirmed": False}},
            {"name": "Docker",          "type": "TOOL",      "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.88, "evidence": {"company": "E.ON SE", "github_confirmed": False}},
        ]
    },

    # ══════════════════════════════════════════════════════
    # 🔴 WEAK 1 — Sneha Reddy — Java dev, no AI experience
    # ══════════════════════════════════════════════════════
    {
        "tier": "WEAK",
        "full_name": "Sneha Reddy",
        "email": "sneha.reddy@demo.com",
        "title": "Java Developer",
        "department": "Engineering",
        "seniority": "JUNIOR",
        "is_sme": False,
        "sme_domains": [],
        "resume_text": (
            "Java developer with 3 years at IT Consulting GmbH. "
            "Developed Java Spring Boot REST APIs for client projects — CRUD operations and database integrations. "
            "Did some Python scripting for automation tasks. "
            "Completed Udemy machine learning fundamentals course. "
            "Familiar with Python basics and Jupyter notebooks. "
            "Interested in transitioning into AI development. "
            "Skills: Java, Spring Boot, Python (basic), SQL, Git, Jira."
        ),
        "github_username": None,
        "github_stats": {
            "username": None,
            "total_commits": 0,
            "top_languages": [],
            "active_repos": 0,
            "recent_repos": [],
            "repo_skill_map": {}
        },
        "availability_pct": 1.0,
        "availability_status": "AVAILABLE",
        "free_from_date": None,
        "skills": [
            {"name": "Java",            "type": "TECHNICAL", "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.88, "evidence": {"company": "IT Consulting GmbH", "project": "Spring Boot APIs", "github_confirmed": False}},
            {"name": "Python",          "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": 2024, "confidence": 0.42, "evidence": {"company": "IT Consulting GmbH", "project": "automation scripts only", "github_confirmed": False}},
            {"name": "LangChain",       "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": None,  "confidence": 0.21, "evidence": {"project": "online course only, no project", "github_confirmed": False}},
            {"name": "RAG",             "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": None,  "confidence": 0.18, "evidence": {"project": "theoretical knowledge only", "github_confirmed": False}},
            {"name": "SQL",             "type": "TECHNICAL", "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.85, "evidence": {"company": "IT Consulting GmbH", "github_confirmed": False}},
        ]
    },

    # ══════════════════════════════════════════════════════
    # 🔴 WEAK 2 — Vikram Joshi — QA engineer, wrong domain
    # ══════════════════════════════════════════════════════
    {
        "tier": "WEAK",
        "full_name": "Vikram Joshi",
        "email": "vikram.joshi@demo.com",
        "title": "QA Automation Engineer",
        "department": "QA",
        "seniority": "MID",
        "is_sme": False,
        "sme_domains": [],
        "resume_text": (
            "QA automation engineer with 4 years at Software Testing AG. "
            "Wrote Selenium + Pytest automation scripts for web application testing. "
            "Tested REST APIs using Postman and Python requests library. "
            "Knows Python primarily for writing test scripts. "
            "Interested in LangChain and AI development but no hands-on project experience. "
            "Read about RAG online — no practical implementation. "
            "Skills: Python (testing), Selenium, Pytest, Postman, REST API testing, SQL basics, Git."
        ),
        "github_username": None,
        "github_stats": {
            "username": None,
            "total_commits": 0,
            "top_languages": [],
            "active_repos": 0,
            "recent_repos": [],
            "repo_skill_map": {}
        },
        "availability_pct": 1.0,
        "availability_status": "AVAILABLE",
        "free_from_date": None,
        "skills": [
            {"name": "Python",          "type": "TECHNICAL", "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.72, "evidence": {"company": "Software Testing AG", "project": "test automation only", "github_confirmed": False}},
            {"name": "Selenium",        "type": "TOOL",      "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.90, "evidence": {"company": "Software Testing AG", "github_confirmed": False}},
            {"name": "Pytest",          "type": "TOOL",      "source": "RESUME",   "proficiency": "ADVANCED",     "is_hands_on": True,  "last_used": 2024, "confidence": 0.91, "evidence": {"company": "Software Testing AG", "github_confirmed": False}},
            {"name": "LangChain",       "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": None,  "confidence": 0.15, "evidence": {"project": "self-study only, no project", "github_confirmed": False}},
            {"name": "RAG",             "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": None,  "confidence": 0.12, "evidence": {"project": "read about it online", "github_confirmed": False}},
            {"name": "REST APIs",       "type": "TECHNICAL", "source": "RESUME",   "proficiency": "INTERMEDIATE", "is_hands_on": True,  "last_used": 2024, "confidence": 0.82, "evidence": {"company": "Software Testing AG", "project": "API testing", "github_confirmed": False}},
        ]
    },

    # ══════════════════════════════════════════════════════
    # 🔴 WEAK 3 — Meera Iyer — Project manager, no dev skills
    # ══════════════════════════════════════════════════════
    {
        "tier": "WEAK",
        "full_name": "Meera Iyer",
        "email": "meera.iyer@demo.com",
        "title": "Technical Project Manager",
        "department": "Management",
        "seniority": "SENIOR",
        "is_sme": False,
        "sme_domains": [],
        "resume_text": (
            "Technical project manager with 5 years at Digital Solutions GmbH. PMP certified. "
            "Managed delivery of 8 software projects including 2 AI/ML projects. "
            "Wrote Python automation scripts for project reporting — not application development. "
            "Familiar with LLM and RAG concepts from managing AI projects — no hands-on implementation. "
            "Oversaw API integration projects — reviewed specs, coordinated vendors, did not write code. "
            "Skills: Python (scripting only), SQL basics, API concepts, Jira, Confluence, PowerBI, Agile."
        ),
        "github_username": None,
        "github_stats": {
            "username": None,
            "total_commits": 0,
            "top_languages": [],
            "active_repos": 0,
            "recent_repos": [],
            "repo_skill_map": {}
        },
        "availability_pct": 1.0,
        "availability_status": "AVAILABLE",
        "free_from_date": None,
        "skills": [
            {"name": "Python",          "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": 2024, "confidence": 0.45, "evidence": {"company": "Digital Solutions GmbH", "project": "scripting only not development", "github_confirmed": False}},
            {"name": "LangChain",       "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": None,  "confidence": 0.14, "evidence": {"project": "conceptual knowledge from managing projects", "github_confirmed": False}},
            {"name": "RAG",             "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": None,  "confidence": 0.12, "evidence": {"project": "heard about it in project meetings", "github_confirmed": False}},
            {"name": "Agile",           "type": "DOMAIN",    "source": "RESUME",   "proficiency": "EXPERT",       "is_hands_on": True,  "last_used": 2025, "confidence": 0.95, "evidence": {"company": "Digital Solutions GmbH", "github_confirmed": False}},
            {"name": "SQL",             "type": "TECHNICAL", "source": "RESUME",   "proficiency": "BEGINNER",     "is_hands_on": False, "last_used": 2023, "confidence": 0.50, "evidence": {"project": "basic reporting queries", "github_confirmed": False}},
        ]
    },
]


async def run():
    from backend.db.database import AsyncSessionLocal, init_db
    from backend.models.employee import Employee, SeniorityLevel, EmploymentType
    from backend.models.skill import Skill, SkillType, SkillSource, ProficiencyLevel
    from backend.models.availability import Availability, AvailabilityStatus
    from sqlalchemy import delete

    await init_db()
    async with AsyncSessionLocal() as db:

        # Clear existing demo employees
        print("Clearing previous demo data...")
        demo_emails = [e["email"] for e in DEMO_EMPLOYEES]
        await db.execute(delete(Employee).where(Employee.email.in_(demo_emails)))
        await db.flush()

        print(f"Loading {len(DEMO_EMPLOYEES)} demo employees...")

        for data in DEMO_EMPLOYEES:
            emp_id = str(uuid.uuid4())

            # Create employee
            emp = Employee(
                id=emp_id,
                email=data["email"],
                full_name=data["full_name"],
                title=data["title"],
                department=data["department"],
                seniority_level=SeniorityLevel[data["seniority"]],
                employment_type=EmploymentType.FULL_TIME,
                resume_text=data["resume_text"],
                github_username=data["github_username"],
                github_stats=data["github_stats"],
                is_active=True,
                is_sme=data["is_sme"],
                sme_domains=data["sme_domains"],
                skill_extraction_confidence=0.85,
            )
            db.add(emp)
            await db.flush()

            # Create skills
            for s in data["skills"]:
                skill = Skill(
                    id=str(uuid.uuid4()),
                    employee_id=emp_id,
                    name=s["name"],
                    normalized_name=s["name"].lower().strip(),
                    skill_type=SkillType[s["type"]],
                    source=SkillSource[s["source"]],
                    proficiency=ProficiencyLevel[s["proficiency"]],
                    is_hands_on=s["is_hands_on"],
                    last_used_year=s.get("last_used"),
                    extraction_confidence=s["confidence"],
                    evidence=s.get("evidence"),
                )
                db.add(skill)

            # Create availability
            avail = Availability(
                id=str(uuid.uuid4()),
                employee_id=emp_id,
                available_percentage=data["availability_pct"],
                status=AvailabilityStatus[data["availability_status"]],
                free_from_date=datetime.fromisoformat(data["free_from_date"]) if data.get("free_from_date") else None,
                availability_score=data["availability_pct"],
            )
            db.add(avail)
            await db.flush()

            tier_emoji = {"STRONG": "🟢", "AVERAGE": "🟡", "WEAK": "🔴"}[data["tier"]]
            print(f"  {tier_emoji} {data['tier']:8} | {data['full_name']:<20} | {len(data['skills'])} skills | {int(data['availability_pct']*100)}% available")

        await db.commit()
        print(f"\nDemo seed complete! {len(DEMO_EMPLOYEES)} employees loaded.")
        print("\nNow run: python build_faiss.py")

asyncio.run(run())

"""
vector_store.py — manages the FAISS vector index.
Employees are indexed by employee_id. The index is persisted to disk
and reloaded on startup. Pinecone swap-in requires only changing this file.
"""
import json
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from backend.rag.embeddings import get_embedding_model, build_employee_embedding_text
from backend.config import settings


INDEX_PATH = Path(settings.FAISS_INDEX_PATH)


def _load_or_create_index() -> FAISS | None:
    if (INDEX_PATH / "index.faiss").exists():
        return FAISS.load_local(
            str(INDEX_PATH),
            get_embedding_model(),
            allow_dangerous_deserialization=True,
        )
    return None


async def index_employee(employee: dict) -> None:
    """Add or update one employee in the vector index."""
    text = build_employee_embedding_text(employee)
    doc = Document(
        page_content=text,
        metadata={
            "employee_id": employee["id"],
            "title": employee.get("title", ""),
            "seniority": employee.get("seniority_level", ""),
            "is_sme": employee.get("is_sme", False),
        }
    )
    store = _load_or_create_index()
    if store is None:
        store = await FAISS.afrom_documents([doc], get_embedding_model())
    else:
        await store.aadd_documents([doc])

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_PATH))


async def index_all_employees(employees: list[dict]) -> None:
    """Bulk index — used during seeding and full reindex jobs."""
    texts = [build_employee_embedding_text(e) for e in employees]
    docs = [
        Document(
            page_content=text,
            metadata={
                "employee_id": emp["id"],
                "title": emp.get("title", ""),
                "seniority": emp.get("seniority_level", ""),
                "is_sme": emp.get("is_sme", False),
            }
        )
        for text, emp in zip(texts, employees)
    ]
    store = await FAISS.afrom_documents(docs, get_embedding_model())
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_PATH))


def get_vector_store() -> FAISS | None:
    return _load_or_create_index()

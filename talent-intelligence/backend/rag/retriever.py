"""
retriever.py — semantic search over the employee vector index.
Takes an enriched requirement search_query and returns top-k candidates.
"""
from langchain_community.vectorstores import FAISS
from backend.rag.vector_store import get_vector_store
from backend.config import settings


async def retrieve_candidates(
    search_query: str,
    top_k: int = None,
    filter_sme_only: bool = False,
) -> list[dict]:
    """
    Semantic search against the FAISS index.
    Returns list of {employee_id, score, metadata} dicts.
    score is cosine similarity (0-1, higher is better).
    """
    store: FAISS = get_vector_store()
    if store is None:
        return []

    k = top_k or settings.RAG_TOP_K

    # FAISS similarity_search_with_score returns (Document, distance)
    # distance is L2 — convert to similarity score
    results = store.similarity_search_with_score(search_query, k=k)

    candidates = []
    for doc, distance in results:
        # Convert L2 distance to 0-1 similarity
        similarity = 1 / (1 + distance)

        if similarity < settings.RAG_SCORE_THRESHOLD:
            continue

        if filter_sme_only and not doc.metadata.get("is_sme"):
            continue

        candidates.append({
            "employee_id": doc.metadata["employee_id"],
            "retrieval_score": round(similarity, 4),
            "title": doc.metadata.get("title", ""),
            "seniority": doc.metadata.get("seniority", ""),
            "matched_text": doc.page_content,
        })

    return candidates


async def retrieve_sme_candidates(domain: str) -> list[dict]:
    """
    Specialized retrieval for SME consultation.
    Searches for employees flagged as SMEs in the given domain.
    """
    query = f"Subject matter expert in {domain} with deep hands-on experience"
    return await retrieve_candidates(query, top_k=5, filter_sme_only=True)

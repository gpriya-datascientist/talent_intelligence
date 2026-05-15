"""
embeddings.py — creates text embeddings for employee skill profiles.
Uses OpenAI text-embedding-3-small. Each employee gets one embedding
built from a rich text representation of their skills.
"""
from langchain_openai import OpenAIEmbeddings
from backend.config import settings


def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        dimensions=settings.EMBEDDING_DIMENSION,
    )


def build_employee_embedding_text(employee: dict) -> str:
    """
    Converts employee profile into a rich text string for embedding.
    This text is what gets embedded — quality here directly affects retrieval.
    We include: title, skills with proficiency, hands-on flags, and evidence.
    """
    lines = [
        f"Name: {employee.get('full_name', '')}",
        f"Title: {employee.get('title', '')}",
        f"Seniority: {employee.get('seniority_level', '')}",
        f"Department: {employee.get('department', '')}",
    ]

    skills = employee.get("skills", [])
    if skills:
        skill_lines = []
        for s in skills:
            hands_on = "hands-on" if s.get("is_hands_on") else "theoretical"
            skill_lines.append(
                f"{s['name']} ({s.get('proficiency','intermediate')}, {hands_on}, "
                f"last used {s.get('last_used_year','unknown')})"
            )
        lines.append("Skills: " + "; ".join(skill_lines))

    if employee.get("resume_text"):
        lines.append(f"Background: {employee['resume_text'][:500]}")

    github = employee.get("github_stats", {})
    if github.get("top_languages"):
        lines.append(f"GitHub languages: {', '.join(github['top_languages'])}")

    return "\n".join(lines)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return await model.aembed_documents(texts)


async def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    return await model.aembed_query(query)

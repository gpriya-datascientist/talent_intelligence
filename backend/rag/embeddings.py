"""
embeddings.py — ENHANCED employee embedding text builder.
KEY CHANGE: FAISS vector now includes WHERE skills were used and WHAT was built.
This means FAISS can match "someone who built a data pipeline" not just "knows Python".
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
    ENHANCED — converts employee profile into a rich text string for embedding.

    BEFORE: "Python (advanced, hands-on, last used 2024)"
    AFTER:  "Python — Bosch: real-time Kafka data pipeline (2022-2024)
             [GitHub: kafka-pipeline, 120 commits] verified"

    This means FAISS can now match on WHAT was built, not just WHAT is known.
    Hidden talent who built real things surfaces correctly.
    """
    lines = [
        f"Name: {employee.get('full_name', '')}",
        f"Title: {employee.get('title', '')}",
        f"Seniority: {employee.get('seniority_level', '')}",
        f"Department: {employee.get('department', '')}",
    ]

    skills = employee.get("skills", [])
    if skills:
        verified_lines   = []   # GitHub confirmed + resume context
        hands_on_lines   = []   # resume context only, no GitHub
        theoretical_lines = []  # just skill name, no project context

        for s in skills:
            evidence = s.get("evidence") or {}

            # Build the context string — WHERE and WHAT
            context_parts = []

            company = evidence.get("company") if isinstance(evidence, dict) else None
            project = evidence.get("project") if isinstance(evidence, dict) else None
            used_with = evidence.get("used_with") if isinstance(evidence, dict) else None
            github_repo = evidence.get("github_repo") if isinstance(evidence, dict) else None
            github_commits = evidence.get("github_commits") if isinstance(evidence, dict) else None
            github_confirmed = evidence.get("github_confirmed", False) if isinstance(evidence, dict) else False

            # Build context: "Bosch: real-time Kafka pipeline"
            if company and project:
                context_parts.append(f"{company}: {project}")
            elif company:
                context_parts.append(company)
            elif project:
                context_parts.append(project)

            # Add co-technologies: "with Kafka, PostgreSQL"
            if used_with and isinstance(used_with, list):
                context_parts.append(f"with {', '.join(used_with[:4])}")

            # Add year range
            year = s.get("last_used_year")
            if year:
                context_parts.append(f"({year})")

            # Build GitHub proof string
            github_proof = ""
            if github_repo:
                if github_commits:
                    github_proof = f"[GitHub: {github_repo}, {github_commits} commits] ✓ verified"
                else:
                    github_proof = f"[GitHub: {github_repo}] ✓ verified"

            # Skill name + proficiency
            skill_name = s.get("name", "")
            proficiency = s.get("proficiency", "intermediate")
            hands_on = s.get("is_hands_on", False)

            if context_parts:
                context_str = " ".join(context_parts)
                skill_line = f"{skill_name} ({proficiency}) — {context_str}"
                if github_proof:
                    skill_line += f" {github_proof}"
                    verified_lines.append(skill_line)
                else:
                    hands_on_lines.append(skill_line)
            else:
                # No context — just skill name and level
                label = "hands-on" if hands_on else "theoretical"
                theoretical_lines.append(f"{skill_name} ({proficiency}, {label})")

        # Put verified skills first — they matter most for matching
        all_skill_lines = verified_lines + hands_on_lines + theoretical_lines
        if all_skill_lines:
            lines.append("Skills:")
            lines.extend([f"  {sl}" for sl in all_skill_lines])

        # Summary counts — helps FAISS understand depth
        if verified_lines:
            lines.append(f"GitHub-verified skills: {len(verified_lines)}")
        if hands_on_lines:
            lines.append(f"Project-evidenced skills: {len(hands_on_lines)}")

    # Resume background — first 500 chars for general context
    if employee.get("resume_text"):
        lines.append(f"Background: {employee['resume_text'][:500]}")

    # GitHub summary
    github = employee.get("github_stats", {})
    if github.get("top_languages"):
        lines.append(f"GitHub languages: {', '.join(github['top_languages'])}")
    if github.get("active_repos"):
        lines.append(f"Active GitHub repos: {github['active_repos']}")

    return "\n".join(lines)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return await model.aembed_documents(texts)


async def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    return await model.aembed_query(query)

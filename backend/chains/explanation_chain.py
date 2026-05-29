"""
explanation_chain.py — generates "why this match" explanations with Langfuse tracing.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.config import settings
from backend.observability import get_langfuse_handler


EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a staffing advisor explaining candidate recommendations to a Product Owner.
Write a 2-3 sentence explanation of why this person is a strong match.
Be specific — mention actual skills, projects, and evidence.
Do NOT use generic phrases like 'strong communicator' or 'team player'.
Focus on technical fit and concrete proof."""),
    ("human", """Project requirement: {requirement_summary}
Candidate: {candidate_name} — {candidate_title}
Matched skills: {matched_skills}
Skill evidence: {skill_evidence}
Availability: {availability}
Match score: {match_score}

Write the explanation."""),
])


def build_explanation_chain():
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        max_tokens=300,
        temperature=0.3,
    )
    return EXPLANATION_PROMPT | llm | StrOutputParser()


async def explain_match(
    requirement_summary: str,
    candidate_name: str,
    candidate_title: str,
    matched_skills: list[str],
    skill_evidence: list[str],
    availability: str,
    match_score: float,
    wish_id: str = None,
    candidate_id: str = None,
) -> str:
    chain = build_explanation_chain()

    # ── Langfuse tracing ─────────────────────────────────────────────────
    handler = get_langfuse_handler(
        trace_name="explanation_chain",
        wish_id=wish_id,
        metadata={
            "candidate_id":    candidate_id,
            "candidate_name":  candidate_name,
            "match_score":     match_score,
            "matched_skills":  matched_skills,
        },
    )
    config = {"callbacks": [handler]} if handler else {}

    return await chain.ainvoke({
        "requirement_summary": requirement_summary,
        "candidate_name":      candidate_name,
        "candidate_title":     candidate_title,
        "matched_skills":      ", ".join(matched_skills),
        "skill_evidence":      "\n".join(skill_evidence),
        "availability":        availability,
        "match_score":         f"{match_score:.0%}",
    }, config=config)

"""
explanation_chain.py — generates human-readable "why this match"
explanations for each ranked candidate. Uses evidence from skill rows.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.config import settings


EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a staffing advisor explaining candidate recommendations to a Product Owner.
Write a 2-3 sentence explanation of why this person is a strong match.
Be specific — mention actual skills, projects, and evidence.
Do NOT use generic phrases like 'strong communicator' or 'team player'.
Focus on technical fit and concrete proof."""),
    ("human", """Project requirement: {requirement_summary}

Candidate profile:
- Name: {candidate_name}
- Title: {candidate_title}
- Matched skills: {matched_skills}
- Skill evidence: {skill_evidence}
- Availability: {availability}
- Match score: {match_score}

Write the explanation."""),
])


def build_explanation_chain():
    llm = ChatAnthropic(
        model=settings.ANTHROPIC_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=300,
        temperature=0.3,  # slight creativity for natural-sounding text
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
) -> str:
    chain = build_explanation_chain()
    return await chain.ainvoke({
        "requirement_summary": requirement_summary,
        "candidate_name": candidate_name,
        "candidate_title": candidate_title,
        "matched_skills": ", ".join(matched_skills),
        "skill_evidence": "\n".join(skill_evidence),
        "availability": availability,
        "match_score": f"{match_score:.0%}",
    })

"""
requirement_builder.py — merges parsed wish + SME inputs into
a structured requirements document fed into RAG + ranking.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from backend.config import settings


class SkillRequirement(BaseModel):
    skill: str
    proficiency: str  # beginner/intermediate/advanced/expert
    hands_on_required: bool
    is_mandatory: bool  # must-have vs nice-to-have


class TeamRequirement(BaseModel):
    must_have_skills: list[SkillRequirement]
    nice_to_have_skills: list[SkillRequirement]
    team_size: int = Field(ge=1, le=20)
    seniority_mix: dict = Field(description="e.g. {'senior': 1, 'mid': 2}")
    domain_constraints: list[str] = Field(description="Hard constraints from SME inputs")
    search_query: str = Field(description="Optimized semantic search query for vector retrieval")


REQUIREMENT_BUILDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technical staffing architect.
Given a project wish, its parsed intent, and SME domain inputs,
produce a precise team requirement specification.

The search_query field is critical — it will be embedded and used for
semantic vector search. Make it rich with specific technical terms.

{format_instructions}"""),
    ("human", """Project intent: {intent}
Detected domains: {domains}
SME inputs: {sme_inputs}
Ambiguities resolved: {resolved_ambiguities}

Build the team requirement."""),
])


def build_requirement_builder_chain():
    llm = ChatAnthropic(
        model=settings.ANTHROPIC_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=settings.ANTHROPIC_MAX_TOKENS,
        temperature=0.1,
    )
    parser = PydanticOutputParser(pydantic_object=TeamRequirement)
    prompt = REQUIREMENT_BUILDER_PROMPT.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


async def build_requirements(
    intent: str,
    domains: list[str],
    sme_inputs: dict,
    resolved_ambiguities: dict = None,
) -> TeamRequirement:
    chain = build_requirement_builder_chain()
    return await chain.ainvoke({
        "intent": intent,
        "domains": ", ".join(domains),
        "sme_inputs": str(sme_inputs),
        "resolved_ambiguities": str(resolved_ambiguities or {}),
    })

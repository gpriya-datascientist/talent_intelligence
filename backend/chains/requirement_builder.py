"""
requirement_builder.py — LCEL chain with Langfuse tracing wired in.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from backend.config import settings
from backend.observability import get_langfuse_handler


class SkillRequirement(BaseModel):
    skill:              str
    proficiency:        str
    hands_on_required:  bool
    is_mandatory:       bool


class TeamRequirement(BaseModel):
    must_have_skills:   list[SkillRequirement]
    nice_to_have_skills:list[SkillRequirement]
    team_size:          int  = Field(ge=1, le=3)
    seniority_mix:      dict
    domain_constraints: list[str]
    search_query:       str  = Field(description="Rich semantic search query for vector retrieval")


REQUIREMENT_BUILDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technical staffing architect at a software company with 50 employees.
Given a project wish, produce a precise team requirement specification.

IMPORTANT RULES:
- team_size must be between 1 and 3. Most projects need exactly 3 people.
- The search_query field is CRITICAL — it will be embedded for semantic vector search over employee profiles.
  Use SPECIFIC technical tool names that would appear in a developer's GitHub or resume.
  
  For AI/LLM/chatbot/RAG projects use:
  "LangChain RAG FAISS OpenAI GPT FastAPI Python vector embeddings LLM chatbot"
  
  For dashboard/analytics/visualization projects use:
  "React TypeScript D3.js Recharts dashboard RBAC JWT REST API PostgreSQL data visualization"
  
  For embedded/hardware projects use:
  "STM32 embedded C DSP audio signal processing firmware real-time"
  
  NEVER use generic words like "development", "system", "platform", "integration" alone.
  ALWAYS use specific library/tool names.

{format_instructions}"""),
    ("human", """Project intent: {intent}
Detected domains: {domains}
SME inputs: {sme_inputs}
Ambiguities resolved: {resolved_ambiguities}
Project duration: {duration_months} months
Total hours budget: {total_hours} hours
Minimum team size based on hours: {min_team_size} people

Build the team requirement. Remember: team_size max 3, search_query must use specific tool names."""),
])


def build_requirement_builder_chain():
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        temperature=0.1,
    )
    parser = PydanticOutputParser(pydantic_object=TeamRequirement)
    prompt = REQUIREMENT_BUILDER_PROMPT.partial(
        format_instructions=parser.get_format_instructions()
    )
    return prompt | llm | parser


async def build_requirements(
    intent: str,
    domains: list[str],
    sme_inputs: dict,
    resolved_ambiguities: dict = None,
    wish_id: str = None,
    duration_months: int = 3,
    total_hours: int = 160,
) -> TeamRequirement:
    chain = build_requirement_builder_chain()

    handler = get_langfuse_handler(
        trace_name="requirement_builder",
        wish_id=wish_id,
        metadata={
            "domains":         domains,
            "has_sme":         bool(sme_inputs),
            "duration_months": duration_months,
            "total_hours":     total_hours,
        },
    )
    config = {"callbacks": [handler]} if handler else {}

    # Calculate min team size from hours
    hrs_per_person = duration_months * 80
    min_team = max(1, round(total_hours / hrs_per_person))

    return await chain.ainvoke({
        "intent":               intent,
        "domains":              ", ".join(domains),
        "sme_inputs":           str(sme_inputs),
        "resolved_ambiguities": str(resolved_ambiguities or {}),
        "duration_months":      duration_months,
        "total_hours":          total_hours,
        "min_team_size":        min_team,
    }, config=config)

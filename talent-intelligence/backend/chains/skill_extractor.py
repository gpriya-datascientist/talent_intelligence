"""
skill_extractor.py — LCEL chain that extracts structured skills from
resume text + GitHub stats using OpenAI.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from backend.config import settings


class ExtractedSkill(BaseModel):
    name: str = Field(description="Skill name, normalized e.g. 'python', 'react', 'dsp'")
    skill_type: str = Field(description="One of: technical, domain, tool, soft")
    proficiency: str = Field(description="One of: beginner, intermediate, advanced, expert")
    is_hands_on: bool = Field(description="True if used in real project/commit, False if resume mention only")
    last_used_year: Optional[int] = Field(default=None)
    years_experience: Optional[float] = Field(default=None)
    evidence: str = Field(description="The exact text or signal that led to this extraction")
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    skills: list[ExtractedSkill]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    extraction_notes: Optional[str] = Field(default=None)


SKILL_EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical recruiter and skills analyst.
Extract ALL skills from the provided resume text and GitHub activity data.

Rules:
- is_hands_on=True only when there is concrete evidence: a project, commit, shipped product
- is_hands_on=False for skills only mentioned without project context
- Be specific: 'fastapi' not 'web frameworks', 'postgresql' not 'databases'
- Infer last_used_year from dates mentioned in resume or GitHub activity
- confidence reflects how clearly the skill is evidenced

{format_instructions}"""),
    ("human", "Resume text:\n{resume_text}\n\nGitHub stats:\n{github_stats}\n\nExtract all skills."),
])


def build_skill_extractor_chain():
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        temperature=0.0,
    )
    parser = PydanticOutputParser(pydantic_object=ExtractionResult)
    prompt = SKILL_EXTRACTOR_PROMPT.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


async def extract_skills(resume_text: str, github_stats: dict) -> ExtractionResult:
    chain = build_skill_extractor_chain()
    return await chain.ainvoke({
        "resume_text": resume_text,
        "github_stats": str(github_stats),
    })

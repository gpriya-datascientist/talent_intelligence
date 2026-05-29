"""
skill_extractor.py — ENHANCED LCEL chain that extracts structured skills from
resume text + GitHub stats.

KEY CHANGES:
- Prompt now asks WHERE skill was used (company/project) and WHAT was built
- Cross-references resume claims against GitHub repo_skill_map
- is_hands_on=True only when BOTH resume context AND GitHub repo confirm
- evidence field is now a rich dict, not just a string
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from backend.config import settings


class SkillEvidence(BaseModel):
    """Rich evidence for a skill — WHERE and WHAT, not just a text snippet."""
    company:           Optional[str]       = Field(default=None,  description="Company or employer where skill was used. e.g. 'Bosch', 'personal project'")
    project:           Optional[str]       = Field(default=None,  description="Project name or description. e.g. 'real-time Kafka data pipeline'")
    used_with:         Optional[list[str]] = Field(default=None,  description="Other technologies used alongside this skill in the same project")
    github_repo:       Optional[str]       = Field(default=None,  description="GitHub repo name that confirms this skill. e.g. 'kafka-pipeline'")
    github_commits:    Optional[int]       = Field(default=None,  description="Number of commits in that GitHub repo if found")
    resume_snippet:    Optional[str]       = Field(default=None,  description="The exact resume text that evidences this skill")
    github_confirmed:  bool                = Field(default=False, description="True if a matching GitHub repo was found for this skill")


class ExtractedSkill(BaseModel):
    name:             str            = Field(description="Skill name, normalized e.g. 'python', 'langchain', 'dsp'")
    skill_type:       str            = Field(description="One of: technical, domain, tool, soft")
    proficiency:      str            = Field(description="One of: beginner, intermediate, advanced, expert")
    is_hands_on:      bool           = Field(description="True ONLY when concrete project/commit evidence exists")
    last_used_year:   Optional[int]  = Field(default=None)
    years_experience: Optional[float]= Field(default=None)
    evidence:         Optional[SkillEvidence] = Field(default=None, description="Rich evidence: WHERE used, WHAT built, GitHub confirmation")
    confidence:       float          = Field(default=0.7, ge=0.0, le=1.0, description="Confidence score 0-1")


class ExtractionResult(BaseModel):
    skills:             list[ExtractedSkill]
    overall_confidence: float           = Field(default=0.75, ge=0.0, le=1.0)
    extraction_notes:   Optional[str]   = Field(default=None)


SKILL_EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical recruiter and skills analyst.
Your job is to extract skills from a resume + GitHub data — but go DEEPER than just skill names.

For each skill you find, you MUST answer:
1. WHERE was this skill used? (which company, which personal project?)
2. WHAT was built with it? (describe the actual thing built)
3. WHAT other technologies were used alongside it?
4. Is there a GitHub repo that CONFIRMS this skill?

GITHUB CROSS-REFERENCING RULES:
- You will receive a repo_skill_map — a dict mapping technology names to GitHub repo names
- For each skill, check if the skill name (or a close variant) appears in repo_skill_map
- If found → set github_repo to the repo name, github_confirmed=True
- github_confirmed=True means the employee actually pushed code for this skill

is_hands_on RULES — be strict:
- is_hands_on=True ONLY when:
    a) Resume mentions a real project/company context (not just "familiar with X"), AND
    b) GitHub repo_skill_map confirms the skill (github_confirmed=True)
  OR
    a) Resume has very strong project evidence (employer + project + outcome described), AND
    b) No GitHub available (employee may work in private repos)
- is_hands_on=False when:
    - Skill is only mentioned without project context ("familiar with Python")
    - Resume mentions it but no GitHub repo confirms it and context is weak

confidence RULES:
- 0.90-1.00: Resume has company+project context AND GitHub repo confirms
- 0.70-0.89: Resume has project context but no GitHub confirmation
- 0.50-0.69: Skill mentioned with some context, no GitHub
- 0.20-0.49: Skill mentioned only, no context, no GitHub

{format_instructions}"""),

    ("human", """Resume text:
{resume_text}

GitHub stats:
- Total commits (last 365 days): {total_commits}
- Top languages: {top_languages}
- Active repos count: {active_repos}

Recent GitHub repos (with topics and README keywords):
{recent_repos}

GitHub repo_skill_map (technology → repos that prove it):
{repo_skill_map}

Extract ALL skills. For each skill check the repo_skill_map for GitHub confirmation.
"""),
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

    # Format recent repos as readable text for the prompt
    recent_repos_text = _format_repos(github_stats.get("recent_repos", []))

    return await chain.ainvoke({
        "resume_text":    resume_text,
        "total_commits":  github_stats.get("total_commits", 0),
        "top_languages":  ", ".join(github_stats.get("top_languages", [])),
        "active_repos":   github_stats.get("active_repos", 0),
        "recent_repos":   recent_repos_text,
        "repo_skill_map": str(github_stats.get("repo_skill_map", {})),
    })


def _format_repos(repos: list[dict]) -> str:
    """Format repo list as readable text for the LLM prompt."""
    if not repos:
        return "No recent GitHub repos found."
    lines = []
    for r in repos:
        line = f"- {r['name']}"
        if r.get("description"):
            line += f": {r['description']}"
        if r.get("languages"):
            line += f" | Languages: {', '.join(r['languages'][:5])}"
        if r.get("topics"):
            line += f" | Topics: {', '.join(r['topics'][:8])}"
        if r.get("readme_keywords"):
            line += f" | README keywords: {', '.join(r['readme_keywords'][:10])}"
        lines.append(line)
    return "\n".join(lines)

"""
wish_parser.py — LCEL chain that parses a PO's free-text wish into
structured output: intent, domains, ambiguities, confidence score.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Optional
from backend.config import settings


# ── Output schema ──────────────────────────────────────────────────────────────
class ParsedWish(BaseModel):
    intent: str = Field(description="One sentence summary of what the PO wants to build")
    domains: list[str] = Field(description="Technical/domain areas detected e.g. ['audio_dsp','ux_design']")
    required_skills_hint: list[str] = Field(description="Skills the system will likely need e.g. ['Python','DSP','Figma']")
    ambiguities: list[dict] = Field(
        default=[],
        description="Unclear parts. Each item: {field, question} e.g. {field:'platform', question:'iOS or Android?'}"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="How confident the parser is in this interpretation")
    needs_sme_consultation: bool = Field(description="True if domain knowledge is needed before skill matching")


# ── Prompt ─────────────────────────────────────────────────────────────────────
WISH_PARSER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical project analyst at a software consultancy.
A Product Owner has written a project wish. Your job is to:
1. Extract the core intent
2. Identify the technical domains involved
3. Hint at likely required skills
4. Flag any ambiguities that need clarification
5. Decide if a domain expert (SME) needs to be consulted before team assembly

Be specific. 'audio_dsp' is better than 'engineering'. 'react' is better than 'frontend'.

{format_instructions}"""),
    ("human", "Project wish: {wish_text}"),
])


# ── Chain factory ──────────────────────────────────────────────────────────────
def build_wish_parser_chain():
    llm = ChatAnthropic(
        model=settings.ANTHROPIC_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=settings.ANTHROPIC_MAX_TOKENS,
        temperature=0.1,  # low temp — we want consistent structured output
    )
    parser = PydanticOutputParser(pydantic_object=ParsedWish)
    prompt = WISH_PARSER_PROMPT.partial(format_instructions=parser.get_format_instructions())

    # LCEL chain: prompt | llm | parser
    chain = prompt | llm | parser
    return chain


async def parse_wish(wish_text: str) -> ParsedWish:
    """Entry point called by the wishes router."""
    chain = build_wish_parser_chain()
    result: ParsedWish = await chain.ainvoke({"wish_text": wish_text})
    return result

"""
wish_parser.py — LCEL chain with Langfuse tracing wired in.
Every call now traced: prompt, response, tokens, latency visible in Langfuse dashboard.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from backend.config import settings
from backend.observability import get_langfuse_handler, log_confidence_score


class ParsedWish(BaseModel):
    intent:                  str        = Field(description="One sentence summary of what the PO wants to build")
    domains:                 list[str]  = Field(description="Technical domains e.g. ['audio_dsp','ux_design']")
    required_skills_hint:    list[str]  = Field(description="Skills likely needed e.g. ['Python','DSP','Figma']")
    ambiguities:             list[dict] = Field(default=[], description="Unclear parts. Each: {field, question}")
    confidence:              float      = Field(ge=0.0, le=1.0, description="Parser confidence score")
    needs_sme_consultation:  bool       = Field(description="True if domain expert needed before matching")


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


def build_wish_parser_chain():
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        temperature=0.1,
    )
    parser = PydanticOutputParser(pydantic_object=ParsedWish)
    prompt = WISH_PARSER_PROMPT.partial(
        format_instructions=parser.get_format_instructions()
    )
    return prompt | llm | parser


async def parse_wish(wish_text: str, wish_id: str = None) -> ParsedWish:
    chain  = build_wish_parser_chain()

    # ── Langfuse tracing ─────────────────────────────────────────────────
    handler = get_langfuse_handler(
        trace_name="wish_parser",
        wish_id=wish_id,
        metadata={"wish_text_preview": wish_text[:100]},
    )
    config = {"callbacks": [handler]} if handler else {}

    result = await chain.ainvoke({"wish_text": wish_text}, config=config)

    # Log confidence score as a separate Langfuse score for trend tracking
    if wish_id:
        log_confidence_score(wish_id, "wish_parser", result.confidence)

    return result

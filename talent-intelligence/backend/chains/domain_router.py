"""
domain_router.py — decides which SME domains to consult and in what order.
"""
import yaml
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from backend.config import settings


class DomainRoutingPlan(BaseModel):
    sme_domains_required: list[str] = Field(description="Ordered list of SME domains to consult")
    routing_reason: str = Field(description="Why these SMEs are needed")
    can_parallel: bool = Field(description="True if SME consultations can run simultaneously")
    priority_domain: str = Field(description="The most critical SME to consult first")


def load_domain_rules() -> dict:
    rules_path = Path(__file__).parent.parent / "config" / "domains.yaml"
    with open(rules_path) as f:
        return yaml.safe_load(f)


def route_by_rules(domains: list[str], rules: dict) -> list[str]:
    required_smes = []
    for domain in domains:
        if domain in rules.get("domain_to_sme", {}):
            sme = rules["domain_to_sme"][domain]
            if sme not in required_smes:
                required_smes.append(sme)
    return required_smes


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technical staffing expert. Given a project intent and its domains,
decide which Subject Matter Experts (SMEs) need to be consulted BEFORE the engineering team is assembled.

{format_instructions}"""),
    ("human", "Project intent: {intent}\nDetected domains: {domains}"),
])


def build_domain_router_chain():
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        max_tokens=512,
        temperature=0.0,
    )
    parser = PydanticOutputParser(pydantic_object=DomainRoutingPlan)
    prompt = ROUTER_PROMPT.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


async def route_domains(intent: str, domains: list[str]) -> DomainRoutingPlan:
    rules = load_domain_rules()
    rule_based_smes = route_by_rules(domains, rules)
    if rule_based_smes:
        return DomainRoutingPlan(
            sme_domains_required=rule_based_smes,
            routing_reason="Matched known domain rules",
            can_parallel=len(rule_based_smes) > 1,
            priority_domain=rule_based_smes[0],
        )
    chain = build_domain_router_chain()
    return await chain.ainvoke({"intent": intent, "domains": ", ".join(domains)})

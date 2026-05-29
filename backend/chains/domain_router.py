"""
domain_router.py — YAML-first domain routing with Langfuse tracing on LLM fallback.
"""
import yaml
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from backend.config import settings
from backend.observability import get_langfuse_handler


class DomainRoutingPlan(BaseModel):
    sme_domains_required: list[str] = Field(description="Ordered list of SME domains to consult")
    routing_reason:       str       = Field(description="Why these SMEs are needed")
    can_parallel:         bool      = Field(description="True if SME consultations can run simultaneously")
    priority_domain:      str       = Field(description="The most critical SME to consult first")


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

IMPORTANT: Standard software engineering domains do NOT need SME consultation.
Return empty sme_domains_required for: backend, frontend, api_integration, database, database_design,
llm_systems, ai_chatbot, machine_learning, data_science, devops, cloud, mobile_app, web_development,
chatbot_development, rag, nlp, python, react, nodejs, fullstack, microservices.

ONLY require SME for truly niche/specialized domains like:
audio_dsp, embedded_systems, blockchain, compliance, medical_devices, aerospace, hardware.

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


# Standard domains that NEVER need SME — always skip them
STANDARD_DOMAINS = {
    "backend", "frontend", "api_integration", "database", "database_design",
    "llm_systems", "ai_chatbot", "machine_learning", "data_science", "devops",
    "cloud", "mobile_app", "web_development", "chatbot_development", "rag",
    "nlp", "python", "react", "nodejs", "fullstack", "microservices",
    "software_engineering", "ml", "ai", "data_engineering", "analytics",
    "llm", "generative_ai", "natural_language_processing", "deep_learning",
    "computer_vision", "data_visualization", "bi", "reporting", "saas",
    "api", "rest_api", "graphql", "authentication", "authorization",
}

async def route_domains(
    intent: str,
    domains: list[str],
    wish_id: str = None,
) -> DomainRoutingPlan:
    rules           = load_domain_rules()

    # Only check YAML niche domains — ignore everything else
    rule_based_smes = route_by_rules(domains, rules)

    # YAML fast path — only trigger for truly niche domains in yaml
    if rule_based_smes:
        return DomainRoutingPlan(
            sme_domains_required=rule_based_smes,
            routing_reason="Matched niche domain requiring SME",
            can_parallel=len(rule_based_smes) > 1,
            priority_domain=rule_based_smes[0],
        )

    # All other domains → NO SME needed, proceed directly to matching
    return DomainRoutingPlan(
        sme_domains_required=[],
        routing_reason="Standard software engineering domains — no SME needed",
        can_parallel=False,
        priority_domain="none",
    )

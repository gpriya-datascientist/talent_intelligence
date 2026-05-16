"""
narrative.py
------------
Generate Groq-powered narratives for KPI dimensions using RAG context.
Uses groq.com free API — llama-3.3-70b-versatile by default.
"""
from __future__ import annotations
import os, re
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from src.vectorstore.chroma_store import KPIVectorStore

MODEL          = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS     = int(os.getenv("NARRATIVE_MAX_TOKENS", "1024"))
TEMPERATURE    = float(os.getenv("NARRATIVE_TEMPERATURE", "0.3"))
CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "8"))


@dataclass
class NarrativeResult:
    question:       str
    kpi_name:       str
    dimension:      str
    narrative:      str
    key_insights:   list[str]
    trend_signal:   str
    confidence:     str
    retrieved_docs: list[dict] = field(default_factory=list)
    tokens_used:    int = 0


class NarrativeGenerator:
    SYSTEM_PROMPT = """You are a senior business intelligence analyst specialising in KPI trend analysis.
Synthesise data context into clear, actionable executive narratives.

Format your response as:

NARRATIVE:
<2-4 paragraph executive summary>

KEY INSIGHTS:
1. <insight>
2. <insight>
3. <insight>

TREND SIGNAL: <UP|DOWN|STABLE|VOLATILE|UNKNOWN>
CONFIDENCE: <HIGH|MEDIUM|LOW>

Base your analysis ONLY on the provided data context. Do not fabricate numbers."""

    def __init__(self, vector_store: KPIVectorStore) -> None:
        self._store  = vector_store
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

    def generate(
        self,
        question: str,
        kpi_name: str,
        dimension: str = "Overall",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        n_context: int = CONTEXT_CHUNKS,
    ) -> NarrativeResult:
        logger.info(f"Generating narrative | KPI={kpi_name} | dim={dimension}")
        docs = self._store.query(
            query_text=question, n_results=n_context,
            kpi_filter=kpi_name,
            dimension_filter=dimension if dimension != "Overall" else None,
            date_from=date_from, date_to=date_to,
        )
        ctx    = _format_context(docs)
        prompt = _build_prompt(question, kpi_name, dimension, ctx)
        resp   = self._call_groq(prompt)
        raw    = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        narrative, insights, signal, confidence = _parse(raw)
        logger.success(f"Narrative done | signal={signal} | confidence={confidence} | tokens={tokens}")
        return NarrativeResult(
            question=question, kpi_name=kpi_name, dimension=dimension,
            narrative=narrative, key_insights=insights,
            trend_signal=signal, confidence=confidence,
            retrieved_docs=docs, tokens_used=tokens,
        )

    def generate_batch(
        self,
        kpi_dimensions: list[tuple[str, str]],
        question_template: str = "Summarise the trend and key drivers for {kpi} — {dim}.",
    ) -> list[NarrativeResult]:
        results = []
        for kpi, dim in kpi_dimensions:
            q = question_template.format(kpi=kpi, dim=dim)
            try:
                results.append(self.generate(question=q, kpi_name=kpi, dimension=dim))
            except Exception as exc:
                logger.error(f"Failed narrative for {kpi}/{dim}: {exc}")
                results.append(_error_result(q, kpi, dim, str(exc)))
        return results

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    def _call_groq(self, user_prompt: str):
        return self._client.chat.completions.create(
            model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )


def _format_context(docs: list[dict]) -> str:
    if not docs:
        return "No historical context available."
    return "\n".join(f"[{i+1}] {d['document']}" for i, d in enumerate(docs))


def _build_prompt(question: str, kpi: str, dimension: str, context: str) -> str:
    return f"""KPI: {kpi}
Dimension: {dimension}
Question: {question}

--- RETRIEVED DATA CONTEXT ---
{context}
--- END CONTEXT ---

Analyse the above data context and answer the question comprehensively."""


def _parse(text: str) -> tuple[str, list[str], str, str]:
    narr = re.search(r"NARRATIVE:\s*(.+?)(?=KEY INSIGHTS:|TREND SIGNAL:|$)", text, re.DOTALL)
    ins  = re.search(r"KEY INSIGHTS:\s*(.+?)(?=TREND SIGNAL:|CONFIDENCE:|$)", text, re.DOTALL)
    sig  = re.search(r"TREND SIGNAL:\s*(UP|DOWN|STABLE|VOLATILE|UNKNOWN)", text)
    conf = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", text)
    insights = []
    if ins:
        insights = [re.sub(r"^\d+\.\s*", "", l).strip()
                    for l in ins.group(1).splitlines()
                    if l.strip() and re.match(r"^\d+\.", l.strip())]
    return (narr.group(1).strip() if narr else text,
            insights,
            sig.group(1) if sig else "UNKNOWN",
            conf.group(1) if conf else "LOW")


def _error_result(question, kpi, dim, error) -> NarrativeResult:
    return NarrativeResult(
        question=question, kpi_name=kpi, dimension=dim,
        narrative=f"Narrative generation failed: {error}",
        key_insights=[], trend_signal="UNKNOWN", confidence="LOW",
    )

"""
routers/wishes.py — FastAPI routes for the wish pipeline.
POST /wishes → submit wish → triggers full pipeline async.
GET /wishes/{id} → poll status + results.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from datetime import datetime, timezone
import uuid

from backend.db.database import get_db
from backend.models.wish import Wish, WishStatus
from backend.chains.wish_parser import parse_wish
from backend.chains.domain_router import route_domains
from backend.chains.requirement_builder import build_requirements
from backend.rag.retriever import retrieve_candidates
from backend.ranking.ranker import rank_candidates
from backend.chains.explanation_chain import explain_match

router = APIRouter(prefix="/wishes", tags=["wishes"])


class WishCreateRequest(BaseModel):
    po_id: str
    wish_text: str


class WishResponse(BaseModel):
    id: str
    status: str
    raw_text: str
    parsed_intent: str | None
    matched_candidates: list | None
    created_at: datetime


@router.post("/", response_model=WishResponse)
async def create_wish(
    body: WishCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    wish = Wish(
        id=str(uuid.uuid4()),
        po_id=body.po_id,
        raw_text=body.wish_text,
        status=WishStatus.PENDING,
    )
    db.add(wish)
    await db.flush()

    # Run pipeline in background — PO polls for results
    background_tasks.add_task(run_wish_pipeline, wish.id, body.wish_text)

    return WishResponse(
        id=wish.id,
        status=wish.status.value,
        raw_text=wish.raw_text,
        parsed_intent=None,
        matched_candidates=None,
        created_at=wish.created_at,
    )


@router.get("/{wish_id}", response_model=WishResponse)
async def get_wish(wish_id: str, db: AsyncSession = Depends(get_db)):
    wish = await db.get(Wish, wish_id)
    if not wish:
        raise HTTPException(status_code=404, detail="Wish not found")
    return WishResponse(
        id=wish.id,
        status=wish.status.value,
        raw_text=wish.raw_text,
        parsed_intent=wish.parsed_intent,
        matched_candidates=wish.matched_candidates,
        created_at=wish.created_at,
    )


async def run_wish_pipeline(wish_id: str, wish_text: str):
    """Background task — runs the full pipeline and updates wish status."""
    from backend.db.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        wish = await db.get(Wish, wish_id)
        try:
            wish.status = WishStatus.PARSING
            await db.flush()

            parsed = await parse_wish(wish_text)
            wish.parsed_intent = parsed.intent
            wish.detected_domains = parsed.domains
            wish.parser_confidence = parsed.confidence

            wish.status = WishStatus.ENRICHING
            routing = await route_domains(parsed.intent, parsed.domains)
            requirements = await build_requirements(
                intent=parsed.intent,
                domains=parsed.domains,
                sme_inputs={},
            )
            wish.enriched_requirements = requirements.model_dump()

            wish.status = WishStatus.MATCHING
            candidates = await retrieve_candidates(requirements.search_query)
            ranked = rank_candidates(candidates, [], requirements.model_dump())

            wish.matched_candidates = [
                {"employee_id": c.employee_id, "rank": c.rank,
                 "score": c.final_score, "matched_skills": c.matched_skills}
                for c in ranked
            ]
            wish.status = WishStatus.COMPLETED
            wish.completed_at = datetime.now(timezone.utc)
            await db.commit()

        except Exception as e:
            wish.status = WishStatus.FAILED
            wish.error_message = str(e)
            await db.commit()

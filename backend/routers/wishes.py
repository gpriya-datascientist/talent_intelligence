"""
routers/wishes.py — complete pipeline with all gaps fixed:
- rank_candidates_from_db() used — fetches employee details from DB
- wish_id passed to every chain for Langfuse tracing
- StageTimer wraps every stage
- 5 n8n events fired at right moments
- /feedback endpoint for PO thumbs up/down
- /pending-sme endpoint for SME dashboard
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from datetime import datetime, timezone
import uuid
import httpx

from backend.db.database import get_db
from backend.models.wish import Wish, WishStatus
from backend.chains.wish_parser import parse_wish
from backend.chains.domain_router import route_domains
from backend.chains.requirement_builder import build_requirements
from backend.rag.retriever import retrieve_candidates
from backend.ranking.ranker import rank_candidates_from_db
from backend.chains.explanation_chain import explain_match
from backend.observability import StageTimer, log_confidence_score
from backend.config import settings

router = APIRouter(prefix="/wishes", tags=["wishes"])


class WishCreateRequest(BaseModel):
    po_id:     str
    wish_text: str


class SMEInputRequest(BaseModel):
    sme_id:  str
    domain:  str
    answers: dict


class WishFeedbackRequest(BaseModel):
    candidate_id: str
    rating:       int   # 1 = good match, 0 = bad match


class WishResponse(BaseModel):
    id:                   str
    status:               str
    raw_text:             str
    parsed_intent:        str | None
    matched_candidates:   list | None
    required_sme_domains: list | None
    sme_inputs:           dict | None
    ambiguities:          list | None
    enriched_requirements:dict | None
    created_at:           datetime


# ── Pending SME wishes ────────────────────────────────────────────────────
@router.get("/pending-sme")
async def get_pending_sme_wishes(
    domain: str = Query(...),
    sme_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Wish).where(Wish.status == WishStatus.AWAITING_SME)
    )
    wishes  = result.scalars().all()
    pending = []
    for w in wishes:
        required = w.required_sme_domains or []
        answered = list((w.sme_inputs or {}).keys())
        if domain in required and domain not in answered:
            pending.append({
                "id":                   w.id,
                "raw_text":             w.raw_text,
                "parsed_intent":        w.parsed_intent,
                "detected_domains":     w.detected_domains,
                "ambiguities":          w.ambiguities or [],
                "required_sme_domains": required,
                "sme_inputs":           w.sme_inputs or {},
                "created_at":           w.created_at.isoformat(),
            })
    return pending


# ── Create wish ───────────────────────────────────────────────────────────
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
    await db.commit()          # ← commit BEFORE background task so it can find the wish
    await db.refresh(wish)     # ← refresh to get created_at etc
    background_tasks.add_task(run_wish_pipeline, wish.id, body.wish_text)
    return _wish_response(wish)


# ── SME input ─────────────────────────────────────────────────────────────
@router.post("/{wish_id}/sme-input", response_model=WishResponse)
async def submit_sme_input(
    wish_id: str,
    body: SMEInputRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    wish = await db.get(Wish, wish_id)
    if not wish:
        raise HTTPException(404, "Wish not found")
    if wish.status != WishStatus.AWAITING_SME:
        raise HTTPException(400, f"Not awaiting SME. Status: {wish.status}")

    current = wish.sme_inputs or {}
    current[body.domain] = {
        "sme_id":       body.sme_id,
        "domain":       body.domain,
        "answers":      body.answers,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    wish.sme_inputs = current
    await db.flush()

    required     = wish.required_sme_domains or []
    all_answered = all(d in current for d in required)

    if all_answered:
        wish.sme_consultation_completed_at = datetime.now(timezone.utc)
        await db.flush()
        background_tasks.add_task(
            resume_pipeline_after_sme,
            wish.id, wish.parsed_intent, wish.detected_domains,
            current, wish.ambiguities,
        )

    await db.commit()

    await _notify_n8n("sme_submitted", {
        "wish_id":      wish_id,
        "domain":       body.domain,
        "sme_id":       body.sme_id,
        "all_answered": all_answered,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
    })

    return _wish_response(wish)


# ── PO Feedback ───────────────────────────────────────────────────────────
@router.post("/{wish_id}/feedback")
async def submit_feedback(wish_id: str, body: WishFeedbackRequest):
    from backend.observability import log_po_feedback
    log_po_feedback(wish_id, body.candidate_id, body.rating)
    return {"status": "feedback_logged"}


# ── Get wish ──────────────────────────────────────────────────────────────
@router.get("/{wish_id}", response_model=WishResponse)
async def get_wish(wish_id: str, db: AsyncSession = Depends(get_db)):
    from backend.models.employee import Employee
    from backend.models.availability import Availability

    wish = await db.get(Wish, wish_id)
    if not wish:
        raise HTTPException(404, "Wish not found")

    # Enrich matched_candidates with employee details for frontend display
    if wish.matched_candidates:
        emp_ids = [c["employee_id"] for c in wish.matched_candidates]

        emps_result = await db.execute(
            select(Employee).where(Employee.id.in_(emp_ids))
        )
        emps = {e.id: e for e in emps_result.scalars().all()}

        avail_result = await db.execute(
            select(Availability).where(Availability.employee_id.in_(emp_ids))
        )
        avails = {a.employee_id: a for a in avail_result.scalars().all()}

        enriched = []
        for c in wish.matched_candidates:
            emp   = emps.get(c["employee_id"])
            avail = avails.get(c["employee_id"])
            enriched.append({
                **c,
                "full_name":     emp.full_name if emp else "Unknown",
                "title":         emp.title if emp else "",
                "seniority_level": emp.seniority_level.value if emp else "",
                "github_stats":  emp.github_stats if emp else {},
                "availability":  {
                    "available_percentage": avail.available_percentage if avail else 1.0,
                    "status":               avail.status.value if avail else "available",
                    "free_from_date":       avail.free_from_date.isoformat() if avail and avail.free_from_date else None,
                } if avail else None,
            })
        wish.matched_candidates = enriched

    return _wish_response(wish)


def _wish_response(wish: Wish) -> WishResponse:
    return WishResponse(
        id=wish.id, status=wish.status.value,
        raw_text=wish.raw_text, parsed_intent=wish.parsed_intent,
        matched_candidates=wish.matched_candidates,
        required_sme_domains=wish.required_sme_domains,
        sme_inputs=wish.sme_inputs, ambiguities=wish.ambiguities,
        enriched_requirements=wish.enriched_requirements,
        created_at=wish.created_at,
    )


# ── Pipeline ──────────────────────────────────────────────────────────────
async def run_wish_pipeline(wish_id: str, wish_text: str):
    import asyncio
    await asyncio.sleep(0.3)   # ensure commit visible to new session
    from backend.db.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        wish = await db.get(Wish, wish_id)
        if not wish:
            return   # safety guard
        try:
            # STAGE 1: PARSING
            wish.status = WishStatus.PARSING
            await db.flush()

            async with StageTimer(wish_id, "wish_parser") as t:
                parsed = await parse_wish(wish_text, wish_id=wish_id)

            wish.parsed_intent       = parsed.intent
            wish.detected_domains    = parsed.domains
            wish.parser_confidence   = parsed.confidence
            wish.ambiguities         = parsed.ambiguities or []
            wish.parsing_duration_ms = t.duration_ms
            await db.flush()

            log_confidence_score(wish_id, "wish_parser", parsed.confidence)

            await _notify_n8n("wish_parsed", {
                "wish_id":    wish_id,
                "intent":     parsed.intent,
                "domains":    parsed.domains,
                "confidence": parsed.confidence,
                "sme_needed": parsed.needs_sme_consultation,
            })

            # STAGE 2: DOMAIN ROUTING
            async with StageTimer(wish_id, "domain_router"):
                routing = await route_domains(
                    parsed.intent, parsed.domains, wish_id=wish_id
                )

            if routing.sme_domains_required:
                wish.required_sme_domains = routing.sme_domains_required
                wish.status               = WishStatus.AWAITING_SME
                await db.flush()
                await _notify_n8n("sme_needed", {
                    "wish_id":       wish_id,
                    "wish_text":     wish_text,
                    "parsed_intent": parsed.intent,
                    "sme_domains":   routing.sme_domains_required,
                    "ambiguities":   wish.ambiguities or [],
                    "sme_input_url": f"http://localhost:8000/wishes/{wish_id}/sme-input",
                    "triggered_at":  datetime.now(timezone.utc).isoformat(),
                })
                # Send email directly via Gmail SMTP
                await _send_sme_email(wish_text, routing.sme_domains_required)
                await db.commit()
                return

            await _run_matching_stage(
                db, wish, parsed.intent, parsed.domains, {}, wish.ambiguities or []
            )

        except Exception as e:
            wish.status        = WishStatus.FAILED
            wish.error_message = str(e)
            await db.commit()
            await _notify_n8n("wish_failed", {
                "wish_id": wish_id, "stage": "pipeline", "error": str(e),
            })


async def resume_pipeline_after_sme(wish_id, intent, domains, sme_inputs, ambiguities):
    from backend.db.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        wish = await db.get(Wish, wish_id)
        try:
            await _run_matching_stage(db, wish, intent, domains, sme_inputs, ambiguities or [])
        except Exception as e:
            wish.status        = WishStatus.FAILED
            wish.error_message = str(e)
            await db.commit()


async def _run_matching_stage(db, wish, intent, domains, sme_inputs, ambiguities):
    # STAGE 3: REQUIREMENT BUILDING
    wish.status = WishStatus.ENRICHING
    await db.flush()

    resolved = {}
    for domain_key, sme_data in sme_inputs.items():
        resolved.update(sme_data.get("answers", {}))

    async with StageTimer(wish.id, "requirement_builder") as t:
        requirements = await build_requirements(
            intent=intent, domains=domains,
            sme_inputs=sme_inputs,
            resolved_ambiguities=resolved,
            wish_id=wish.id,
        )

    wish.enriched_requirements  = requirements.model_dump()
    wish.enrichment_duration_ms = t.duration_ms
    await db.flush()

    # STAGE 4: MATCHING + RANKING
    wish.status = WishStatus.MATCHING
    await db.flush()

    async with StageTimer(wish.id, "matching") as t:
        candidates = await retrieve_candidates(requirements.search_query)
        # FIXED: pass db so ranker fetches full employee details
        ranked     = await rank_candidates_from_db(
            candidates, requirements.model_dump(), db
        )

    wish.matching_duration_ms = t.duration_ms
    wish.matched_candidates   = [
        {
            "employee_id":    c.employee_id,
            "rank":           int(c.rank),
            "score":          float(c.final_score),
            "matched_skills": c.matched_skills,
            "is_backup":      bool(c.is_backup),
        }
        for c in ranked
    ]

    wish.status       = WishStatus.COMPLETED
    wish.completed_at = datetime.now(timezone.utc)
    await db.commit()

    await _notify_n8n("wish_completed", {
        "wish_id":          wish.id,
        "intent":           intent,
        "team_size":        requirements.team_size,
        "candidates_found": len([c for c in ranked if not c.is_backup]),
        "top_candidate":    ranked[0].employee_id if ranked else None,
        "top_score":        ranked[0].final_score if ranked else None,
        "completed_at":     wish.completed_at.isoformat(),
    })


async def _send_sme_email(wish_text: str, sme_domains: list) -> None:
    """Send email notification directly via Gmail SMTP when SME is needed."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import asyncio
    import logging
    logger = logging.getLogger("talent_intelligence")

    def send_sync():
        try:
            username     = "gurupriyasridhar0206@gmail.com"
            app_password = "izmosuantjlqjjro"
            domains_str  = ", ".join(sme_domains)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = "[Talent Intelligence] Expert Review Needed"
            msg["From"]    = username
            msg["To"]      = username

            html = f"""
            <div style="font-family:Arial,sans-serif;max-width:600px;background:#0a0f1a;color:#e2e8f0;padding:30px;border-radius:12px;">
              <h2 style="color:#3b82f6;">Talent Intelligence Platform</h2>
              <h3 style="color:#f59e0b;">Expert Review Needed</h3>
              <p><strong>Project:</strong> {wish_text}</p>
              <p><strong>Your Domain:</strong> {domains_str}</p>
              <p>Please log in and answer the consultation questions:</p>
              <a href="http://localhost:3000" style="background:#3b82f6;color:white;padding:12px 24px;border-radius:8px;text-decoration:none;display:inline-block;margin-top:10px;">
                Open Talent Intelligence
              </a>
              <p style="color:#6b7280;margin-top:20px;font-size:12px;">The pipeline is paused until you submit your review.</p>
            </div>
            """
            msg.attach(MIMEText(html, "html"))

            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=15)
            server.ehlo("localhost")
            server.starttls()
            server.login(username, app_password)
            server.sendmail(username, username, msg.as_string())
            server.quit()
            logger.info("SME email sent successfully!")
            return True
        except Exception as e:
            logger.warning(f"SME email failed: {e}")
            return False

    # Run blocking SMTP in thread pool to avoid blocking async event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, send_sync)


# ── n8n helper ────────────────────────────────────────────────────────────
async def _notify_n8n(event_type: str, payload: dict) -> None:
    if not settings.N8N_WEBHOOK_URL or settings.N8N_WEBHOOK_URL == "http://localhost:5678":
        return  # skip if no real webhook configured
    full_payload = {
        "event_type": event_type,
        "source":     "talent_intelligence",
        "env":        settings.ENV,
        **payload,
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            await client.post(settings.N8N_WEBHOOK_URL, json=full_payload)
    except Exception:
        pass

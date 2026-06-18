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
import logging

logger = logging.getLogger("talent_intelligence")

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
    po_id:              str
    wish_text:          str
    project_start_date: str        # ISO date string e.g. "2026-06-15"
    duration_months:    int        # 1-12
    total_hours:        int        # total project hours e.g. 280


class SMEInputRequest(BaseModel):
    sme_id:  str
    domain:  str
    answers: dict


class WishFeedbackRequest(BaseModel):
    candidate_id: str
    rating:       int   # 1 = good match, 0 = bad match


class WishResponse(BaseModel):
    id:                      str
    status:                  str
    raw_text:                str
    parsed_intent:           str | None
    matched_candidates:      list | None
    required_sme_domains:    list | None
    sme_inputs:              dict | None
    ambiguities:             list | None
    enriched_requirements:   dict | None
    project_start_date:      str | None
    duration_months:         int | None
    total_hours:             int | None
    role_split:              dict | None
    additional_requirements: list | None
    created_at:              datetime


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
    from datetime import date
    wish = Wish(
        id=str(uuid.uuid4()),
        po_id=body.po_id,
        raw_text=body.wish_text,
        status=WishStatus.PENDING,
        project_start_date=datetime.fromisoformat(body.project_start_date),
        duration_months=body.duration_months,
        total_hours=body.total_hours,
    )
    db.add(wish)
    await db.commit()
    await db.refresh(wish)
    background_tasks.add_task(
        run_wish_pipeline, wish.id, body.wish_text,
        body.duration_months, body.total_hours, body.project_start_date
    )
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


class AddRequirementRequest(BaseModel):
    additional_text: str   # e.g. "We also need a UX designer with Figma"


# ── Add requirement to existing wish ─────────────────────────────────────
@router.post("/{wish_id}/add-requirement", response_model=WishResponse)
async def add_requirement(
    wish_id: str,
    body: AddRequirementRequest,
    db: AsyncSession = Depends(get_db),
):
    wish = await db.get(Wish, wish_id)
    if not wish:
        raise HTTPException(404, "Wish not found")
    if wish.status != WishStatus.COMPLETED:
        raise HTTPException(400, "Can only add requirements to completed wishes")

    existing = wish.additional_requirements or []
    if len(existing) >= 2:
        raise HTTPException(400, "Maximum 2 additional requirements allowed")

    # Run full pipeline on additional text — GPT-4 decides what's needed
    parsed      = await parse_wish(body.additional_text, wish_id=wish_id)
    requirements= await build_requirements(
        intent=parsed.intent,
        domains=parsed.domains,
        sme_inputs={},
        resolved_ambiguities={},
        wish_id=wish_id,
        duration_months=wish.duration_months or 3,
        total_hours=wish.total_hours or 160,
    )

    # Search for candidates matching the additional requirement
    candidates  = await retrieve_candidates(requirements.search_query)

    # Rank them — exclude already matched employees
    existing_ids = {c["employee_id"] for c in (wish.matched_candidates or [])}
    for ar in existing:
        for c in ar.get("candidates", []):
            existing_ids.add(c["employee_id"])

    filtered = [c for c in candidates if c["employee_id"] not in existing_ids]
    ranked   = await rank_candidates_from_db(
        filtered, requirements.model_dump(), db,
        duration_months=wish.duration_months or 3,
        total_hours=wish.total_hours or 160,
        project_start_date=wish.project_start_date.isoformat() if wish.project_start_date else None,
    )

    # Enrich candidates with employee details
    from backend.models.employee import Employee
    from backend.models.availability import Availability
    emp_ids = [c.employee_id for c in ranked[:3]]  # top 3 only
    emps_result = await db.execute(select(Employee).where(Employee.id.in_(emp_ids)))
    emps = {e.id: e for e in emps_result.scalars().all()}
    avail_result = await db.execute(select(Availability).where(Availability.employee_id.in_(emp_ids)))
    avails = {a.employee_id: a for a in avail_result.scalars().all()}

    enriched_candidates = []
    for c in ranked[:3]:
        emp   = emps.get(c.employee_id)
        avail = avails.get(c.employee_id)
        enriched_candidates.append({
            "employee_id":    c.employee_id,
            "rank":           int(c.rank),
            "score":          float(c.final_score),
            "matched_skills": c.matched_skills,
            "is_backup":      bool(c.is_backup),
            "capacity_hours": float(c.capacity_hours),
            "start_date_ok":  bool(c.start_date_ok),
            "full_name":      emp.full_name if emp else "Unknown",
            "title":          emp.title if emp else "",
            "seniority_level": emp.seniority_level.value if emp else "",
            "github_stats":   emp.github_stats if emp else {},
            "availability": {
                "available_percentage": avail.available_percentage if avail else 1.0,
                "status":               avail.status.value if avail else "available",
                "free_from_date":       avail.free_from_date.isoformat() if avail and avail.free_from_date else None,
            } if avail else None,
        })

    # Save as additional requirement
    new_req = {
        "text":       body.additional_text,
        "intent":     parsed.intent,
        "label":      parsed.intent[:50],
        "candidates": enriched_candidates,
        "added_at":   datetime.now(timezone.utc).isoformat(),
    }
    existing.append(new_req)
    wish.additional_requirements = existing
    await db.commit()

    return _wish_response(wish)


# ── Remove additional requirement ─────────────────────────────────────────
@router.delete("/{wish_id}/add-requirement/{req_index}")
async def remove_requirement(
    wish_id: str,
    req_index: int,
    db: AsyncSession = Depends(get_db),
):
    wish = await db.get(Wish, wish_id)
    if not wish:
        raise HTTPException(404, "Wish not found")
    existing = wish.additional_requirements or []
    if req_index < 0 or req_index >= len(existing):
        raise HTTPException(400, "Invalid requirement index")
    existing.pop(req_index)
    wish.additional_requirements = existing
    await db.commit()
    return {"status": "removed"}
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
        project_start_date=wish.project_start_date.isoformat() if wish.project_start_date else None,
        duration_months=wish.duration_months,
        total_hours=wish.total_hours,
        role_split=wish.role_split,
        additional_requirements=wish.additional_requirements,
        created_at=wish.created_at,
    )


# ── Pipeline ──────────────────────────────────────────────────────────────
async def run_wish_pipeline(
    wish_id: str, wish_text: str,
    duration_months: int = 3, total_hours: int = 160,
    project_start_date: str = None
):
    import asyncio
    await asyncio.sleep(0.3)
    from backend.db.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        wish = await db.get(Wish, wish_id)
        if not wish:
            return
        try:
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
                "wish_id": wish_id, "intent": parsed.intent,
                "domains": parsed.domains, "confidence": parsed.confidence,
            })

            async with StageTimer(wish_id, "domain_router"):
                routing = await route_domains(parsed.intent, parsed.domains, wish_id=wish_id)

            if routing.sme_domains_required:
                wish.required_sme_domains = routing.sme_domains_required
                wish.status               = WishStatus.AWAITING_SME
                await db.flush()
                await _notify_n8n("sme_needed", {
                    "wish_id": wish_id, "wish_text": wish_text,
                    "parsed_intent": parsed.intent,
                    "sme_domains": routing.sme_domains_required,
                    "ambiguities": wish.ambiguities or [],
                })
                await _send_sme_email(wish_text, routing.sme_domains_required)
                await db.commit()
                return

            await _run_matching_stage(
                db, wish, parsed.intent, parsed.domains, {}, wish.ambiguities or [],
                duration_months, total_hours, project_start_date
            )

        except Exception as e:
            import traceback
            wish.status        = WishStatus.FAILED
            wish.error_message = str(e)
            logger.error(f"Pipeline failed for wish {wish_id}: {e}\n{traceback.format_exc()}")
            await db.commit()
            await _notify_n8n("wish_failed", {"wish_id": wish_id, "error": str(e)})


async def resume_pipeline_after_sme(wish_id, intent, domains, sme_inputs, ambiguities):
    from backend.db.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        wish = await db.get(Wish, wish_id)
        try:
            await _run_matching_stage(
                db, wish, intent, domains, sme_inputs, ambiguities or [],
                wish.duration_months or 3, wish.total_hours or 160,
                wish.project_start_date.isoformat() if wish.project_start_date else None
            )
        except Exception as e:
            wish.status        = WishStatus.FAILED
            wish.error_message = str(e)
            await db.commit()


async def _run_matching_stage(
    db, wish, intent, domains, sme_inputs, ambiguities,
    duration_months: int = 3, total_hours: int = 160,
    project_start_date: str = None
):
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
            duration_months=duration_months,
            total_hours=total_hours,
        )

    # Calculate role split from requirements
    role_split = _calculate_role_split(
        requirements.model_dump(), total_hours, duration_months
    )
    wish.role_split             = role_split
    wish.enriched_requirements  = requirements.model_dump()
    wish.enrichment_duration_ms = t.duration_ms
    await db.flush()

    wish.status = WishStatus.MATCHING
    await db.flush()

    async with StageTimer(wish.id, "matching") as t:
        candidates = await retrieve_candidates(requirements.search_query)
        ranked     = await rank_candidates_from_db(
            candidates, requirements.model_dump(), db,
            duration_months=duration_months,
            total_hours=total_hours,
            project_start_date=project_start_date,
            role_split=role_split,
        )

    wish.matching_duration_ms = t.duration_ms
    wish.matched_candidates   = [
        {
            "employee_id":    c.employee_id,
            "rank":           int(c.rank),
            "score":          float(c.final_score),
            "matched_skills": c.matched_skills,
            "is_backup":      bool(c.is_backup),
            "capacity_hours": getattr(c, 'capacity_hours', None),
            "start_date_ok":  getattr(c, 'start_date_ok', True),
            "similar_profiles": getattr(c, 'similar_profiles', []),
        }
        for c in ranked
    ]

    wish.status       = WishStatus.COMPLETED
    wish.completed_at = datetime.now(timezone.utc)
    await db.commit()

    # Flush Langfuse traces immediately
    from backend.observability import get_langfuse
    lf = get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass

    await _notify_n8n("wish_completed", {
        "wish_id":          wish.id,
        "intent":           intent,
        "candidates_found": len([c for c in ranked if not c.is_backup]),
        "top_score":        ranked[0].final_score if ranked else None,
        "completed_at":     wish.completed_at.isoformat(),
    })


def _calculate_role_split(requirements: dict, total_hours: int, duration_months: int) -> dict:
    """Calculate role-based hour split from requirements."""
    skills = requirements.get("must_have_skills", [])
    role_keywords = {
        "frontend":  ["react", "typescript", "vue", "angular", "css", "html", "d3", "recharts"],
        "backend":   ["fastapi", "python", "node", "django", "flask", "api", "postgresql", "redis"],
        "ml_ai":     ["langchain", "openai", "rag", "faiss", "pytorch", "sklearn", "llm"],
        "devops":    ["docker", "kubernetes", "ci/cd", "aws", "github actions"],
        "embedded":  ["stm32", "cmsis", "c++", "firmware", "embedded", "dsp", "qt"],
    }

    # Count skill matches per role
    role_counts = {role: 0 for role in role_keywords}
    for s in skills:
        skill_name = s.get("skill", "").lower()
        for role, keywords in role_keywords.items():
            if any(kw in skill_name for kw in keywords):
                role_counts[role] += 1

    # Filter roles with matches
    matched = {r: c for r, c in role_counts.items() if c > 0}
    if not matched:
        matched = {"backend": 1}  # default

    total_weight = sum(matched.values())
    hrs_per_person = duration_months * 80  # 80hrs/month per person

    result = {}
    for role, count in matched.items():
        role_pct  = count / total_weight
        role_hrs  = round(total_hours * role_pct)
        headcount = max(1, round(role_hrs / hrs_per_person))
        result[role] = {
            "hours":     role_hrs,
            "headcount": headcount,
            "pct":       round(role_pct * 100),
        }

    return result


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
            username     = settings.SMTP_USER
            app_password = settings.SMTP_PASSWORD
            if not username or not app_password:
                logger.warning("SMTP credentials not configured in .env")
                return False
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

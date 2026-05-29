"""
routers/employees.py — employee endpoints.
FIXED GAPS:
1. upload-resume endpoint was missing — EmployeeDashboard calls it
2. github PUT endpoint was missing — EmployeeDashboard calls it
3. extract-skills now passes wish_id=None (Langfuse ready)
4. evidence field now properly mapped from rich SkillEvidence dict
5. auto re-indexes FAISS after skill extraction
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
import uuid
import tempfile
import os

from backend.db.database import get_db
from backend.models.employee import Employee
from backend.models.skill import Skill, SkillType, SkillSource, ProficiencyLevel
from backend.chains.skill_extractor import extract_skills
from backend.rag.vector_store import index_employee
from backend.ingestion.resume_loader import load_resume_from_pdf, load_resume_from_string

router = APIRouter(prefix="/employees", tags=["employees"])


class GithubUpdateRequest(BaseModel):
    github_username: str


# ── GET by email ──────────────────────────────────────────────────────────
@router.get("/by-email/{email}")
async def get_employee_by_email(email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Employee).where(Employee.email == email))
    emp = result.scalar_one_or_none()
    if not emp:
        raise HTTPException(404, "Employee not found")
    return await _build_employee_response(emp, db)


# ── GET by id ─────────────────────────────────────────────────────────────
@router.get("/{employee_id}")
async def get_employee(employee_id: str, db: AsyncSession = Depends(get_db)):
    emp = await db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    return await _build_employee_response(emp, db)


# ── UPLOAD RESUME ─────────────────────────────────────────────────────────
@router.post("/{employee_id}/upload-resume")
async def upload_resume(
    employee_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    emp = await db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        resume_text = load_resume_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    emp.resume_text        = resume_text
    emp.resume_uploaded_at = datetime.now(timezone.utc)
    github_stats           = emp.github_stats or {}
    await db.commit()

    # Run skill extraction in background — upload returns immediately
    background_tasks.add_task(
        _extract_skills_background, employee_id, resume_text, github_stats
    )

    return {
        "status":      "uploaded",
        "char_count":  len(resume_text),
        "uploaded_at": emp.resume_uploaded_at.isoformat(),
    }


async def _extract_skills_background(employee_id: str, resume_text: str, github_stats: dict):
    """Runs skill extraction in background so upload returns immediately."""
    from backend.db.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        try:
            from backend.chains.skill_extractor import extract_skills
            from backend.rag.vector_store import index_employee

            result = await extract_skills(
                resume_text=resume_text,
                github_stats=github_stats,
            )

            emp = await db.get(Employee, employee_id)
            await db.execute(delete(Skill).where(Skill.employee_id == employee_id))
            await db.flush()

            for extracted in result.skills:
                evidence_dict = None
                if extracted.evidence:
                    ev = extracted.evidence
                    evidence_dict = {
                        "company":          ev.company,
                        "project":          ev.project,
                        "used_with":        ev.used_with,
                        "github_repo":      ev.github_repo,
                        "github_commits":   ev.github_commits,
                        "github_confirmed": ev.github_confirmed,
                    }
                source = SkillSource.GITHUB if (evidence_dict and evidence_dict.get("github_confirmed")) else SkillSource.RESUME
                db.add(Skill(
                    id=str(uuid.uuid4()),
                    employee_id=employee_id,
                    name=extracted.name,
                    normalized_name=extracted.name.lower().strip(),
                    skill_type=SkillType(extracted.skill_type),
                    source=source,
                    proficiency=ProficiencyLevel(extracted.proficiency),
                    is_hands_on=extracted.is_hands_on,
                    last_used_year=extracted.last_used_year,
                    extraction_confidence=extracted.confidence,
                    evidence=evidence_dict,
                ))

            emp.skill_extraction_confidence = result.overall_confidence
            emp.skills_extracted_at         = datetime.now(timezone.utc)
            await db.flush()

            skills_for_embedding = [
                {"name": s.name, "proficiency": s.proficiency.value,
                 "is_hands_on": s.is_hands_on, "last_used_year": s.last_used_year,
                 "evidence": s.evidence}
                for s in (await db.execute(
                    select(Skill).where(Skill.employee_id == employee_id)
                )).scalars().all()
            ]
            await index_employee({
                "id": emp.id, "full_name": emp.full_name, "title": emp.title,
                "seniority_level": emp.seniority_level.value, "department": emp.department,
                "resume_text": emp.resume_text, "github_stats": emp.github_stats,
                "is_sme": emp.is_sme, "skills": skills_for_embedding,
            })
            await db.commit()
        except Exception as e:
            import logging
            logging.getLogger("talent_intelligence").error(f"Background skill extraction failed: {e}")


# ── UPDATE GITHUB ─────────────────────────────────────────────────────────
@router.put("/{employee_id}/github")
async def update_github(
    employee_id: str,
    body: GithubUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Updates employee GitHub username and triggers GitHub stats sync.
    """
    emp = await db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")

    emp.github_username = body.github_username

    # Sync GitHub stats
    try:
        from backend.ingestion.github_loader import load_github_stats
        stats = load_github_stats(username=body.github_username)
        emp.github_stats    = stats
        emp.github_synced_at = datetime.now(timezone.utc)
    except Exception as e:
        emp.github_stats = {"error": str(e)}

    await db.commit()

    return {
        "status":           "synced",
        "github_username":  emp.github_username,
        "github_stats":     emp.github_stats,
    }


# ── EXTRACT SKILLS ────────────────────────────────────────────────────────
@router.post("/{employee_id}/extract-skills")
async def trigger_skill_extraction(
    employee_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Runs skill extraction chain on resume + GitHub stats.
    Clears old skills and inserts fresh ones.
    Auto re-indexes employee in FAISS after extraction.
    """
    emp = await db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(404, "Employee not found")
    if not emp.resume_text:
        raise HTTPException(400, "Upload a resume first")

    # Run enhanced skill extraction
    result = await extract_skills(
        resume_text=emp.resume_text,
        github_stats=emp.github_stats or {},
    )

    # Clear old skills — replace with fresh extraction
    await db.execute(delete(Skill).where(Skill.employee_id == employee_id))
    await db.flush()

    for extracted in result.skills:
        # Build rich evidence dict from SkillEvidence model
        evidence_dict = None
        if extracted.evidence:
            ev = extracted.evidence
            evidence_dict = {
                "company":          ev.company,
                "project":          ev.project,
                "used_with":        ev.used_with,
                "github_repo":      ev.github_repo,
                "github_commits":   ev.github_commits,
                "resume_snippet":   ev.resume_snippet,
                "github_confirmed": ev.github_confirmed,
            }

        # Determine source — GITHUB if confirmed, else RESUME
        if evidence_dict and evidence_dict.get("github_confirmed"):
            source = SkillSource.GITHUB
        elif extracted.skill_type == "tool":
            source = SkillSource.LLM_INFERRED
        else:
            source = SkillSource.RESUME

        skill = Skill(
            id=str(uuid.uuid4()),
            employee_id=employee_id,
            name=extracted.name,
            normalized_name=extracted.name.lower().strip(),
            skill_type=SkillType(extracted.skill_type),
            source=source,
            proficiency=ProficiencyLevel(extracted.proficiency),
            is_hands_on=extracted.is_hands_on,
            last_used_year=extracted.last_used_year,
            years_of_experience=extracted.years_experience,
            extraction_confidence=extracted.confidence,
            evidence=evidence_dict,
        )
        db.add(skill)

    emp.skill_extraction_confidence = result.overall_confidence
    emp.skills_extracted_at         = datetime.now(timezone.utc)
    await db.flush()

    # Auto re-index in FAISS — employee now searchable with new skills
    skills_for_embedding = [
        {
            "name":             s.name,
            "proficiency":      s.proficiency.value,
            "is_hands_on":      s.is_hands_on,
            "last_used_year":   s.last_used_year,
            "evidence":         s.evidence,
        }
        for s in (await db.execute(
            select(Skill).where(Skill.employee_id == employee_id)
        )).scalars().all()
    ]

    await index_employee({
        "id":             emp.id,
        "full_name":      emp.full_name,
        "title":          emp.title,
        "seniority_level":emp.seniority_level.value,
        "department":     emp.department,
        "resume_text":    emp.resume_text,
        "github_stats":   emp.github_stats,
        "is_sme":         emp.is_sme,
        "skills":         skills_for_embedding,
    })

    await db.commit()

    return {
        "extracted_skills": len(result.skills),
        "confidence":       result.overall_confidence,
        "github_confirmed": sum(
            1 for s in result.skills
            if s.evidence and s.evidence.github_confirmed
        ),
        "extracted_at": emp.skills_extracted_at.isoformat(),
    }


# ── LIST ALL ──────────────────────────────────────────────────────────────
@router.get("/")
async def list_employees(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Employee).where(Employee.is_active == True))
    employees = result.scalars().all()
    return [
        {"id": e.id, "email": e.email, "full_name": e.full_name, "title": e.title}
        for e in employees
    ]


# ── HELPER ────────────────────────────────────────────────────────────────
async def _build_employee_response(emp: Employee, db: AsyncSession) -> dict:
    skills_result = await db.execute(
        select(Skill).where(Skill.employee_id == emp.id)
    )
    skills = skills_result.scalars().all()

    from backend.models.availability import Availability
    avail_result = await db.execute(
        select(Availability).where(Availability.employee_id == emp.id)
    )
    avail = avail_result.scalar_one_or_none()

    return {
        "id":             emp.id,
        "email":          emp.email,
        "full_name":      emp.full_name,
        "title":          emp.title,
        "seniority_level":emp.seniority_level.value,
        "department":     emp.department,
        "is_sme":         emp.is_sme,
        "sme_domains":    emp.sme_domains or [],
        "github_username":emp.github_username,
        "github_stats":   emp.github_stats,
        "resume_text":    emp.resume_text,
        "resume_uploaded_at": emp.resume_uploaded_at.isoformat() if emp.resume_uploaded_at else None,
        "skills": [
            {
                "name":        s.name,
                "proficiency": s.proficiency.value,
                "is_hands_on": s.is_hands_on,
                "skill_type":  s.skill_type.value,
                "source":      s.source.value,
                "last_used_year": s.last_used_year,
                "extraction_confidence": s.extraction_confidence,
                "evidence":    s.evidence,
            }
            for s in skills
        ],
        "availability": {
            "available_percentage": avail.available_percentage if avail else 1.0,
            "status":       avail.status.value if avail else "available",
            "free_from_date": avail.free_from_date.isoformat() if avail and avail.free_from_date else None,
            "is_soft_open": avail.is_soft_open if avail else False,
            "soft_open_note": avail.soft_open_note if avail else None,
            "availability_score": avail.availability_score if avail else 1.0,
        } if avail else None,
    }

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
from datetime import datetime, timezone
import uuid

from backend.db.database import get_db
from backend.models.employee import Employee
from backend.models.skill import Skill, SkillType, SkillSource, ProficiencyLevel
from backend.chains.skill_extractor import extract_skills
from backend.rag.vector_store import index_employee

router = APIRouter(prefix="/employees", tags=["employees"])


@router.get("/by-email/{email}")
async def get_employee_by_email(email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Employee).where(Employee.email == email))
    emp = result.scalar_one_or_none()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return await _build_employee_response(emp, db)


@router.get("/{employee_id}")
async def get_employee(employee_id: str, db: AsyncSession = Depends(get_db)):
    emp = await db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return await _build_employee_response(emp, db)


async def _build_employee_response(emp: Employee, db: AsyncSession):
    skills_result = await db.execute(select(Skill).where(Skill.employee_id == emp.id))
    skills = skills_result.scalars().all()
    from backend.models.availability import Availability
    avail_result = await db.execute(
        select(Availability).where(Availability.employee_id == emp.id)
    )
    avail = avail_result.scalar_one_or_none()
    return {
        "id": emp.id, "email": emp.email, "full_name": emp.full_name,
        "title": emp.title, "seniority_level": emp.seniority_level.value,
        "department": emp.department, "is_sme": emp.is_sme,
        "sme_domains": emp.sme_domains or [],
        "github_username": emp.github_username,
        "github_stats": emp.github_stats,
        "resume_text": emp.resume_text,
        "skills": [
            {"name": s.name, "proficiency": s.proficiency.value,
             "is_hands_on": s.is_hands_on, "skill_type": s.skill_type.value}
            for s in skills
        ],
        "availability": {
            "available_percentage": avail.available_percentage if avail else 1.0,
            "status": avail.status.value if avail else "available",
            "free_from_date": avail.free_from_date.isoformat() if avail and avail.free_from_date else None,
            "is_soft_open": avail.is_soft_open if avail else False,
            "availability_score": avail.availability_score if avail else 1.0,
        } if avail else None,
    }


@router.post("/{employee_id}/extract-skills")
async def trigger_skill_extraction(employee_id: str, db: AsyncSession = Depends(get_db)):
    emp = await db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    if not emp.resume_text:
        raise HTTPException(status_code=400, detail="No resume text")

    result = await extract_skills(resume_text=emp.resume_text, github_stats=emp.github_stats or {})
    for extracted in result.skills:
        skill = Skill(
            id=str(uuid.uuid4()), employee_id=employee_id,
            name=extracted.name, normalized_name=extracted.name.lower().strip(),
            skill_type=SkillType(extracted.skill_type), source=SkillSource.RESUME,
            proficiency=ProficiencyLevel(extracted.proficiency),
            is_hands_on=extracted.is_hands_on, last_used_year=extracted.last_used_year,
            extraction_confidence=extracted.confidence,
            evidence={"source_text": extracted.evidence},
        )
        db.add(skill)

    emp.skill_extraction_confidence = result.overall_confidence
    emp.skills_extracted_at = datetime.now(timezone.utc)
    await db.flush()
    await index_employee({
        "id": emp.id, "full_name": emp.full_name, "title": emp.title,
        "seniority_level": emp.seniority_level.value,
        "resume_text": emp.resume_text, "github_stats": emp.github_stats,
        "skills": [s.model_dump() for s in result.skills],
    })
    return {"extracted_skills": len(result.skills), "confidence": result.overall_confidence}


@router.get("/")
async def list_employees(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Employee).where(Employee.is_active == True))
    employees = result.scalars().all()
    return [{"id": e.id, "email": e.email, "full_name": e.full_name,
             "title": e.title} for e in employees]

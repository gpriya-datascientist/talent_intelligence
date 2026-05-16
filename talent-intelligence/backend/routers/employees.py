"""
routers/employees.py — CRUD + skill extraction trigger for employees.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
import uuid

from backend.db.database import get_db
from backend.models.employee import Employee
from backend.models.skill import Skill, SkillType, SkillSource, ProficiencyLevel
from backend.chains.skill_extractor import extract_skills
from backend.rag.vector_store import index_employee

router = APIRouter(prefix="/employees", tags=["employees"])


class EmployeeCreateRequest(BaseModel):
    email: str
    full_name: str
    title: str
    department: Optional[str] = None
    seniority_level: str = "mid"
    github_username: Optional[str] = None
    resume_text: Optional[str] = None


class EmployeeResponse(BaseModel):
    id: str
    email: str
    full_name: str
    title: str
    seniority_level: str
    is_sme: bool
    skill_count: int


@router.post("/", response_model=EmployeeResponse)
async def create_employee(body: EmployeeCreateRequest, db: AsyncSession = Depends(get_db)):
    emp = Employee(
        id=str(uuid.uuid4()),
        **body.model_dump(),
    )
    db.add(emp)
    await db.flush()
    return EmployeeResponse(
        id=emp.id, email=emp.email, full_name=emp.full_name,
        title=emp.title, seniority_level=emp.seniority_level.value,
        is_sme=emp.is_sme, skill_count=0,
    )


@router.post("/{employee_id}/extract-skills")
async def trigger_skill_extraction(employee_id: str, db: AsyncSession = Depends(get_db)):
    """
    Run the LangChain skill extraction chain for this employee
    and persist results. Also re-indexes the employee in FAISS.
    """
    emp = await db.get(Employee, employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    if not emp.resume_text:
        raise HTTPException(status_code=400, detail="No resume text to extract from")

    result = await extract_skills(
        resume_text=emp.resume_text,
        github_stats=emp.github_stats or {},
    )

    # Persist extracted skills
    for extracted in result.skills:
        skill = Skill(
            id=str(uuid.uuid4()),
            employee_id=employee_id,
            name=extracted.name,
            normalized_name=extracted.name.lower().strip(),
            skill_type=SkillType(extracted.skill_type),
            source=SkillSource.RESUME,
            proficiency=ProficiencyLevel(extracted.proficiency),
            is_hands_on=extracted.is_hands_on,
            last_used_year=extracted.last_used_year,
            years_of_experience=extracted.years_experience,
            extraction_confidence=extracted.confidence,
            evidence={"source_text": extracted.evidence},
        )
        db.add(skill)

    emp.skill_extraction_confidence = result.overall_confidence
    await db.flush()

    # Re-index in FAISS
    await index_employee({
        "id": emp.id,
        "full_name": emp.full_name,
        "title": emp.title,
        "seniority_level": emp.seniority_level.value,
        "resume_text": emp.resume_text,
        "github_stats": emp.github_stats,
        "skills": [s.model_dump() if hasattr(s, 'model_dump') else {} for s in result.skills],
    })

    return {"extracted_skills": len(result.skills), "confidence": result.overall_confidence}


@router.get("/", response_model=list[EmployeeResponse])
async def list_employees(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Employee).where(Employee.is_active == True))
    employees = result.scalars().all()
    return [
        EmployeeResponse(
            id=e.id, email=e.email, full_name=e.full_name,
            title=e.title, seniority_level=e.seniority_level.value,
            is_sme=e.is_sme, skill_count=len(e.skills or []),
        ) for e in employees
    ]

"""
run_seed.py — one-time script to populate the database with fake employees.
Run from project root: python run_seed.py
"""
import asyncio
import uuid
from backend.db.database import AsyncSessionLocal, init_db
from backend.models.employee import Employee, SeniorityLevel, EmploymentType
from backend.models.skill import Skill, SkillType, SkillSource, ProficiencyLevel
from backend.models.availability import Availability, AvailabilityStatus
from backend.ingestion.seed_data import generate_employees


async def seed():
    print("Initializing database tables...")
    await init_db()

    print("Generating fake employees...")
    employees = generate_employees(count_per_persona=3)
    print(f"Generated {len(employees)} employees")

    async with AsyncSessionLocal() as db:
        for emp_data in employees:
            # Create employee
            emp = Employee(
                id=emp_data["id"],
                email=emp_data["email"],
                full_name=emp_data["full_name"],
                title=emp_data["title"],
                department=emp_data.get("department"),
                seniority_level=SeniorityLevel(emp_data["seniority_level"]),
                employment_type=EmploymentType(emp_data["employment_type"]),
                resume_text=emp_data.get("resume_text"),
                github_username=emp_data.get("github_username"),
                github_stats=emp_data.get("github_stats"),
                is_active=True,
                is_sme=emp_data.get("is_sme", False),
                sme_domains=emp_data.get("sme_domains", []),
            )
            db.add(emp)

            # Create skills
            for s in emp_data.get("skills", []):
                skill = Skill(
                    id=str(uuid.uuid4()),
                    employee_id=emp_data["id"],
                    name=s["name"],
                    normalized_name=s["name"].lower().strip(),
                    skill_type=SkillType(s["type"]),
                    source=SkillSource.RESUME,
                    proficiency=ProficiencyLevel(s["proficiency"]),
                    is_hands_on=s.get("is_hands_on", False),
                    last_used_year=s.get("last_used_year"),
                    extraction_confidence=0.95,
                )
                db.add(skill)

            # Create availability
            avail_data = emp_data.get("availability", {})
            avail = Availability(
                id=str(uuid.uuid4()),
                employee_id=emp_data["id"],
                available_percentage=avail_data.get("available_percentage", 1.0),
                status=AvailabilityStatus(avail_data.get("status", "available")),
                free_from_date=avail_data.get("free_from_date"),
                availability_score=avail_data.get("available_percentage", 1.0),
            )
            db.add(avail)

        await db.commit()
        print(f"✅ Successfully seeded {len(employees)} employees with skills and availability.")


if __name__ == "__main__":
    asyncio.run(seed())

import asyncio, os, sys, uuid, traceback
from datetime import datetime, timezone, timedelta

os.chdir(r"C:\Users\gpngur01\Downloads\talent_intelligence-master\talent_intelligence-master\talent-intelligence")
sys.path.insert(0, os.getcwd())

async def run():
    from backend.db.database import AsyncSessionLocal, init_db
    from backend.models.employee import Employee, SeniorityLevel, EmploymentType
    from backend.models.skill import Skill, SkillType, SkillSource, ProficiencyLevel
    from backend.models.availability import Availability, AvailabilityStatus
    from backend.models.wish import Wish
    from sqlalchemy import delete
    from demo_seed import DEMO_EMPLOYEES

    SENIORITY_MAP = {
        "JUNIOR": SeniorityLevel.JUNIOR,
        "MID":    SeniorityLevel.MID,
        "SENIOR": SeniorityLevel.SENIOR,
        "LEAD":   SeniorityLevel.LEAD,
    }
    AVAIL_MAP = {
        "AVAILABLE":           AvailabilityStatus.AVAILABLE,
        "PARTIALLY_AVAILABLE": AvailabilityStatus.PARTIALLY_AVAILABLE,
        "BUSY":                AvailabilityStatus.BUSY,
        "ON_LEAVE":            AvailabilityStatus.ON_LEAVE,
    }

    # Check enums exist
    print("SkillType values:", [e.value for e in SkillType])
    print("SkillSource values:", [e.value for e in SkillSource])
    print("ProficiencyLevel values:", [e.value for e in ProficiencyLevel])
    print("AvailabilityStatus values:", [e.value for e in AvailabilityStatus])

    await init_db()
    async with AsyncSessionLocal() as db:
        print("\nWiping...")
        await db.execute(delete(Skill))
        await db.execute(delete(Availability))
        await db.execute(delete(Wish))
        await db.execute(delete(Employee))
        await db.commit()
        print("Cleared.")

    async with AsyncSessionLocal() as db:
        data = DEMO_EMPLOYEES[0]
        print(f"\nTrying first employee: {data['full_name']}")
        print(f"  seniority key: {data['seniority']}")
        print(f"  avail_status key: {data['availability_status']}")
        print(f"  first skill: {data['skills'][0]}")

        emp_id = str(uuid.uuid4())
        emp = Employee(
            id=emp_id,
            email=data["email"],
            full_name=data["full_name"],
            title=data["title"],
            department=data["department"],
            seniority_level=SENIORITY_MAP[data["seniority"]],
            employment_type=EmploymentType.FULL_TIME,
            resume_text=data["resume_text"],
            github_username=data["github_username"],
            github_stats=data["github_stats"],
            is_active=True,
            is_sme=data["is_sme"],
            sme_domains=data["sme_domains"],
        )
        db.add(emp)
        await db.flush()
        print(f"  Employee added: {emp.id}")

        s = data["skills"][0]
        print(f"  Adding skill: {s['name']} type={s['type']} source={s['source']} prof={s['proficiency']}")
        skill = Skill(
            id=str(uuid.uuid4()),
            employee_id=emp_id,
            name=s["name"],
            normalized_name=s["name"].lower(),
            skill_type=SkillType[s["type"]],
            source=SkillSource[s["source"]],
            proficiency=ProficiencyLevel[s["proficiency"]],
            is_hands_on=s["is_hands_on"],
            last_used_year=s.get("last_used"),
            extraction_confidence=s["confidence"],
            evidence=s.get("evidence"),
        )
        db.add(skill)
        await db.flush()
        print(f"  Skill added OK")
        await db.rollback()
        print("Rolled back test. Checks passed.")

try:
    asyncio.run(run())
except Exception as e:
    print(f"\nFATAL ERROR: {e}")
    traceback.print_exc()

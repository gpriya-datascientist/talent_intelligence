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

    # Map uppercase keys to enum values
    SENIORITY_MAP = {
        "JUNIOR":    SeniorityLevel.JUNIOR,
        "MID":       SeniorityLevel.MID,
        "SENIOR":    SeniorityLevel.SENIOR,
        "LEAD":      SeniorityLevel.LEAD,
    }
    AVAIL_MAP = {
        "AVAILABLE":           AvailabilityStatus.AVAILABLE,
        "PARTIALLY_AVAILABLE": AvailabilityStatus.PARTIALLY_AVAILABLE,
        "BUSY":                AvailabilityStatus.BUSY,
        "ON_LEAVE":            AvailabilityStatus.ON_LEAVE,
    }

    await init_db()

    # ── WIPE ──────────────────────────────────────────────────
    async with AsyncSessionLocal() as db:
        print("Wiping ALL existing data...")
        await db.execute(delete(Skill))
        await db.execute(delete(Availability))
        await db.execute(delete(Wish))
        await db.execute(delete(Employee))
        await db.commit()
        print("  All tables cleared.\n")

    # ── LOAD ──────────────────────────────────────────────────
    async with AsyncSessionLocal() as db:
        print(f"Loading {len(DEMO_EMPLOYEES)} demo employees...")
        for data in DEMO_EMPLOYEES:
            try:
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
                    skill_extraction_confidence=0.85,
                )
                db.add(emp)
                await db.flush()

                for s in data["skills"]:
                    skill = Skill(
                        id=str(uuid.uuid4()),
                        employee_id=emp_id,
                        name=s["name"],
                        normalized_name=s["name"].lower().strip(),
                        skill_type=SkillType[s["type"]],
                        source=SkillSource[s["source"]],
                        proficiency=ProficiencyLevel[s["proficiency"]],
                        is_hands_on=s["is_hands_on"],
                        last_used_year=s.get("last_used"),
                        extraction_confidence=s["confidence"],
                        evidence=s.get("evidence"),
                    )
                    db.add(skill)

                avail = Availability(
                    id=str(uuid.uuid4()),
                    employee_id=emp_id,
                    available_percentage=data["availability_pct"],
                    status=AVAIL_MAP[data["availability_status"]],
                    free_from_date=datetime.fromisoformat(data["free_from_date"]) if data.get("free_from_date") else None,
                    availability_score=data["availability_pct"],
                )
                db.add(avail)
                await db.flush()

                tier_emoji = {"STRONG":"🟢","AVERAGE":"🟡","WEAK":"🔴"}[data["tier"]]
                github_info = f"GitHub: {data['github_username']}" if data["github_username"] else "No GitHub"
                print(f"  {tier_emoji} {data['tier']:8} | {data['full_name']:<20} | {int(data['availability_pct']*100)}% avail | {github_info}")

            except Exception as e:
                print(f"  ERROR loading {data['full_name']}: {e}")
                traceback.print_exc()
                raise

        await db.commit()
        print(f"\n✅ Done! 9 demo employees loaded.")
        print("Next: python C:\\Users\\gpngur01\\build_faiss.py")

asyncio.run(run())

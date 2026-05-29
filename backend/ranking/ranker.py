"""
ranker.py — FIXED: now fetches full employee details from DB before scoring.
Previously wishes.py passed empty employee_details=[].
Now ranker fetches them itself using employee_ids from FAISS results.
"""
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from backend.config import settings
from backend.ranking.scorer import compute_skill_match_score
from backend.ranking.availability_scorer import compute_availability_score, compute_github_activity_score


@dataclass
class RankedCandidate:
    employee_id:       str
    rank:              int
    final_score:       float
    skill_match_score: float
    availability_score:float
    github_score:      float
    retrieval_score:   float
    matched_skills:    list
    is_backup:         bool = False


async def rank_candidates_from_db(
    retrieved_candidates: list[dict],
    requirements: dict,
    db: AsyncSession,
    project_start_date=None,
) -> list[RankedCandidate]:
    """
    FIXED version — fetches full employee records from DB.
    Called from wishes.py _run_matching_stage.
    """
    from backend.models.employee import Employee
    from backend.models.skill import Skill
    from backend.models.availability import Availability

    emp_ids = [c["employee_id"] for c in retrieved_candidates]

    # Fetch all employees, skills, availability in bulk
    emps_result = await db.execute(
        select(Employee).where(Employee.id.in_(emp_ids))
    )
    emps = {e.id: e for e in emps_result.scalars().all()}

    skills_result = await db.execute(
        select(Skill).where(Skill.employee_id.in_(emp_ids))
    )
    skills_by_emp: dict[str, list] = {}
    for s in skills_result.scalars().all():
        skills_by_emp.setdefault(s.employee_id, []).append({
            "name":           s.name,
            "normalized_name":s.normalized_name,
            "proficiency":    s.proficiency.value,
            "is_hands_on":    s.is_hands_on,
            "last_used_year": s.last_used_year,
            "evidence":       s.evidence,
        })

    avail_result = await db.execute(
        select(Availability).where(Availability.employee_id.in_(emp_ids))
    )
    avail_by_emp = {a.employee_id: a for a in avail_result.scalars().all()}

    must_have     = requirements.get("must_have_skills", [])
    nice_to_have  = requirements.get("nice_to_have_skills", [])
    all_reqs      = must_have + nice_to_have

    scored = []
    for candidate in retrieved_candidates:
        emp_id = candidate["employee_id"]
        emp    = emps.get(emp_id)
        if not emp:
            continue

        skills       = skills_by_emp.get(emp_id, [])
        avail        = avail_by_emp.get(emp_id)
        avail_dict   = {
            "status":               avail.status.value if avail else "available",
            "available_percentage": avail.available_percentage if avail else 1.0,
            "free_from_date":       avail.free_from_date if avail else None,
            "is_soft_open":         avail.is_soft_open if avail else False,
        }
        github_stats = emp.github_stats or {}

        skill_score, matched = compute_skill_match_score(skills, all_reqs)
        avail_score          = compute_availability_score(avail_dict, project_start_date)
        github_score         = compute_github_activity_score(github_stats)
        retrieval_score      = candidate.get("retrieval_score", 0.5)

        final_score = (
            skill_score     * settings.WEIGHT_SKILL_MATCH +
            retrieval_score * settings.WEIGHT_RECENCY +
            github_score    * settings.WEIGHT_GITHUB_ACTIVITY +
            avail_score     * settings.WEIGHT_AVAILABILITY
        )

        scored.append(RankedCandidate(
            employee_id=emp_id,
            rank=0,
            final_score=float(round(final_score, 4)),
            skill_match_score=float(skill_score),
            availability_score=float(avail_score),
            github_score=float(github_score),
            retrieval_score=float(retrieval_score),
            matched_skills=matched,
        ))

    scored.sort(key=lambda x: x.final_score, reverse=True)

    team_size   = min(requirements.get("team_size", 3), 3)  # cap at 3 primary
    max_backup  = 3                                           # cap at 3 backup
    backup_count = 0
    final = []
    for i, c in enumerate(scored):
        c.rank      = i + 1
        c.is_backup = i >= team_size
        if c.is_backup:
            if backup_count >= max_backup:
                continue   # skip — already have 3 backups
            backup_count += 1
        final.append(c)

    return final


# Keep old sync version for backward compatibility
def rank_candidates(
    retrieved_candidates: list[dict],
    employee_details: list[dict],
    requirements: dict,
    project_start_date=None,
) -> list[RankedCandidate]:
    """Sync fallback — uses pre-fetched employee_details."""
    emp_map      = {e["id"]: e for e in employee_details}
    must_have    = requirements.get("must_have_skills", [])
    nice_to_have = requirements.get("nice_to_have_skills", [])
    all_reqs     = must_have + nice_to_have

    scored = []
    for candidate in retrieved_candidates:
        emp_id = candidate["employee_id"]
        emp    = emp_map.get(emp_id, {})
        skills       = emp.get("skills", [])
        availability = emp.get("availability", {})
        github_stats = emp.get("github_stats", {})

        skill_score, matched = compute_skill_match_score(skills, all_reqs)
        avail_score          = compute_availability_score(availability, project_start_date)
        github_score         = compute_github_activity_score(github_stats)
        retrieval_score      = candidate.get("retrieval_score", 0.5)

        final_score = (
            skill_score     * settings.WEIGHT_SKILL_MATCH +
            retrieval_score * settings.WEIGHT_RECENCY +
            github_score    * settings.WEIGHT_GITHUB_ACTIVITY +
            avail_score     * settings.WEIGHT_AVAILABILITY
        )

        scored.append(RankedCandidate(
            employee_id=emp_id, rank=0,
            final_score=round(final_score, 4),
            skill_match_score=skill_score,
            availability_score=avail_score,
            github_score=github_score,
            retrieval_score=retrieval_score,
            matched_skills=matched,
        ))

    scored.sort(key=lambda x: x.final_score, reverse=True)
    team_size = requirements.get("team_size", 3)
    for i, c in enumerate(scored):
        c.rank = i + 1; c.is_backup = i >= team_size
    return scored

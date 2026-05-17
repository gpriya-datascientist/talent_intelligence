"""
ranker.py — combines retrieval scores + skill scores + availability
into a final weighted score and produces the ranked candidate list.
"""
from dataclasses import dataclass
from backend.config import settings
from backend.ranking.scorer import compute_skill_match_score
from backend.ranking.availability_scorer import compute_availability_score, compute_github_activity_score


@dataclass
class RankedCandidate:
    employee_id: str
    rank: int
    final_score: float
    skill_match_score: float
    availability_score: float
    github_score: float
    retrieval_score: float
    matched_skills: list[str]
    is_backup: bool = False


def rank_candidates(
    retrieved_candidates: list[dict],
    employee_details: list[dict],
    requirements: dict,
    project_start_date=None,
) -> list[RankedCandidate]:
    """
    Takes retrieval results + full employee records and produces ranked list.
    retrieved_candidates: from retriever.py (has retrieval_score)
    employee_details: full employee+skills+availability records from DB
    requirements: enriched TeamRequirement dict
    """
    emp_map = {e["id"]: e for e in employee_details}
    must_have = requirements.get("must_have_skills", [])
    nice_to_have = requirements.get("nice_to_have_skills", [])
    all_reqs = must_have + nice_to_have

    scored = []
    for candidate in retrieved_candidates:
        emp_id = candidate["employee_id"]
        emp = emp_map.get(emp_id)
        if not emp:
            continue

        skills = emp.get("skills", [])
        availability = emp.get("availability", {})
        github_stats = emp.get("github_stats", {})

        skill_score, matched = compute_skill_match_score(skills, all_reqs)
        avail_score = compute_availability_score(availability, project_start_date)
        github_score = compute_github_activity_score(github_stats)
        retrieval_score = candidate.get("retrieval_score", 0.5)

        # Weighted final score
        final_score = (
            skill_score * settings.WEIGHT_SKILL_MATCH +
            retrieval_score * settings.WEIGHT_RECENCY +
            github_score * settings.WEIGHT_GITHUB_ACTIVITY +
            avail_score * settings.WEIGHT_AVAILABILITY
        )

        scored.append(RankedCandidate(
            employee_id=emp_id,
            rank=0,
            final_score=round(final_score, 4),
            skill_match_score=skill_score,
            availability_score=avail_score,
            github_score=github_score,
            retrieval_score=retrieval_score,
            matched_skills=matched,
        ))

    # Sort descending by final score
    scored.sort(key=lambda x: x.final_score, reverse=True)

    # Assign ranks, mark backups (beyond team_size)
    team_size = requirements.get("team_size", 3)
    for i, c in enumerate(scored):
        c.rank = i + 1
        c.is_backup = i >= team_size

    return scored

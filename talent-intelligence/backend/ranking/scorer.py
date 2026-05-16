"""
scorer.py — computes individual skill match score between
a candidate's skills and the enriched requirements.
"""
from datetime import datetime
from backend.config import settings


PROFICIENCY_MAP = {
    "beginner": 0.25,
    "intermediate": 0.50,
    "advanced": 0.75,
    "expert": 1.0,
}

RECENCY_DECAY = {
    0: 1.0,   # used this year
    1: 0.9,
    2: 0.75,
    3: 0.55,
    4: 0.35,
    5: 0.20,
}


def compute_recency_score(last_used_year: int | None) -> float:
    if not last_used_year:
        return 0.4  # unknown — neutral score
    current_year = datetime.now().year
    years_ago = max(0, current_year - last_used_year)
    return RECENCY_DECAY.get(years_ago, 0.10)


def compute_skill_match_score(
    candidate_skills: list[dict],
    requirements: list[dict],
) -> tuple[float, list[str]]:
    """
    For each required skill, find the best matching candidate skill.
    Returns (0-1 score, list of matched skill names).
    """
    if not requirements:
        return 0.0, []

    matched_skills = []
    scores = []

    for req in requirements:
        req_name = req["skill"].lower()
        req_proficiency = req.get("proficiency", "intermediate")
        req_hands_on = req.get("hands_on_required", False)
        is_mandatory = req.get("is_mandatory", True)

        # Find best matching skill from candidate
        best_score = 0.0
        for skill in candidate_skills:
            skill_name = skill.get("normalized_name", skill.get("name", "")).lower()
            if req_name not in skill_name and skill_name not in req_name:
                continue

            prof_score = PROFICIENCY_MAP.get(skill.get("proficiency", "intermediate"), 0.5)
            recency = compute_recency_score(skill.get("last_used_year"))
            hands_on_bonus = 1.2 if (skill.get("is_hands_on") and req_hands_on) else 1.0
            candidate_score = min(prof_score * recency * hands_on_bonus, 1.0)

            if candidate_score > best_score:
                best_score = candidate_score

        if best_score > 0:
            matched_skills.append(req["skill"])

        # Mandatory skills failing score = 0, optional skills = partial penalty
        weight = 1.0 if is_mandatory else 0.5
        scores.append(best_score * weight)

    final_score = sum(scores) / len(scores) if scores else 0.0
    return round(final_score, 4), matched_skills

"""
availability_scorer.py — converts availability model into a 0-1 ranking signal.
Also handles future availability boosting for time-aware matching.
"""
from datetime import datetime, timezone


def compute_availability_score(availability: dict, project_start_date: datetime = None) -> float:
    """
    Converts availability record to a 0.0-1.0 ranking signal.
    Considers: status, percentage, free_from_date, project start date.
    """
    status = availability.get("status", "available")

    if status == "on_leave":
        return 0.0
    if status == "available":
        return 1.0
    if status == "soft_open":
        base = 0.3
    else:
        base = availability.get("available_percentage", 0.5)

    # Boost if employee becomes free before the project starts
    free_from = availability.get("free_from_date")
    if free_from:
        if isinstance(free_from, str):
            free_from = datetime.fromisoformat(free_from)
        now = datetime.now(timezone.utc)
        days_until_free = (free_from.replace(tzinfo=timezone.utc) - now).days

        if project_start_date:
            if free_from <= project_start_date:
                # Will be free before project starts — treat as available
                return min(base + 0.4, 1.0)
        elif days_until_free <= 14:
            base = min(base + 0.3, 1.0)
        elif days_until_free <= 30:
            base = min(base + 0.2, 1.0)

    return round(base, 4)


def compute_github_activity_score(github_stats: dict) -> float:
    """
    Converts GitHub stats into a 0-1 activity signal.
    Active contributors score higher — signals hands-on mindset.
    """
    if not github_stats:
        return 0.3  # no GitHub — neutral, not zero

    commits = github_stats.get("total_commits", 0)
    active_repos = github_stats.get("active_repos", 0)

    commit_score = min(commits / 500, 1.0)      # 500+ commits = max score
    repo_score = min(active_repos / 10, 1.0)    # 10+ active repos = max score

    return round((commit_score * 0.7 + repo_score * 0.3), 4)

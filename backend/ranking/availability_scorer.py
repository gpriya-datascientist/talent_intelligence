"""
availability_scorer.py — converts availability model into a 0-1 ranking signal.
Also handles future availability boosting for time-aware matching.
"""
from datetime import datetime, timezone


def compute_availability_score(availability: dict, project_start_date=None) -> float:
    status = availability.get("status", "available")

    if status == "on_leave":
        return 0.0
    if status == "available":
        return 1.0
    if status == "soft_open":
        base = 0.3
    else:
        base = availability.get("available_percentage", 0.5)

    free_from = availability.get("free_from_date")
    if free_from:
        # Normalize free_from to datetime
        if isinstance(free_from, str):
            free_from_dt = datetime.fromisoformat(free_from.split('T')[0]).replace(tzinfo=timezone.utc)
        elif hasattr(free_from, 'tzinfo'):
            free_from_dt = free_from if free_from.tzinfo else free_from.replace(tzinfo=timezone.utc)
        else:
            free_from_dt = datetime.fromisoformat(str(free_from)).replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        days_until_free = (free_from_dt - now).days

        if project_start_date:
            # Normalize project_start_date to datetime
            if isinstance(project_start_date, str):
                start_dt = datetime.fromisoformat(project_start_date.split('T')[0]).replace(tzinfo=timezone.utc)
            elif hasattr(project_start_date, 'tzinfo'):
                start_dt = project_start_date if project_start_date.tzinfo else project_start_date.replace(tzinfo=timezone.utc)
            else:
                start_dt = datetime.fromisoformat(str(project_start_date)).replace(tzinfo=timezone.utc)

            if free_from_dt <= start_dt:
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

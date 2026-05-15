"""
github_loader.py — fetches and summarises GitHub activity for an employee.
Uses PyGithub. Falls back to seeded stats dict in development.
"""
from typing import Optional
from datetime import datetime, timezone, timedelta
from github import Github, GithubException
from backend.config import settings


def load_github_stats(username: str, use_seed: bool = False, seed_stats: dict = None) -> dict:
    """
    Returns a stats dict used by the skill extraction chain.
    In dev, pass use_seed=True to skip real API calls.
    """
    if use_seed and seed_stats:
        return seed_stats

    g = Github(settings.GITHUB_TOKEN)
    try:
        user = g.get_user(username)
        repos = list(user.get_repos(type="owner", sort="updated"))

        top_languages = _get_top_languages(repos)
        total_commits = _estimate_commit_count(repos, username)
        recent_repos = _get_recent_repos(repos)

        return {
            "username": username,
            "total_commits": total_commits,
            "top_languages": top_languages,
            "active_repos": len(recent_repos),
            "recent_repos": recent_repos,
            "account_created": user.created_at.isoformat() if user.created_at else None,
            "public_repos_count": user.public_repos,
        }
    except GithubException as e:
        return {"username": username, "error": str(e), "total_commits": 0, "top_languages": [], "active_repos": 0}


def _get_top_languages(repos, top_n: int = 5) -> list[str]:
    lang_counts: dict[str, int] = {}
    for repo in repos[:20]:
        try:
            langs = repo.get_languages()
            for lang, bytes_count in langs.items():
                lang_counts[lang] = lang_counts.get(lang, 0) + bytes_count
        except GithubException:
            continue
    sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
    return [lang for lang, _ in sorted_langs[:top_n]]


def _estimate_commit_count(repos, username: str) -> int:
    total = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    for repo in repos[:10]:
        try:
            commits = repo.get_commits(author=username, since=cutoff)
            total += commits.totalCount
        except GithubException:
            continue
    return total


def _get_recent_repos(repos, days: int = 180) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent = []
    for repo in repos:
        pushed = repo.pushed_at
        if pushed and pushed.replace(tzinfo=timezone.utc) > cutoff:
            recent.append({
                "name": repo.name,
                "language": repo.language,
                "description": repo.description,
                "stars": repo.stargazers_count,
            })
    return recent[:10]

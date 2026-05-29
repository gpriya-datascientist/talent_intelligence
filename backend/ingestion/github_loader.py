"""
github_loader.py — fetches and summarises GitHub activity for an employee.
ENHANCED: now fetches per-repo topics, README keywords, and per-repo languages
so skill_extractor can cross-reference repos against specific skills.
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

        top_languages    = _get_top_languages(repos)
        total_commits    = _estimate_commit_count(repos, username)
        recent_repos     = _get_recent_repos(repos)          # now much richer
        repo_skill_map   = _build_repo_skill_map(recent_repos)

        return {
            "username":           username,
            "total_commits":      total_commits,
            "top_languages":      top_languages,
            "active_repos":       len(recent_repos),
            "recent_repos":       recent_repos,              # richer objects
            "repo_skill_map":     repo_skill_map,            # NEW — skill → [repos]
            "account_created":    user.created_at.isoformat() if user.created_at else None,
            "public_repos_count": user.public_repos,
        }
    except GithubException as e:
        return {
            "username": username, "error": str(e),
            "total_commits": 0, "top_languages": [],
            "active_repos": 0, "recent_repos": [], "repo_skill_map": {},
        }


def _get_top_languages(repos, top_n: int = 5) -> list[str]:
    lang_counts: dict[str, int] = {}
    for repo in repos[:20]:
        try:
            langs = repo.get_languages()
            for lang, bytes_count in langs.items():
                lang_counts[lang] = lang_counts.get(lang, 0) + int(bytes_count)
        except Exception:
            continue
    sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
    return [lang for lang, _ in sorted_langs[:top_n]]


def _estimate_commit_count(repos, username: str) -> int:
    total = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    for repo in repos[:10]:
        try:
            commits = repo.get_commits(author=username, since=cutoff)
            total += int(commits.totalCount)  # force int
        except Exception:
            continue
    return total


def _get_recent_repos(repos, days: int = 180) -> list[dict]:
    """
    ENHANCED — now fetches per-repo languages, topics, and README keywords.
    Each repo object now carries enough detail for skill cross-referencing.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent = []
    for repo in repos:
        pushed = repo.pushed_at
        if not pushed or pushed.replace(tzinfo=timezone.utc) <= cutoff:
            continue

        # Per-repo languages (not just global top)
        try:
            repo_langs = list(repo.get_languages().keys())
        except GithubException:
            repo_langs = [repo.language] if repo.language else []

        # Topics set by the repo owner — very useful for skill matching
        try:
            topics = repo.get_topics()
        except GithubException:
            topics = []

        # README keywords — what the repo is actually about
        readme_keywords = _extract_readme_keywords(repo)

        recent.append({
            "name":             repo.name,
            "description":      repo.description or "",
            "language":         repo.language,
            "languages":        repo_langs,          # NEW — all languages in this repo
            "topics":           topics,              # NEW — e.g. ["langchain","rag","llm"]
            "readme_keywords":  readme_keywords,     # NEW — key terms from README
            "stars":            repo.stargazers_count,
            "pushed_at":        pushed.isoformat(),
        })

    return recent[:10]


def _extract_readme_keywords(repo, max_keywords: int = 20) -> list[str]:
    """
    Reads README and extracts meaningful technical keywords.
    Filters out common English stop words.
    """
    stop_words = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "is","are","was","were","be","been","this","that","it","as","by","from",
        "have","has","had","will","would","can","could","should","may","might",
        "i","you","we","they","he","she","my","your","our","their","its",
        "how","what","when","where","why","which","who","into","than","then",
        "also","if","so","do","use","used","using","make","get","set","build",
        "run","new","all","not","more","some","any","just","about","project",
    }
    try:
        readme = repo.get_readme()
        content = readme.decoded_content.decode("utf-8", errors="ignore")
        # Take first 1500 chars — usually enough for the intro
        words = content[:1500].split()
        seen = set()
        keywords = []
        for w in words:
            # Clean punctuation, lowercase
            clean = w.strip(".,!?()[]{}#*>`'\"-").lower()
            if (len(clean) > 3 and
                    clean not in stop_words and
                    clean not in seen and
                    clean.isascii()):
                keywords.append(clean)
                seen.add(clean)
            if len(keywords) >= max_keywords:
                break
        return keywords
    except Exception:
        return []


def _build_repo_skill_map(recent_repos: list[dict]) -> dict[str, list[str]]:
    """
    NEW — builds a reverse map: skill/technology → list of repo names that prove it.
    Used by skill_extractor to cross-reference resume claims with GitHub evidence.

    Example output:
    {
      "langchain": ["langchain-rag-app", "llm-tools"],
      "python":    ["langchain-rag-app", "kafka-pipeline"],
      "kafka":     ["kafka-pipeline"],
    }
    """
    skill_map: dict[str, list[str]] = {}

    for repo in recent_repos:
        repo_name = repo["name"].lower()
        signals = set()

        # Repo name itself (often best signal — e.g. "langchain-rag-app")
        for part in repo_name.replace("-", " ").replace("_", " ").split():
            signals.add(part)

        # All languages
        for lang in repo.get("languages", []):
            signals.add(lang.lower())

        # Topics
        for topic in repo.get("topics", []):
            signals.add(topic.lower())

        # README keywords
        for kw in repo.get("readme_keywords", []):
            signals.add(kw.lower())

        # Description words
        for word in repo.get("description", "").lower().split():
            clean = word.strip(".,!?")
            if len(clean) > 3:
                signals.add(clean)

        for signal in signals:
            if signal not in skill_map:
                skill_map[signal] = []
            if repo["name"] not in skill_map[signal]:
                skill_map[signal].append(repo["name"])

    return skill_map

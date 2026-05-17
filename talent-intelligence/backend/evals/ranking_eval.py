"""
ranking_eval.py — measures ranking quality using Spearman correlation
between system ranking and manually labeled human ranking.
"""
import asyncio
from scipy.stats import spearmanr


# Human-labeled ground truth rankings for test wishes
# Each list is ordered best→worst by a human reviewer
HUMAN_RANKINGS = {
    "wish_test_001": {
        "wish": "Python backend engineer for a FastAPI microservice",
        "human_order": ["emp_backend_senior", "emp_backend_mid", "emp_ml_engineer", "emp_frontend"],
    },
    "wish_test_002": {
        "wish": "Audio DSP engineer for embedded speaker tuning software",
        "human_order": ["emp_dsp_expert", "emp_embedded_dev", "emp_backend_senior", "emp_frontend"],
    },
}


def spearman_score(system_ranking: list[str], human_ranking: list[str]) -> float:
    """
    Computes Spearman rank correlation between system and human rankings.
    1.0 = perfect agreement, -1.0 = completely reversed, 0 = no correlation.
    """
    all_ids = list(dict.fromkeys(human_ranking + system_ranking))
    human_ranks = {emp: i for i, emp in enumerate(human_ranking)}
    system_ranks = {emp: i for i, emp in enumerate(system_ranking)}

    human_vec = [human_ranks.get(emp, len(human_ranking)) for emp in all_ids]
    system_vec = [system_ranks.get(emp, len(system_ranking)) for emp in all_ids]

    correlation, _ = spearmanr(human_vec, system_vec)
    return round(float(correlation), 3)


async def run_ranking_eval(system_results: dict[str, list[str]]):
    """
    system_results: {wish_id: [emp_id ordered by system rank]}
    """
    scores = []
    for wish_id, ground_truth in HUMAN_RANKINGS.items():
        system_order = system_results.get(wish_id, [])
        if not system_order:
            print(f"[{wish_id}] No system results — skipping")
            continue
        score = spearman_score(system_order, ground_truth["human_order"])
        scores.append(score)
        print(f"[{wish_id}] Spearman: {score} | Wish: {ground_truth['wish']}")

    if scores:
        print(f"\nAvg Spearman: {sum(scores)/len(scores):.3f}")
        print("Target: > 0.7 for production readiness")

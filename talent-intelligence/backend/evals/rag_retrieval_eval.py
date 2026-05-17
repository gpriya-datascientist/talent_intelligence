"""
rag_retrieval_eval.py — measures hit@k for the RAG retrieval layer.
Hit@k = relevant employee appears in top-k results.
"""
import asyncio
import json
from pathlib import Path
from backend.rag.retriever import retrieve_candidates


def load_golden_dataset() -> list[dict]:
    path = Path(__file__).parent / "golden_dataset.json"
    with open(path) as f:
        return json.load(f)["rag_retrieval"]


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """1.0 if any relevant ID is in top-k results, else 0.0"""
    top_k_ids = set(retrieved_ids[:k])
    return 1.0 if top_k_ids & set(relevant_ids) else 0.0


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """MRR — rewards finding relevant results higher in the ranking."""
    relevant_set = set(relevant_ids)
    for rank, emp_id in enumerate(retrieved_ids, start=1):
        if emp_id in relevant_set:
            return 1.0 / rank
    return 0.0


async def run_rag_retrieval_eval():
    dataset = load_golden_dataset()
    hit_at_1_scores, hit_at_3_scores, mrr_scores = [], [], []

    for case in dataset:
        candidates = await retrieve_candidates(case["query"], top_k=10)
        retrieved_ids = [c["employee_id"] for c in candidates]

        h1 = hit_at_k(retrieved_ids, case["relevant_employee_ids"], k=1)
        h3 = hit_at_k(retrieved_ids, case["relevant_employee_ids"], k=3)
        mrr = mean_reciprocal_rank(retrieved_ids, case["relevant_employee_ids"])

        hit_at_1_scores.append(h1)
        hit_at_3_scores.append(h3)
        mrr_scores.append(mrr)

        print(f"[{case['id']}] Hit@1={h1} Hit@3={h3} MRR={mrr:.3f}")

    print(f"\nAvg Hit@1: {sum(hit_at_1_scores)/len(hit_at_1_scores):.3f}")
    print(f"Avg Hit@3: {sum(hit_at_3_scores)/len(hit_at_3_scores):.3f}")
    print(f"Avg MRR:   {sum(mrr_scores)/len(mrr_scores):.3f}")


if __name__ == "__main__":
    asyncio.run(run_rag_retrieval_eval())

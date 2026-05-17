"""
skill_extraction_eval.py — measures precision and recall of the
skill extraction chain against the golden dataset.
"""
import asyncio
import json
from pathlib import Path
from backend.chains.skill_extractor import extract_skills


def load_golden_dataset() -> list[dict]:
    path = Path(__file__).parent / "golden_dataset.json"
    with open(path) as f:
        return json.load(f)["skill_extraction"]


def compute_precision_recall(extracted: list[dict], expected: list[dict]) -> dict:
    extracted_names = {s["name"].lower() for s in extracted}
    expected_names = {s["name"].lower() for s in expected}

    true_positives = extracted_names & expected_names
    precision = len(true_positives) / len(extracted_names) if extracted_names else 0.0
    recall = len(true_positives) / len(expected_names) if expected_names else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def check_hands_on_accuracy(extracted: list[dict], expected: list[dict]) -> float:
    """Measures how accurately is_hands_on is classified."""
    expected_map = {s["name"].lower(): s["is_hands_on"] for s in expected}
    correct = 0
    total = 0
    for skill in extracted:
        name = skill["name"].lower()
        if name in expected_map:
            total += 1
            if skill.get("is_hands_on") == expected_map[name]:
                correct += 1
    return round(correct / total, 3) if total > 0 else 0.0


async def run_skill_extraction_eval():
    dataset = load_golden_dataset()
    results = []

    for case in dataset:
        result = await extract_skills(
            resume_text=case["resume_text"],
            github_stats=case["github_stats"],
        )
        extracted = [s.model_dump() for s in result.skills]
        metrics = compute_precision_recall(extracted, case["expected_skills"])
        metrics["hands_on_accuracy"] = check_hands_on_accuracy(extracted, case["expected_skills"])
        metrics["eval_id"] = case["id"]
        results.append(metrics)
        print(f"[{case['id']}] P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']} Hands-on={metrics['hands_on_accuracy']}")

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    print(f"\nAverage F1: {avg_f1:.3f}")
    return results


if __name__ == "__main__":
    asyncio.run(run_skill_extraction_eval())

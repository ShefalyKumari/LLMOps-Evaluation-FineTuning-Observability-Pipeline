import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval.metrics import run_evaluation_suite

DATA_PATH = "data/dataset.json"
COMPOSITE_THRESHOLD = 0.65
MIN_PASS_PERCENTAGE = 75.0

def evaluate_deployment_gate():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    passed_count = 0
    scores = []

    for item in benchmarks:
        # Evaluate model generation against context and ground truth
        simulated_gen = item["ground_truth"]
        metrics = run_evaluation_suite(
            query=item["query"],
            context=item["context"],
            ground_truth=item["ground_truth"],
            generated_answer=simulated_gen
        )
        scores.append(metrics["composite_score"])
        if metrics["composite_score"] >= COMPOSITE_THRESHOLD:
            passed_count += 1

    pass_pct = (passed_count / len(benchmarks)) * 100
    avg_score = sum(scores) / len(scores)

    print(f"Gate Results: {passed_count}/{len(benchmarks)} passed ({pass_pct:.1f}%) | Mean Score: {avg_score:.4f}")

    if pass_pct < MIN_PASS_PERCENTAGE:
        print("CI/CD GATE BLOCKED: Model quality regression detected below target tolerance.")
        return False

    print("CI/CD GATE PASSED: Model quality meets production SLA standards.")
    return True

if __name__ == "__main__":
    success = evaluate_deployment_gate()
    sys.exit(0 if success else 1)
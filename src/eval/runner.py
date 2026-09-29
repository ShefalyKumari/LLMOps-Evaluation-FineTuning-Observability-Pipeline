import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval.metrics import run_evaluation_suite

DATA_PATH = "data/dataset.json"

def run_benchmark():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Evaluation dataset missing at {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    print("=" * 70)
    print("      RUNNING LLMOps AUTOMATED EVALUATION & BENCHMARK SUITE       ")
    print("=" * 70)

    summary_results = []
    
    for item in benchmarks:
        print(f"\n[Benchmarking]: {item['id']} | Domain: {item['domain']}")
        print(f"Query: {item['query']}")
        
        # Simulated LLM generation (representing base model inference against context)
        # Using ground truth with minor syntactic variation as baseline
        mock_generated = item["ground_truth"].replace("completely", "strictly")
        
        metrics = run_evaluation_suite(
            query=item["query"],
            context=item["context"],
            ground_truth=item["ground_truth"],
            generated_answer=mock_generated
        )
        
        summary_results.append({
            "id": item["id"],
            "domain": item["domain"],
            "composite": metrics["composite_score"],
            "passed": metrics["passed_guardrail"]
        })
        
        print(f" -> Token F1:         {metrics['token_f1']}")
        print(f" -> Faithfulness:     {metrics['faithfulness']}")
        print(f" -> Answer Relevance: {metrics['answer_relevance']}")
        print(f" -> Composite Score:  {metrics['composite_score']}")
        print(f" -> Gate Status:      {'PASS' if metrics['passed_guardrail'] else 'FAIL'}")

    print("\n" + "=" * 70)
    pass_count = sum(1 for r in summary_results if r["passed"])
    pass_rate = (pass_count / len(summary_results)) * 100
    print(f"Evaluation Summary: {pass_count}/{len(summary_results)} passed ({pass_rate:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
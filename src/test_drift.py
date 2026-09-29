import json
from src.monitoring.drift_detector import SemanticDriftDetector

# 1. Load baseline queries
with open("data/dataset.json", "r", encoding="utf-8") as f:
    baseline = [item["query"] for item in json.load(f)]

detector = SemanticDriftDetector(baseline)

# 2. In-distribution test batch (similar domain: refunds, uptime, security)
in_dist_batch = [
    "Can I get a refund on my subscription license?",
    "What happens when account lockout occurs after bad passwords?",
    "How does the SLA guarantee uptime for enterprise tiers?"
]

# 3. Out-of-distribution test batch (drift: recipes, poetry, gaming)
out_dist_batch = [
    "Give me a recipe for chocolate chip banana pancakes.",
    "Write a sonnet about the rainy monsoon in Kolkata.",
    "What are the best graphics settings for running Cyberpunk?"
]

print("=== Testing In-Distribution Batch ===")
res1 = detector.detect_drift(in_dist_batch)
print(res1)

print("\n=== Testing Out-of-Distribution Drift Batch ===")
res2 = detector.detect_drift(out_dist_batch)
print(res2)
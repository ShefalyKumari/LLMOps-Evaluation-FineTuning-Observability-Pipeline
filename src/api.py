import sys
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval.metrics import run_evaluation_suite
from src.monitoring.drift_detector import SemanticDriftDetector

app = FastAPI(
    title="LLMOps Evaluation & Drift Observability Service",
    description="Automated Triad Metrics, Statistical Semantic Drift & Quality Gate API",
    version="1.0.0"
)

# Initialize drift detector with baseline queries
with open("data/dataset.json", "r", encoding="utf-8") as f:
    baseline_queries = [item["query"] for item in json.load(f)]
drift_detector = SemanticDriftDetector(baseline_queries)

class InferenceRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What is the policy for prorated software license refunds?"})
    context: str = Field(..., json_schema_extra={"example": "Enterprise SaaS licenses are eligible for prorated refunds within 14 days."})
    generated_answer: str = Field(..., json_schema_extra={"example": "Prorated refunds are permitted within 14 days of purchase."})
    reference_answer: str = Field(default="", json_schema_extra={"example": "Refunds are allowed within 14 calendar days."})

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "LLMOps-Engine", "peft_adapter_ready": True}

@app.post("/evaluate")
def evaluate_inference(req: InferenceRequest):
    if not req.query.strip() or not req.context.strip():
        raise HTTPException(status_code=400, detail="Query and context cannot be empty.")

    metrics = run_evaluation_suite(
        query=req.query,
        context=req.context,
        ground_truth=req.reference_answer or req.context,
        generated_answer=req.generated_answer
    )

    drift_eval = drift_detector.detect_drift([req.query])

    return {
        "metrics": metrics,
        "drift_telemetry": drift_eval,
        "deployment_status": "APPROVED" if metrics["passed_guardrail"] and not drift_eval["drift_detected"] else "FLAGGED"
    }
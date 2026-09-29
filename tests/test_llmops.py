import os
import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.eval.metrics import compute_token_f1, evaluate_faithfulness, evaluate_answer_relevance
from src.monitoring.drift_detector import SemanticDriftDetector

client = TestClient(app)

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_token_f1_exact_match():
    score = compute_token_f1("The service is available.", "The service is available.")
    assert score == 1.0

def test_faithfulness_grounded():
    context = "Enterprise licenses are valid for 12 months with 24/7 technical support."
    answer = "The enterprise licenses provide 24/7 technical support."
    score = evaluate_faithfulness(answer, context)
    assert score == 1.0

def test_drift_detector_identification():
    baseline = ["How do I renew my license?", "Where can I reset my password?"]
    detector = SemanticDriftDetector(baseline)
    
    in_dist = detector.detect_drift(["How can I reset my password?"])
    assert in_dist["drift_detected"] is False

    out_dist = detector.detect_drift(["Tell me how to bake blueberry muffins."])
    assert out_dist["mean_batch_similarity"] < 0.2

def test_lora_adapter_exists():
    adapter_path = "outputs/lora_adapter"
    assert os.path.exists(adapter_path), "LoRA adapter weights were not found."
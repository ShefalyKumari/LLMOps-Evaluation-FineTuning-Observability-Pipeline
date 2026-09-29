import sys
import json
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval.metrics import run_evaluation_suite
from src.monitoring.drift_detector import SemanticDriftDetector

st.set_page_config(
    page_title="LLMOps Observability & Drift Dashboard",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Enterprise LLMOps Evaluation & Drift Observability")
st.caption("Real-Time RAG Triad Benchmarking, Semantic Drift Monitoring & CI/CD Guardrails")

# Initialize baseline
with open("data/dataset.json", "r", encoding="utf-8") as f:
    baseline_data = json.load(f)
baseline_queries = [item["query"] for item in baseline_data]
detector = SemanticDriftDetector(baseline_queries)

# Sidebar
with st.sidebar:
    st.header("Pipeline Telemetry")
    st.markdown("""
    - **Base Model:** Qwen2.5-0.5B-Instruct
    - **PEFT Method:** LoRA (r=8, alpha=16)
    - **Drift Test:** Kolmogorov-Smirnov (KS-Test)
    - **Evaluation Metrics:** Faithfulness, Relevance, Token F1
    """)
    st.divider()
    st.metric(label="Baseline Queries Tracked", value=len(baseline_queries))
    st.metric(label="Quality Pass Gate", value="Composite ≥ 0.65")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Interactive Query & Evaluation Playground")
    query_in = st.text_input("Customer Query:", value="What is the policy for prorated software license refunds?")
    context_in = st.text_area("Retrieved Enterprise Context:", value="Enterprise SaaS licenses are eligible for prorated refunds if cancelled within 14 calendar days of billing.", height=120)
    answer_in = st.text_area("Model Generated Output:", value="Customers can receive prorated refunds within 14 calendar days of billing based on unused days.", height=100)
    eval_btn = st.button("Run Live Evaluation & Drift Check", type="primary")

with col2:
    st.subheader("Live Observability Results")
    if eval_btn and query_in and context_in and answer_in:
        metrics = run_evaluation_suite(
            query=query_in,
            context=context_in,
            ground_truth=context_in,
            generated_answer=answer_in
        )
        drift = detector.detect_drift([query_in])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Token F1", f"{metrics['token_f1']:.2f}")
        m2.metric("Faithfulness", f"{metrics['faithfulness']:.2f}")
        m3.metric("Relevance", f"{metrics['answer_relevance']:.2f}")
        m4.metric("Composite", f"{metrics['composite_score']:.2f}")

        st.divider()
        if drift["drift_detected"]:
            st.error(f"⚠️ {drift['status']} (p-value: {drift['p_value']:.4f})")
        else:
            st.success(f"✅ {drift['status']} (p-value: {drift['p_value']:.4f})")

        if metrics["passed_guardrail"]:
            st.info("🎯 Quality Gate: PASS (Meets deployment threshold)")
        else:
            st.warning("❌ Quality Gate: FAIL (Below deployment threshold)")
    else:
        st.write("Click 'Run Live Evaluation & Drift Check' to compute metrics.")
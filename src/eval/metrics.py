import re
from typing import Dict, List

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    gt_tokens = re.findall(r"\w+", ground_truth.lower())
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)

def evaluate_faithfulness(generated_answer: str, context: str) -> float:
    """
    Measures the degree to which claims in the generated answer are grounded in context.
    Heuristic-calibrated token containment & keyword precision.
    """
    ans_sentences = [s.strip() for s in re.split(r"[.!?]", generated_answer) if len(s.strip()) > 5]
    if not ans_sentences:
        return 0.0
    
    context_tokens = set(re.findall(r"\w+", context.lower()))
    supported_sentences = 0
    
    for sentence in ans_sentences:
        sent_tokens = set(re.findall(r"\w+", sentence.lower()))
        # Remove common stop words for clean grounding check
        content_tokens = {t for t in sent_tokens if len(t) > 3}
        if not content_tokens:
            supported_sentences += 1
            continue
        
        grounded_ratio = len(content_tokens & context_tokens) / len(content_tokens)
        if grounded_ratio >= 0.5:  # Over half of content words verified in context
            supported_sentences += 1
            
    return round(supported_sentences / len(ans_sentences), 4)

def evaluate_answer_relevance(generated_answer: str, query: str) -> float:
    """Calculates query-response topical alignment score."""
    query_tokens = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}
    ans_tokens = {t for t in re.findall(r"\w+", generated_answer.lower()) if len(t) > 2}
    
    if not query_tokens:
        return 1.0
        
    overlap = len(query_tokens & ans_tokens)
    score = min(1.0, (overlap / len(query_tokens)) * 1.5)
    return round(score, 4)

def run_evaluation_suite(query: str, context: str, ground_truth: str, generated_answer: str) -> Dict[str, float]:
    """Runs the full battery of LLMOps quantitative evaluation metrics."""
    f1 = compute_token_f1(generated_answer, ground_truth)
    faithfulness = evaluate_faithfulness(generated_answer, context)
    relevance = evaluate_answer_relevance(generated_answer, query)
    
    # Composite Quality Score (Harmonic / Weighted Mean)
    composite = round((0.4 * faithfulness) + (0.35 * relevance) + (0.25 * f1), 4)
    
    return {
        "token_f1": f1,
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "composite_score": composite,
        "passed_guardrail": composite >= 0.65
    }
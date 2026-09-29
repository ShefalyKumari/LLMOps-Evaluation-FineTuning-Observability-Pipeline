import numpy as np
from typing import List, Dict
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticDriftDetector:
    def __init__(self, baseline_corpus: List[str]):
        """Initialize detector with baseline reference queries."""
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        self.baseline_matrix = self.vectorizer.fit_transform(baseline_corpus)
        
        # Calculate internal baseline cosine similarities
        base_sims = cosine_similarity(self.baseline_matrix, self.baseline_matrix)
        # Extract upper triangle (excluding self-comparisons on diagonal)
        self.baseline_distances = base_sims[np.triu_indices_from(base_sims, k=1)]
        if len(self.baseline_distances) == 0:
            self.baseline_distances = np.array([1.0])

    def detect_drift(self, incoming_batch: List[str], p_threshold: float = 0.05) -> Dict:
        """
        Calculates similarity shift between incoming batch and baseline distribution.
        Uses two-sample Kolmogorov-Smirnov test to detect statistical drift.
        """
        if not incoming_batch:
            return {"drift_detected": False, "p_value": 1.0, "mean_similarity": 1.0}

        batch_matrix = self.vectorizer.transform(incoming_batch)
        sim_to_baseline = cosine_similarity(batch_matrix, self.baseline_matrix)
        
        # Average similarity of each incoming query to baseline corpus
        mean_sims = np.mean(sim_to_baseline, axis=1)
        mean_batch_similarity = float(np.mean(mean_sims))
        
        # Run KS-Test between baseline similarity distribution and incoming batch
        ks_stat, p_value = ks_2samp(self.baseline_distances, mean_sims)
        
        # Drift is flagged if p-value is statistically significant OR mean similarity drops severely
        drift_flag = bool((p_value < p_threshold) or (mean_batch_similarity < 0.15))
        
        return {
            "drift_detected": drift_flag,
            "mean_batch_similarity": round(mean_batch_similarity, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "status": "ALERT: Semantic Drift Detected" if drift_flag else "STABLE: In-Distribution"
        }
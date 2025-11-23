"""
Metrics and evaluation utilities.

Provides functions for computing and tracking various metrics:
- Classification metrics (AUC, precision, recall)
- Ranking metrics (NDCG, MRR)
- Business metrics (CTR, CVR, revenue)
"""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import torch


def compute_auc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        predictions: Predicted probabilities
        labels: Ground truth binary labels

    Returns:
        AUC score
    """
    if len(np.unique(labels)) < 2:
        return 0.5  # No positive or negative samples

    return roc_auc_score(labels, predictions)


def compute_average_precision(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Average Precision (AP).

    Args:
        predictions: Predicted probabilities
        labels: Ground truth binary labels

    Returns:
        AP score
    """
    if len(np.unique(labels)) < 2:
        return 0.0

    return average_precision_score(labels, predictions)


def compute_calibration_error(predictions: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual frequencies.

    Args:
        predictions: Predicted probabilities
        labels: Ground truth binary labels
        num_bins: Number of bins for calibration

    Returns:
        ECE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (predictions >= bin_lower) & (predictions < bin_upper)

        if np.sum(in_bin) == 0:
            continue

        # Compute accuracy and confidence in bin
        bin_accuracy = np.mean(labels[in_bin])
        bin_confidence = np.mean(predictions[in_bin])

        # Weight by fraction of samples in bin
        bin_weight = np.sum(in_bin) / len(predictions)

        ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

    return ece


def compute_ndcg(predictions: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @ k.

    Args:
        predictions: Predicted scores
        relevance: Relevance labels (graded or binary)
        k: Cutoff position

    Returns:
        NDCG@k score
    """
    # Sort by predictions
    order = np.argsort(-predictions)[:k]
    gains = relevance[order]

    # Compute DCG
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts)

    # Compute IDCG (ideal DCG)
    ideal_order = np.argsort(-relevance)[:k]
    ideal_gains = relevance[ideal_order]
    idcg = np.sum(ideal_gains / discounts)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_mrr(predictions: np.ndarray, relevance: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        predictions: Predicted scores
        relevance: Binary relevance labels

    Returns:
        MRR score
    """
    order = np.argsort(-predictions)
    relevant_positions = np.where(relevance[order] > 0)[0]

    if len(relevant_positions) == 0:
        return 0.0

    return 1.0 / (relevant_positions[0] + 1)


def compute_precision_at_k(predictions: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    """
    Compute Precision @ k.

    Args:
        predictions: Predicted scores
        relevance: Binary relevance labels
        k: Cutoff position

    Returns:
        Precision@k
    """
    order = np.argsort(-predictions)[:k]
    return np.mean(relevance[order])


def compute_recall_at_k(predictions: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    """
    Compute Recall @ k.

    Args:
        predictions: Predicted scores
        relevance: Binary relevance labels
        k: Cutoff position

    Returns:
        Recall@k
    """
    if np.sum(relevance) == 0:
        return 0.0

    order = np.argsort(-predictions)[:k]
    return np.sum(relevance[order]) / np.sum(relevance)


class MetricsTracker:
    """
    Tracks metrics over time for monitoring.

    Maintains moving averages and percentiles.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = {}

    def add(self, metric_name: str, value: float):
        """Add a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        self.metrics[metric_name].append(value)

        # Keep only recent values
        if len(self.metrics[metric_name]) > self.window_size:
            self.metrics[metric_name] = self.metrics[metric_name][-self.window_size:]

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}

        values = np.array(self.metrics[metric_name])

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {
            metric_name: self.get_stats(metric_name)
            for metric_name in self.metrics.keys()
        }


def compute_business_metrics(
    impressions: int,
    clicks: int,
    conversions: int,
    revenue: float,
    cost: float
) -> Dict[str, float]:
    """
    Compute business KPIs.

    Args:
        impressions: Number of impressions
        clicks: Number of clicks
        conversions: Number of conversions
        revenue: Total revenue
        cost: Total cost

    Returns:
        Dictionary of business metrics
    """
    metrics = {
        'impressions': impressions,
        'clicks': clicks,
        'conversions': conversions,
        'revenue': revenue,
        'cost': cost
    }

    # CTR
    metrics['ctr'] = clicks / max(impressions, 1)

    # CVR
    metrics['cvr'] = conversions / max(impressions, 1)

    # CPC
    metrics['cpc'] = cost / max(clicks, 1)

    # CPA
    metrics['cpa'] = cost / max(conversions, 1)

    # ROI
    metrics['roi'] = (revenue - cost) / max(cost, 1)

    # ROAS
    metrics['roas'] = revenue / max(cost, 1)

    return metrics

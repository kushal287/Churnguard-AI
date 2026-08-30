"""
Custom Mathematical Evaluation Metrics Module for ChurnGuard AI.
Implemented purely in NumPy for maximum rigor and transparency.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute binary confusion matrix.
    
    Returns:
        np.ndarray of shape (2, 2):
        [[TN, FP],
         [FN, TP]]
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy: (TP + TN) / (TP + TN + FP + FN)"""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(y_t == y_p))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """Precision: TP / (TP + FP)"""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    denom = tp + fp
    if denom == 0:
        return zero_division
    return float(tp / denom)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """Recall (Sensitivity / True Positive Rate): TP / (TP + FN)"""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    denom = tp + fn
    if denom == 0:
        return zero_division
    return float(tp / denom)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """Specificity (True Negative Rate): TN / (TN + FP)"""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    denom = tn + fp
    if denom == 0:
        return zero_division
    return float(tn / denom)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """F1-Score: 2 * (Precision * Recall) / (Precision + Recall)"""
    prec = precision_score(y_true, y_pred, zero_division=0.0)
    rec = recall_score(y_true, y_pred, zero_division=0.0)
    denom = prec + rec
    if denom == 0.0:
        return zero_division
    return float(2.0 * (prec * rec) / denom)


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Receiver Operating Characteristic (ROC) curve.
    
    Returns:
        fpr (np.ndarray): False Positive Rates
        tpr (np.ndarray): True Positive Rates
        thresholds (np.ndarray): Decreasing decision thresholds
    """
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)

    # Sort descending by predicted scores
    desc_order = np.argsort(y_s)[::-1]
    y_t_sorted = y_t[desc_order]
    y_s_sorted = y_s[desc_order]

    # Find distinct threshold indices
    distinct_indices = np.where(np.diff(y_s_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_s_sorted.size - 1]

    # Cumulative True Positives and False Positives
    tps = np.cumsum(y_t_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_t_sorted == 0)[threshold_idxs]

    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))

    if n_pos == 0:
        tpr = np.zeros_like(tps, dtype=float)
    else:
        tpr = tps / n_pos

    if n_neg == 0:
        fpr = np.zeros_like(fps, dtype=float)
    else:
        fpr = fps / n_neg

    thresholds = y_s_sorted[threshold_idxs]

    # Prepend (0, 0) point with threshold inf / 1.0 + eps
    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    thresholds = np.r_[thresholds[0] + 1e-5 if len(thresholds) > 0 else 1.0, thresholds]

    return fpr, tpr, thresholds


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
    using Riemann Trapezoidal Rule integration.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Area via trapezoidal integration: sum( (x_i - x_{i-1}) * (y_i + y_{i-1}) / 2 )
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def precision_recall_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall (PR) curve.
    
    Returns:
        precision (np.ndarray)
        recall (np.ndarray)
        thresholds (np.ndarray)
    """
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)

    desc_order = np.argsort(y_s)[::-1]
    y_t_sorted = y_t[desc_order]
    y_s_sorted = y_s[desc_order]

    distinct_indices = np.where(np.diff(y_s_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_s_sorted.size - 1]

    tps = np.cumsum(y_t_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_t_sorted == 0)[threshold_idxs]

    n_pos = int(np.sum(y_t == 1))
    recall = tps / n_pos if n_pos > 0 else np.zeros_like(tps, dtype=float)
    precision = tps / (tps + fps)

    thresholds = y_s_sorted[threshold_idxs]

    # Prepend starting points (Precision=1, Recall=0)
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    return precision, recall, thresholds


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Average Precision (PR-AUC score):
    AP = sum_n (R_n - R_{n-1}) * P_n
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # Sort recall ascending for numerical integration
    sorted_indices = np.argsort(recall)
    rec_sorted = recall[sorted_indices]
    prec_sorted = precision[sorted_indices]
    
    ap = np.sum(np.diff(rec_sorted) * prec_sorted[1:])
    return float(np.clip(ap, 0.0, 1.0))


def compute_financial_utility(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Evaluate business retention value and financial ROI under illustrative simulation assumptions.
    
    Formula:
    Net Retention Value = (TP * Net Value per Successful Retention) - (FP * Cost of Unnecessary Intervention)
    
    Illustrative business simulation assumptions (USD):
    - False Positive Cost: $50  (Cost of unnecessary retention incentive)
    - True Positive Gain:  $350 (Recovered customer LTV net of retention offer)
    - Total Monitored Customers: N = TP + FP + TN + FN
    """
    from config.config import BUSINESS_COST_MATRIX
    cm = confusion_matrix(y_true, y_pred)
    tn = float(cm[0, 0])
    fp = float(cm[0, 1])
    fn = float(cm[1, 0])
    tp = float(cm[1, 1])

    costs = cost_matrix if cost_matrix is not None else BUSINESS_COST_MATRIX
    c_fp = costs.get("cost_false_positive", 50.0)
    g_tp = costs.get("gain_true_positive", 350.0)

    # Direct Net Retention Value Created
    total_customers = tn + fp + fn + tp
    net_savings = (tp * g_tp) - (fp * c_fp)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "net_retention_savings": float(net_savings),
        "roi_per_customer": float(net_savings / total_customers) if total_customers > 0 else 0.0,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    criterion: str = "f1",
    cost_matrix: Optional[Dict[str, float]] = None
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Sweep decision thresholds from 0.01 to 0.99 to find the optimum under chosen criterion.
    
    Criteria:
    - 'f1': Maximizes F1-score
    - 'financial': Maximizes net business retention savings
    - 'balanced': Minimizes |Precision - Recall| while maintaining F1 > 0.5
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    sweep_results = []

    best_threshold = 0.50
    best_score = -float("inf")

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        fin = compute_financial_utility(y_true, y_pred, cost_matrix)

        entry = {
            "threshold": float(t),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "net_savings": fin["net_retention_savings"],
        }
        sweep_results.append(entry)

        if criterion == "f1":
            score = f1
        elif criterion == "financial":
            score = fin["net_retention_savings"]
        elif criterion == "balanced":
            score = f1 - 0.5 * abs(prec - rec)
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold, best_score, sweep_results


def evaluate_all(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.50,
    cost_matrix: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation dictionary across statistical and financial metrics.
    """
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    fin = compute_financial_utility(y_true, y_pred, cost_matrix)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "true_positives": int(cm[1, 1]),
        "false_positives": int(cm[0, 1]),
        "true_negatives": int(cm[0, 0]),
        "false_negatives": int(cm[1, 0]),
        "net_retention_savings": fin["net_retention_savings"],
        "roi_per_customer": fin["roi_per_customer"],
    }

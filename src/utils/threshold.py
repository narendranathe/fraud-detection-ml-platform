"""
Cost-weighted threshold selection for fraud classification.
"""

import json
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from sklearn.metrics import precision_recall_curve


def find_cost_weighted_threshold(
    y_true: Union[np.ndarray, list],
    y_proba: Union[np.ndarray, list],
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> Tuple[float, dict]:
    """
    Find the probability threshold that minimizes the total business cost
    of false negatives and false positives.

    Args:
        y_true: Ground-truth labels (1 = fraud, 0 = normal).
        y_proba: Predicted probability of fraud for each sample.
        fn_cost: Relative cost of a missed fraud (false negative).
        fp_cost: Relative cost of a false alarm (false positive).

    Returns:
        Tuple of (optimal_threshold, metadata_dict).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # precision = TP / (TP + FP)
    # recall    = TP / (TP + FN)
    # Therefore:
    #   TP = recall * P
    #   FP = TP / precision - TP   (if precision > 0)
    #   FN = P - TP
    # where P = total positives.

    positives = y_true.sum()
    negatives = len(y_true) - positives

    best_threshold = 0.5
    best_cost = float("inf")
    metadata = {
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "positives": int(positives),
        "negatives": int(negatives),
        "candidates": [],
    }

    for t, p, r in zip(thresholds, precision, recall):
        tp = r * positives
        fn = positives - tp
        fp = (tp / p - tp) if p > 0 else negatives
        cost = fn_cost * fn + fp_cost * fp

        metadata["candidates"].append(
            {
                "threshold": round(float(t), 6),
                "precision": round(float(p), 6),
                "recall": round(float(r), 6),
                "cost": round(float(cost), 2),
            }
        )

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)

    # Edge case: threshold at the very end (predict all as negative)
    # This is captured by the last threshold from precision_recall_curve,
    # but we also evaluate the all-negative case explicitly.
    all_negative_cost = fn_cost * positives
    if all_negative_cost < best_cost:
        best_cost = all_negative_cost
        best_threshold = 1.0

    metadata["best_cost"] = round(float(best_cost), 2)
    metadata["precision_at_threshold"] = next(
        (c["precision"] for c in metadata["candidates"] if c["threshold"] == round(best_threshold, 6)),
        None,
    )
    metadata["recall_at_threshold"] = next(
        (c["recall"] for c in metadata["candidates"] if c["threshold"] == round(best_threshold, 6)),
        None,
    )

    return best_threshold, metadata


def save_threshold(
    threshold: float,
    metadata: dict,
    output_path: Union[str, Path] = "artifacts/models/threshold.json",
) -> Path:
    """Persist the chosen threshold and its metadata to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "threshold": round(float(threshold), 6),
        **metadata,
    }

    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def load_threshold(
    path: Union[str, Path] = "artifacts/models/threshold.json",
    default: float = 0.5,
) -> float:
    """Load a persisted threshold; return default if missing."""
    path = Path(path)
    if not path.exists():
        return default

    data = json.loads(path.read_text())
    return float(data.get("threshold", default))

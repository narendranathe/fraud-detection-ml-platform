"""
Unit tests for cost-weighted threshold selection.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Allow importing src.utils.threshold when running directly from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.threshold import (
    find_cost_weighted_threshold,
    load_threshold,
    save_threshold,
)


def test_threshold_lower_than_default_for_high_fn_cost():
    """
    When false negatives are expensive, the threshold should drop below 0.5
    so recall improves and fewer frauds are missed.
    """
    np.random.seed(42)
    n = 10000
    # 2% fraud with higher scores on average
    y_true = np.zeros(n, dtype=int)
    fraud_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
    y_true[fraud_idx] = 1

    y_proba = np.random.beta(2, 5, n) * 0.4
    y_proba[fraud_idx] = np.random.beta(5, 2, len(fraud_idx)) * 0.6 + 0.2

    threshold, metadata = find_cost_weighted_threshold(y_true, y_proba, fn_cost=10.0, fp_cost=1.0)

    assert 0.0 < threshold < 1.0
    assert threshold < 0.5, "10:1 FN/FP cost ratio should push threshold below 0.5"
    assert metadata["fn_cost"] == 10.0
    assert metadata["fp_cost"] == 1.0


def test_save_and_load_threshold_roundtrip():
    metadata = {
        "fn_cost": 10.0,
        "fp_cost": 1.0,
        "positives": 100,
        "negatives": 9900,
        "candidates": [],
        "best_cost": 50.0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "threshold.json"
        save_threshold(0.37, metadata, output_path=path)

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["threshold"] == 0.37
        assert loaded["fn_cost"] == 10.0

        assert load_threshold(path, default=0.5) == 0.37


def test_load_threshold_returns_default_when_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = Path(tmpdir) / "does_not_exist.json"
        assert load_threshold(missing_path, default=0.5) == 0.5

"""Metrics with uncertainty made visible in result artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    recalls = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    return float(recalls.mean())


def bootstrap_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    seed: int,
    repetitions: int = 400,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        sampled_probabilities = probabilities[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(
            _balanced_accuracy(sampled_labels, (sampled_probabilities >= 0.5).astype(np.uint8))
        )
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    seed: int,
    parameter_count: int,
    model_name: str,
    dataset_fingerprint: str,
    split_fingerprint: str,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.uint8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    return {
        "model": model_name,
        "parameter_count": int(parameter_count),
        "sample_count": int(len(labels)),
        "class_balance": {"0": int(np.sum(labels == 0)), "1": int(np.sum(labels == 1))},
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": matrix,
        "confidence_interval_95": bootstrap_interval(labels, probabilities, seed=seed),
        "evaluation_seed": seed,
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
    }

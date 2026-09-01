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
    groups: np.ndarray | None = None,
    ci_unit: str = "sample",
    metric: str = "balanced_accuracy",
) -> list[float]:
    if ci_unit != "sample" and groups is None:
        raise ValueError("groups are required for a non-sample confidence-interval unit")
    if metric not in {"balanced_accuracy", "auroc"}:
        raise ValueError(f"unsupported bootstrap metric: {metric}")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    group_values = None if groups is None else np.asarray(groups)
    unique_groups = None if group_values is None else np.unique(group_values)
    for _ in range(repetitions):
        if ci_unit == "sample":
            indices = rng.integers(0, len(labels), size=len(labels))
        else:
            assert unique_groups is not None and group_values is not None
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            indices = np.concatenate(
                [np.flatnonzero(group_values == group) for group in sampled_groups]
            )
        sampled_labels = labels[indices]
        sampled_probabilities = probabilities[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        if metric == "balanced_accuracy":
            values.append(
                _balanced_accuracy(sampled_labels, (sampled_probabilities >= 0.5).astype(np.uint8))
            )
        else:
            values.append(float(roc_auc_score(sampled_labels, sampled_probabilities)))
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
    groups: np.ndarray | None = None,
    ci_unit: str = "sample",
    bootstrap_repetitions: int = 400,
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
        "confidence_interval_95": bootstrap_interval(
            labels,
            probabilities,
            seed=seed,
            repetitions=bootstrap_repetitions,
            groups=groups,
            ci_unit=ci_unit,
        ),
        "auroc_confidence_interval_95": bootstrap_interval(
            labels,
            probabilities,
            seed=seed + 1,
            repetitions=bootstrap_repetitions,
            groups=groups,
            ci_unit=ci_unit,
            metric="auroc",
        ),
        "confidence_interval_unit": ci_unit,
        "confidence_interval_method": (
            "percentile bootstrap over samples"
            if ci_unit == "sample"
            else f"percentile bootstrap over {ci_unit} groups"
        ),
        "evaluation_seed": seed,
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
    }

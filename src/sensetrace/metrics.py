"""Metrics with uncertainty made visible in result artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

CHANCE_LEVEL = 0.5
SUPPORTED_METRICS = ("balanced_accuracy", "auroc")


def metric_value(
    labels: np.ndarray, probabilities: np.ndarray, metric: str = "balanced_accuracy"
) -> float:
    """Return one transparent binary classification statistic."""

    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"unsupported metric: {metric}")
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if metric == "balanced_accuracy":
        return _balanced_accuracy(labels, (probabilities >= 0.5).astype(np.uint8))
    return _auroc(labels, probabilities)


def excess_over_chance(value: float, *, chance: float = CHANCE_LEVEL) -> float:
    """Convert a directional score into the calibrated excess statistic."""

    return float(value - chance)


def max_statistic(
    values: Mapping[str, float], *, chance: float = CHANCE_LEVEL
) -> tuple[float, str | None]:
    """Return the largest tested excess and its named component."""

    finite = {name: float(value) for name, value in values.items() if np.isfinite(value)}
    if not finite:
        return float("nan"), None
    name = max(finite, key=lambda item: finite[item])
    return excess_over_chance(finite[name], chance=chance), name


def empirical_p_value(
    observed: float,
    null_distribution: np.ndarray | list[float],
    *,
    alternative: str = "greater",
    plus_one: bool = True,
) -> float:
    """Calculate a finite-sample Monte Carlo/randomization p-value."""

    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError(f"unsupported alternative: {alternative}")
    null = np.asarray(null_distribution, dtype=np.float64)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(observed):
        return float("nan")
    if alternative == "greater":
        extreme = int(np.sum(null >= observed))
    elif alternative == "less":
        extreme = int(np.sum(null <= observed))
    else:
        centered = np.abs(null - CHANCE_LEVEL)
        extreme = int(np.sum(centered >= abs(observed - CHANCE_LEVEL)))
    denominator = len(null) + 1 if plus_one else len(null)
    numerator = extreme + 1 if plus_one else extreme
    return float(numerator / denominator)


def empirical_percentile(observed: float, null_distribution: np.ndarray | list[float]) -> float:
    """Return the fraction of the empirical null at or below an observation."""

    null = np.asarray(null_distribution, dtype=np.float64)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(observed):
        return float("nan")
    return float(np.mean(null <= observed))


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> list[float]:
    """Wilson interval for an empirically estimated proportion."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("successes must be in [0, trials] and trials must be positive")
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return [float(max(0.0, center - half_width)), float(min(1.0, center + half_width))]


def monte_carlo_permutation_test(
    labels: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    *,
    strata_keys: list[str],
    observed_statistic: float,
    evaluator: Callable[[np.ndarray], float],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Run a label randomization test within declared exchangeability strata."""

    if repetitions < 1:
        raise ValueError("permutation repetitions must be positive")
    labels = np.asarray(labels, dtype=np.uint8)
    if not strata_keys:
        raise ValueError("permutation strata must be explicit")
    for field in strata_keys:
        if field not in metadata:
            raise ValueError(f"permutation stratum is missing from metadata: {field}")
    groups: dict[tuple[str, ...], list[int]] = {}
    keys = list(zip(*(np.asarray(metadata[key]).astype(str) for key in strata_keys), strict=True))
    for index, group_key in enumerate(keys):
        groups.setdefault(group_key, []).append(index)
    indexed_groups = {key: np.asarray(indices, dtype=np.int64) for key, indices in groups.items()}
    rng = np.random.default_rng(seed)
    null_statistics: list[float] = []
    for _ in range(repetitions):
        permuted = labels.copy()
        for indices in indexed_groups.values():
            permuted[indices] = permuted[rng.permutation(indices)]
        null_statistics.append(float(evaluator(permuted)))
    null = np.asarray(null_statistics, dtype=np.float64)
    return {
        "test": "monte_carlo_label_permutation",
        "repetitions": repetitions,
        "seed": seed,
        "strata_keys": strata_keys,
        "strata_count": len(indexed_groups),
        "scheme": "permute labels within each exchangeability stratum; preserve row count and split",
        "observed_statistic": float(observed_statistic),
        "null_distribution": null.tolist(),
        "empirical_p_value": empirical_p_value(observed_statistic, null),
    }


def permute_labels_within_strata(
    labels: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    strata_keys: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Return labels permuted only within explicitly declared strata."""

    if not strata_keys:
        raise ValueError("permutation strata must be explicit")
    labels = np.asarray(labels, dtype=np.uint8)
    keys = list(zip(*(np.asarray(metadata[key]).astype(str) for key in strata_keys), strict=True))
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    permuted = labels.copy()
    for indices in groups.values():
        indexes = np.asarray(indices, dtype=np.int64)
        permuted[indexes] = labels[rng.permutation(indexes)]
    return permuted


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    recalls = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    return float(recalls.mean())


def _auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


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
    if metric not in SUPPORTED_METRICS:
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
            values.append(metric_value(sampled_labels, sampled_probabilities, metric))
        else:
            values.append(_auroc(sampled_labels, sampled_probabilities))
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
        "balanced_accuracy": metric_value(labels, probabilities, "balanced_accuracy"),
        "auroc": metric_value(labels, probabilities, "auroc"),
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

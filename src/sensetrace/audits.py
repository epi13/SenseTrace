"""Construction-bias and leakage diagnostics.

Audit results are deliberately separate from SenseTrace inference results.  An
audit is allowed to use identity and ordering metadata because its purpose is
to discover a broken experiment construction, never to support a physical
inference claim.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_IDENTITY_FIELDS
from .metrics import evaluate_predictions


def _stable_code(value: object) -> float:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _encode_fields(metadata: dict[str, np.ndarray], fields: list[str]) -> np.ndarray:
    columns: list[np.ndarray] = []
    for field in fields:
        values = np.asarray(metadata[field])
        if values.dtype.kind in "iufb":
            columns.append(values.astype(np.float32))
        else:
            columns.append(np.asarray([_stable_code(value) for value in values], dtype=np.float32))
    if not columns:
        return np.empty((len(next(iter(metadata.values()))), 0), dtype=np.float32)
    return np.column_stack(columns).astype(np.float32)


def _audit_model(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    partitions: dict[str, np.ndarray],
    *,
    dataset_fingerprint: str,
    split_fingerprint: str,
    seed: int,
) -> dict[str, Any]:
    train = partitions["train"]
    test = partitions["test"]
    if features.shape[1] == 0 or len(np.unique(labels[train])) < 2:
        return {"audit": name, "status": "unavailable", "reason": "insufficient features/classes"}
    scaler = StandardScaler().fit(features[train])
    model = LogisticRegression(C=0.1, max_iter=500, random_state=seed, solver="lbfgs")
    model.fit(scaler.transform(features[train]), labels[train])
    probabilities = model.predict_proba(scaler.transform(features[test]))[:, 1]
    result = evaluate_predictions(
        labels[test],
        probabilities,
        seed=seed,
        parameter_count=int(model.coef_.size + model.intercept_.size),
        model_name=f"audit:{name}",
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=split_fingerprint,
        bootstrap_repetitions=100,
    )
    result["status"] = "reported_audit_only"
    return result


def _label_balance(
    labels: np.ndarray, metadata: dict[str, np.ndarray], field: str
) -> dict[str, Any]:
    values = np.asarray(metadata[field])
    groups: list[dict[str, Any]] = []
    for value in np.unique(values):
        mask = values == value
        count = int(mask.sum())
        zeros = int(np.sum(labels[mask] == 0))
        ones = int(np.sum(labels[mask] == 1))
        groups.append(
            {
                "value": str(value),
                "count": count,
                "class_balance": {"0": zeros, "1": ones},
                "positive_rate": ones / count if count else float("nan"),
                "absolute_deviation_from_half": abs(ones / count - 0.5) if count else float("nan"),
            }
        )
    return {"field": field, "group_count": len(groups), "groups": groups}


def _feature_distribution_differences(
    feature_matrix: np.ndarray, partitions: dict[str, np.ndarray]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    names = list(partitions)
    for left_name, right_name in zip(names, names[1:], strict=False):
        left = feature_matrix[partitions[left_name]]
        right = feature_matrix[partitions[right_name]]
        rows: list[dict[str, Any]] = []
        for index in range(feature_matrix.shape[1]):
            left_mean = float(np.mean(left[:, index]))
            right_mean = float(np.mean(right[:, index]))
            left_std = float(np.std(left[:, index]))
            right_std = float(np.std(right[:, index]))
            pooled = max(float(np.sqrt((left_std**2 + right_std**2) / 2)), 1e-12)
            rows.append(
                {
                    "feature_index": index,
                    "left_mean": left_mean,
                    "right_mean": right_mean,
                    "left_std": left_std,
                    "right_std": right_std,
                    "standardized_mean_difference": (right_mean - left_mean) / pooled,
                }
            )
        comparisons[f"{left_name}_vs_{right_name}"] = {
            "left": left_name,
            "right": right_name,
            "feature_count": len(rows),
            "features": rows,
            "max_absolute_standardized_mean_difference": max(
                (abs(row["standardized_mean_difference"]) for row in rows), default=float("nan")
            ),
        }
    return {
        "comparison": "adjacent materialized partitions",
        "comparisons": comparisons,
    }


def run_leakage_audits(
    labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    feature_matrix: np.ndarray,
    partitions: dict[str, np.ndarray],
    *,
    dataset_fingerprint: str,
    split_fingerprint: str,
    identity_fields: list[str] | None = None,
    balance_fields: list[str] | None = None,
    seed: int = 991,
) -> dict[str, Any]:
    """Run visible, non-inference diagnostics for one materialized dataset."""

    identity = [
        field for field in (identity_fields or DEFAULT_IDENTITY_FIELDS) if field in metadata
    ]
    numeric_or_context = [field for field in metadata if field not in set(identity) | {"label"}]
    audits: dict[str, Any] = {
        "metadata_only": _audit_model(
            "metadata_only",
            _encode_fields(metadata, numeric_or_context),
            labels,
            partitions,
            dataset_fingerprint=dataset_fingerprint,
            split_fingerprint=split_fingerprint,
            seed=seed,
        ),
        "identity_only": _audit_model(
            "identity_only",
            _encode_fields(metadata, identity),
            labels,
            partitions,
            dataset_fingerprint=dataset_fingerprint,
            split_fingerprint=split_fingerprint,
            seed=seed + 1,
        ),
        "trial_order": _audit_model(
            "trial_order",
            _encode_fields(metadata, ["trial_index"] if "trial_index" in metadata else []),
            labels,
            partitions,
            dataset_fingerprint=dataset_fingerprint,
            split_fingerprint=split_fingerprint,
            seed=seed + 2,
        ),
    }
    fields = balance_fields or ["device_id", "session_id", "row_id", "cell_or_offset_id"]
    audits["label_balance"] = {
        field: _label_balance(labels, metadata, field) for field in fields if field in metadata
    }
    audits["feature_distribution_differences"] = _feature_distribution_differences(
        feature_matrix, partitions
    )
    audits["warning"] = (
        "Audit models may use identity/order metadata and are never valid SenseTrace inference results."
    )
    return audits

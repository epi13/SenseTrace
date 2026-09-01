"""Dataset schema and default-deny feature policy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_IDENTITY_FIELDS
from .errors import ForbiddenFeatureError, SchemaError

SCHEMA_VERSION = "sensetrace.dataset.v1"
REQUIRED_METADATA_FIELDS = {
    "sample_id",
    "session_id",
    "device_id",
    "bank_id",
    "row_id",
    "cell_or_offset_id",
    "trial_index",
}
ALLOWED_NUMERIC_FEATURES = {"temperature_c", "vdd_v", "refresh_age_ns", "wait_ns"}


@dataclass(frozen=True)
class FeaturePolicy:
    prohibited: frozenset[str] = frozenset(DEFAULT_IDENTITY_FIELDS)
    allowed_numeric: frozenset[str] = frozenset(ALLOWED_NUMERIC_FEATURES)

    def validate(self, fields: Iterable[str], *, allow_identity: bool = False) -> None:
        requested = set(fields)
        if not allow_identity:
            forbidden = sorted(requested & self.prohibited)
            if forbidden:
                raise ForbiddenFeatureError(
                    "identity/audit fields are grouping-only and cannot be model inputs: "
                    + ", ".join(forbidden)
                )
        permitted_identity = self.prohibited if allow_identity else frozenset()
        unsupported = sorted(requested - self.allowed_numeric - {"trace"} - permitted_identity)
        if unsupported:
            raise SchemaError(f"unsupported model feature fields: {', '.join(unsupported)}")


def validate_metadata(metadata: dict[str, np.ndarray], rows: int) -> None:
    missing = sorted(REQUIRED_METADATA_FIELDS - set(metadata))
    if missing:
        raise SchemaError(f"missing metadata fields: {', '.join(missing)}")
    for name, values in metadata.items():
        if len(values) != rows:
            raise SchemaError(f"metadata field {name!r} has {len(values)} rows; expected {rows}")
    labels = metadata.get("label")
    if labels is not None and not np.isin(labels, [0, 1]).all():
        raise SchemaError("labels must be binary values 0 or 1")


def validate_arrays(trace: np.ndarray, labels: np.ndarray, metadata: dict[str, np.ndarray]) -> None:
    if trace.ndim != 2 or trace.shape[0] == 0:
        raise SchemaError("trace must be a non-empty [samples, time] array")
    if labels.ndim != 1 or len(labels) != len(trace):
        raise SchemaError("labels must be a one-dimensional array matching trace rows")
    if not np.isin(labels, [0, 1]).all():
        raise SchemaError("labels must contain only 0 and 1")
    validate_metadata(metadata, len(trace))

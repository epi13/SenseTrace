"""Stable interface between acquisition and persistence."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Sample:
    trace: np.ndarray
    label: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class RecoveryPolicy:
    """Backend recovery contract used by the runner instead of backend names."""

    allow_resume: bool
    deterministic_replay: bool
    continuity_requirement: str


@dataclass(frozen=True)
class RecoveryDecision:
    """Auditable result of asking a backend whether a run may resume."""

    allowed: bool
    reason: str
    continuity_evidence: dict[str, Any] = field(default_factory=dict)


class AcquisitionBackend:
    name = "abstract"
    recovery_policy = RecoveryPolicy(
        allow_resume=False,
        deterministic_replay=False,
        continuity_requirement="backend has not declared resumable identity continuity",
    )

    def samples(self, start_index: int = 0) -> Iterator[Sample]:
        raise NotImplementedError

    def recovery_identity(self) -> dict[str, Any]:
        """Return immutable backend identity used by the run resume contract."""

        return {"backend": self.name}

    def validate_resume(
        self,
        *,
        persisted_run: dict[str, Any],
        persisted_config: dict[str, Any],
        current_config: dict[str, Any],
        resume_index: int,
    ) -> RecoveryDecision:
        """Validate recovery capability before finalized evidence is reopened.

        Backends must opt in through this hook.  The default remains fail-closed
        even when a subclass happens to expose a permissive-looking policy.
        """

        del persisted_run, persisted_config, current_config, resume_index
        return RecoveryDecision(
            False,
            "backend has not implemented an explicit recovery-validation hook",
            {"declared_policy": self.recovery_policy.allow_resume},
        )

    def close(self) -> None:
        """Release backend resources; backends without resources are no-ops."""

        return None

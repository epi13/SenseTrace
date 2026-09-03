"""Preregistered worker-03 experiment lifecycle and packet acquisition helpers."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .acquisition.base import Sample
from .acquisition.commodity import CommodityDramBackend
from .attestation import evidence_source_record
from .config import config_fingerprint, normalized_config, validate_config
from .errors import IntegrityError
from .excitation import CodedExcitationSchedule, ExcitationExecution, ExcitationFamily
from .fragmented import (
    authorize_claim_level,
    build_fragmented_split,
    validate_fragmented_split,
    write_fragmented_split,
)
from .hashing import sha256_json
from .packets import (
    EvidencePacket,
    FragmentStatus,
    PacketShardInfo,
    PacketShardWriter,
    ProbeFragment,
    iter_packets,
    write_packet_manifest,
)
from .protocol import (
    WORKER03_FRAGMENTED_EXACT_HOST_VERSION,
    worker03_fragmented_exact_host_protocol,
    worker03_fragmented_exact_host_protocol_hash,
)
from .receiver import (
    PACKET_RECEIVER_LADDER,
    ClaimLevel,
    NoiseResidualizer,
    ReceiverTournament,
    execute_receiver_tournament,
)
from .worker03 import WORKER03_HARDWARE_ID, collect_worker03_inventory

EXPERIMENT_STATE_SCHEMA = "sensetrace.worker03-experiment-state.v1"
EXPERIMENT_DECISION_SCHEMA = "sensetrace.worker03-experiment-decision.v1"
EXPERIMENT_STATES = (
    "planned",
    "protocol_frozen",
    "inventory_verified",
    "reference_acquisition",
    "controlled_acquisition",
    "evidence_finalized",
    "split_frozen",
    "training",
    "validation_selection",
    "test_evaluation",
    "decision",
)
_NEXT_STATE = {
    left: right for left, right in zip(EXPERIMENT_STATES[:-1], EXPERIMENT_STATES[1:], strict=True)
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class ExperimentProtocolError(ValueError):
    """A protocol or lifecycle deviation that requires a new experiment identity."""


@dataclass
class ExperimentStateMachine:
    """Fail-closed, append-only lifecycle for one preregistered experiment."""

    root: Path
    experiment_id: str
    protocol_id: str = WORKER03_FRAGMENTED_EXACT_HOST_VERSION
    protocol_hash: str = ""

    def __init__(
        self,
        root: str | Path,
        experiment_id: str | None = None,
        *,
        protocol_id: str = WORKER03_FRAGMENTED_EXACT_HOST_VERSION,
        protocol_hash: str = "",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id or self.root.name
        self.protocol_id = protocol_id
        self.protocol_hash = protocol_hash
        self._state_path = self.root / "experiment-state.json"
        self._events_path = self.root / "experiment-events.jsonl"
        if self._state_path.exists():
            try:
                record = json.loads(self._state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise IntegrityError("cannot read immutable experiment state") from exc
            if record.get("schema") != EXPERIMENT_STATE_SCHEMA:
                raise IntegrityError("unsupported experiment state schema")
            if record.get("experiment_id") != self.experiment_id:
                raise ExperimentProtocolError("experiment identity differs from persisted state")
            if protocol_hash and record.get("protocol_hash") != protocol_hash:
                raise ExperimentProtocolError(
                    "protocol fingerprint changed for existing experiment"
                )
            self.protocol_id = str(record["protocol_id"])
            self.protocol_hash = str(record["protocol_hash"])
            self._state = record
        else:
            if not protocol_hash:
                protocol_hash = "unfrozen"
            self._state = {
                "schema": EXPERIMENT_STATE_SCHEMA,
                "experiment_id": self.experiment_id,
                "protocol_id": self.protocol_id,
                "protocol_hash": protocol_hash,
                "state": "planned",
                "created_at": datetime.now(UTC).isoformat(),
                "event_count": 0,
                "immutable": {
                    "protocol_hash": protocol_hash,
                    "split_fingerprint": None,
                    "test_evaluated": False,
                },
            }
            self.protocol_hash = protocol_hash
            _atomic_json(self._state_path, self._state)

    @property
    def state(self) -> str:
        return str(self._state["state"])

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._state, sort_keys=True))

    def _append(self, event: str, **details: Any) -> None:
        event_record = {
            "schema": "sensetrace.worker03-experiment-event.v1",
            "event": event,
            "experiment_id": self.experiment_id,
            "from_state": self.state,
            "at": datetime.now(UTC).isoformat(),
            **details,
        }
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._state["event_count"] = int(self._state.get("event_count", 0)) + 1

    def transition(self, target: str, **details: Any) -> dict[str, Any]:
        if target not in EXPERIMENT_STATES:
            raise ExperimentProtocolError(f"unsupported experiment state {target!r}")
        if _NEXT_STATE.get(self.state) != target:
            raise ExperimentProtocolError(
                f"cannot transition from {self.state!r} to {target!r}; expected {_NEXT_STATE.get(self.state)!r}"
            )
        self._append("state_transition", to_state=target, **details)
        self._state["state"] = target
        _atomic_json(self._state_path, self._state)
        return self.as_dict()

    def freeze_protocol(
        self, protocol: dict[str, Any], *, config: dict[str, Any] | None = None
    ) -> str:
        if self.state != "planned":
            raise ExperimentProtocolError("protocol can only be frozen from planned state")
        protocol_hash = sha256_json(protocol)
        if self.protocol_hash not in {"", "unfrozen", protocol_hash}:
            raise ExperimentProtocolError(
                "requested protocol hash disagrees with experiment identity"
            )
        self.protocol_hash = protocol_hash
        self.protocol_id = str(protocol.get("version", self.protocol_id))
        protocol_record = {
            "schema": "sensetrace.worker03-preregistered-protocol.v1",
            "experiment_id": self.experiment_id,
            "protocol_id": self.protocol_id,
            "protocol_hash": protocol_hash,
            "protocol": protocol,
        }
        if config is not None:
            protocol_record["configuration_hash"] = config_fingerprint(config)
            _atomic_json(self.root / "config.json", normalized_config(config))
        protocol_path = self.root / "protocol.json"
        if protocol_path.exists():
            existing = json.loads(protocol_path.read_text(encoding="utf-8"))
            if existing.get("protocol_hash") != protocol_hash:
                raise ExperimentProtocolError("protocol artifact is immutable")
        else:
            _atomic_json(protocol_path, protocol_record)
        self._state["protocol_id"] = self.protocol_id
        self._state["protocol_hash"] = protocol_hash
        self._state["immutable"]["protocol_hash"] = protocol_hash
        self.transition("protocol_frozen", protocol_hash=protocol_hash)
        return protocol_hash

    def set_fingerprint(
        self, name: Literal["reference", "evidence", "split"], fingerprint: str
    ) -> None:
        if not fingerprint or fingerprint in {"unknown", "unavailable"}:
            raise ExperimentProtocolError(f"{name} fingerprint must be concrete")
        key = f"{name}_fingerprint"
        previous = self._state["immutable"].get(key)
        if previous not in {None, fingerprint}:
            raise ExperimentProtocolError(f"{name} fingerprint changed after freeze")
        self._state["immutable"][key] = fingerprint
        _atomic_json(self._state_path, self._state)

    def record_deviation(
        self, *, requested: Any, executed: Any, material: bool, reason: str
    ) -> None:
        self._append(
            "protocol_deviation",
            requested=requested,
            executed=executed,
            material=material,
            reason=reason,
        )
        if material:
            self._state["deviation"] = {"material": True, "reason": reason}
            _atomic_json(self._state_path, self._state)
            raise ExperimentProtocolError(
                "material protocol deviation requires a new experiment identity"
            )
        _atomic_json(self._state_path, self._state)

    def mark_test_evaluated(self) -> None:
        if self._state["immutable"].get("test_evaluated"):
            raise ExperimentProtocolError("test evaluation is a one-time operation")
        if self.state != "validation_selection":
            raise ExperimentProtocolError("test evaluation requires validation selection state")
        self._state["immutable"]["test_evaluated"] = True
        self.transition("test_evaluation")

    def decision(self, *, outcome: str, report: dict[str, Any]) -> dict[str, Any]:
        if self.state != "test_evaluation":
            raise ExperimentProtocolError("decision requires one completed test evaluation")
        if not self._state["immutable"].get("test_evaluated"):
            raise ExperimentProtocolError("decision requires test evaluation")
        self.transition("decision", outcome=outcome)
        record = {
            "schema": EXPERIMENT_DECISION_SCHEMA,
            "experiment_id": self.experiment_id,
            "protocol_id": self.protocol_id,
            "protocol_hash": self.protocol_hash,
            "state": self.state,
            "outcome": outcome,
            "report": report,
            "claim_boundary": "reported result cannot exceed collected independent evidence",
        }
        _atomic_json(self.root / "decision.json", record)
        return record


DEFAULT_FRAGMENT_PROBES = (
    ("cached_control", "native-v4"),
    ("dependency_chain", "native-v4"),
    ("repeated_load", "native-v4"),
    ("paired_cached_differential", "native-v4"),
)


def _sample_provenance(sample: Sample) -> dict[str, Any]:
    return {key: value for key, value in sample.metadata.items()}


def packet_from_sample(
    sample: Sample,
    *,
    packet_id: str,
    protocol_id: str,
    acquisition_id: str,
    probes: tuple[tuple[str, str], ...] = DEFAULT_FRAGMENT_PROBES,
    executed_code: tuple[int, ...] = (),
    include_label: bool = True,
    fragment_payloads: tuple[np.ndarray | None, ...] | None = None,
    fragment_statuses: tuple[str, ...] | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> EvidencePacket:
    """Packetize separate native probe payloads without metadata leakage.

    The trace-span path remains available for small custom backends, while the
    worker-03 native backend supplies one payload per declared v4 probe.
    """

    trace = np.asarray(sample.trace, dtype=np.float32)
    if trace.ndim != 1 or not np.isfinite(trace).all():
        raise ValueError("native sample trace must be finite and one-dimensional")
    payloads: tuple[np.ndarray | None, ...]
    statuses: tuple[str, ...]
    if fragment_payloads is None:
        if trace.size < len(probes):
            raise ValueError("native sample trace must fit all fallback fragments")
        payloads = tuple(
            trace[span].copy() for span in np.array_split(np.arange(trace.size), len(probes))
        )
        statuses = tuple("observed" for _ in probes)
        source_metadata = tuple(
            {
                "trace_span": [int(span[0]), int(span[-1]) + 1],
                "payload_source": "fallback trace partition; not a multi-probe acquisition",
            }
            for span in np.array_split(np.arange(trace.size), len(probes))
        )
    else:
        if len(fragment_payloads) != len(probes):
            raise ValueError("fragment_payloads must match the declared probe count")
        payloads = fragment_payloads
        statuses = fragment_statuses or tuple(
            "observed" if payload is not None else "failed" for payload in payloads
        )
        if len(statuses) != len(probes):
            raise ValueError("fragment_statuses must match the declared probe count")
        source_metadata = tuple(
            {"payload_source": "separate native-v4 probe invocation"} for _ in probes
        )
    fragments = tuple(
        ProbeFragment(
            fragment_id=f"{packet_id}:fragment-{position:02d}",
            probe_type=probe_type,
            probe_version=probe_version,
            sequence_position=position,
            target_role="target" if position == 0 else "reference",
            payload=None if payload is None else np.asarray(payload, dtype=np.float32).copy(),
            status=cast(FragmentStatus, statuses[position]),
            quality=1.0 if statuses[position] == "observed" else 0.0,
            excitation_code=executed_code,
            model_eligible=statuses[position] == "observed",
            audit_metadata={
                "source_sample_id": str(sample.metadata.get("sample_id", "unavailable")),
                **source_metadata[position],
                "packetization": "ordered weak native observations; source metadata remains outside model arrays",
            },
        )
        for position, ((probe_type, probe_version), payload) in enumerate(
            zip(probes, payloads, strict=True)
        )
    )
    packet = EvidencePacket(
        packet_id=packet_id,
        target_reference=str(sample.metadata.get("target_reference", WORKER03_HARDWARE_ID)),
        acquisition_id=acquisition_id,
        protocol_id=protocol_id,
        fragments=fragments,
        controls={
            "requested_vs_executed_separate": True,
            "executed_excitation_code": list(executed_code),
            "excitation_execution": execution_metadata or {"status": "unavailable"},
            "native_only": True,
        },
        provenance=_sample_provenance(sample),
        label=int(sample.label) if include_label else None,
    )
    packet.validate()
    return packet


def iter_fragmented_packets(
    samples: Iterable[Sample],
    *,
    protocol_id: str,
    acquisition_id: str,
    probes: tuple[tuple[str, str], ...] = DEFAULT_FRAGMENT_PROBES,
    executed_code: tuple[int, ...] = (),
    include_labels: bool = True,
    execution_metadata: dict[str, Any] | None = None,
) -> Iterator[EvidencePacket]:
    for index, sample in enumerate(samples):
        source_id = str(sample.metadata.get("sample_id", f"sample-{index:012d}"))
        yield packet_from_sample(
            sample,
            packet_id=f"{acquisition_id}:packet-{index:012d}:{source_id}",
            protocol_id=protocol_id,
            acquisition_id=acquisition_id,
            probes=probes,
            executed_code=executed_code,
            include_label=include_labels,
            execution_metadata=execution_metadata,
        )


def iter_native_fragmented_packets(
    samples: Iterable[tuple[Sample, tuple[np.ndarray | None, ...], tuple[str, ...]]],
    *,
    protocol_id: str,
    acquisition_id: str,
    probes: tuple[tuple[str, str], ...] = DEFAULT_FRAGMENT_PROBES,
    executed_codebook: tuple[tuple[int, ...], ...] = (),
    include_labels: bool = True,
    execution_metadata: dict[str, Any] | None = None,
) -> Iterator[EvidencePacket]:
    """Packetize the separate payloads emitted by the native v4 backend."""

    for index, (sample, payloads, statuses) in enumerate(samples):
        source_id = str(sample.metadata.get("sample_id", f"sample-{index:012d}"))
        code = executed_codebook[index % len(executed_codebook)] if executed_codebook else ()
        yield packet_from_sample(
            sample,
            packet_id=f"{acquisition_id}:packet-{index:012d}:{source_id}",
            protocol_id=protocol_id,
            acquisition_id=acquisition_id,
            probes=probes,
            executed_code=code,
            include_label=include_labels,
            fragment_payloads=payloads,
            fragment_statuses=statuses,
            execution_metadata=execution_metadata,
        )


def write_fragmented_packet_dataset(
    packets: Iterable[EvidencePacket],
    root: str | Path,
    *,
    config: dict[str, Any],
    protocol: dict[str, Any],
    purpose: str = "worker03_fragmented_exact_host_native",
    max_packets_per_shard: int = 256,
    acquisition_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one packet at a time and finalize an immutable packet manifest."""

    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "packet-dataset.json").exists() or any(
        destination.glob("packet-shard-*.jsonl*")
    ):
        raise IntegrityError("fragmented packet dataset output is already populated and immutable")
    protocol_hash = sha256_json(protocol)
    protocol_path = destination / "protocol.json"
    if protocol_path.exists():
        try:
            existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError("cannot read immutable protocol artifact") from exc
        if existing_protocol.get("protocol_hash") != protocol_hash:
            raise IntegrityError("protocol artifact fingerprint disagrees with packet dataset")
    else:
        _write_immutable_json(protocol_path, {"protocol": protocol, "protocol_hash": protocol_hash})
    writer = PacketShardWriter(destination, max_packets_per_shard=max_packets_per_shard)
    infos: list[PacketShardInfo] = []
    for packet in packets:
        info = writer.add(packet)
        if info is not None:
            infos.append(info)
    info = writer.finalize()
    if info is not None:
        infos.append(info)
    if not infos:
        raise IntegrityError("cannot finalize an empty fragmented packet dataset")
    config_hash = config_fingerprint(config)
    manifest = write_packet_manifest(
        destination,
        config_hash=config_hash,
        infos=infos,
        protocol_id=str(protocol.get("version", WORKER03_FRAGMENTED_EXACT_HOST_VERSION)),
        purpose=purpose,
        additional_fields={
            "protocol_hash": protocol_hash,
            "evidence_source": evidence_source_record(
                internally_consistent=True,
                tier="native_exact_host",
            ),
            "claim_boundary": "native exact-host CPU observations only; no controlled physical-memory claim",
            "model_input_policy": {
                "allowed": ["values", "observed_mask", "fragment_mask", "quality", "excitation"],
                "excluded": [
                    "labels",
                    "packet_id",
                    "session_id",
                    "boot_id",
                    "host_id",
                    "address",
                    "ordering",
                ],
            },
            "memory_policy": "stream packets and hold at most one bounded shard/batch",
            **({"acquisition": acquisition_record} if acquisition_record is not None else {}),
        },
    )
    return manifest


def _write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"cannot read immutable artifact {path.name}") from exc
        if existing != value:
            raise IntegrityError(
                f"immutable artifact {path.name} already exists with different content"
            )
        return
    _atomic_json(path, value)


def acquire_worker03_fragmented(
    config: dict[str, Any],
    output: str | Path,
    *,
    phase: Literal["reference", "controlled"] = "controlled",
    sample_count: int | None = None,
    inventory: dict[str, Any] | None = None,
    backend_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run a small exact-host native acquisition, suitable for smoke validation.

    This function intentionally does not invoke a controlled-memory adapter.
    The ``reference`` phase omits labels and uses a label-independent random
    word pattern; the ``controlled`` phase uses balanced paired labels.
    """

    validated = validate_config(config)
    protocol = worker03_fragmented_exact_host_protocol(validated)
    protocol_hash = worker03_fragmented_exact_host_protocol_hash(validated)
    if phase not in {"reference", "controlled"}:
        raise ValueError("phase must be reference or controlled")
    root = Path(output)
    experiment_id = str(validated.get("experiment", {}).get("name", root.name))
    lifecycle = ExperimentStateMachine(root, experiment_id, protocol_hash=protocol_hash)
    if lifecycle.state == "planned":
        lifecycle.freeze_protocol(protocol, config=validated)
    current_inventory: dict[str, Any] | None = inventory
    if lifecycle.state == "protocol_frozen":
        current_inventory = inventory or collect_worker03_inventory()
        target_match = current_inventory.get("target_match")
        if target_match != "matched" and bool(protocol["target"]["inventory_match_required"]):
            raise ExperimentProtocolError("worker-03 inventory does not match the frozen target")
        _write_immutable_json(root / "inventory.json", current_inventory)
        lifecycle.transition("inventory_verified", target_match=target_match)
    if current_inventory is None:
        inventory_path = root / "inventory.json"
        try:
            current_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentProtocolError(
                "resumed acquisition requires immutable inventory"
            ) from exc
    default_count = (
        protocol["reference_baseline"]["repetition_counts"].get("packets", 1024)
        if phase == "reference"
        else protocol["acquisition"]["repetition_counts"].get(
            "packets", validated.get("data", {}).get("samples", 64)
        )
    )
    count = sample_count if sample_count is not None else int(default_count)
    if count < 2 or count % 2:
        raise ValueError("sample_count must be a positive even number")
    acquisition_id = f"{experiment_id}:{phase}:{uuid.uuid4().hex}"
    if lifecycle.state == "inventory_verified":
        lifecycle.transition(
            "reference_acquisition",
            acquisition_id=acquisition_id if phase == "reference" else None,
            skipped=phase != "reference",
        )
        if phase == "controlled":
            lifecycle.transition("controlled_acquisition", acquisition_id=acquisition_id)
    elif lifecycle.state not in {"reference_acquisition", "controlled_acquisition"}:
        raise ExperimentProtocolError("acquisition can only start after inventory verification")
    if backend_factory is None:
        backend_factory = CommodityDramBackend
    acquisition = validated.get("acquisition", {})
    worker = validated.get("worker03_experiment", {})
    core_roles = protocol.get("core_roles", {})
    measurement_cores = core_roles.get("measurement", []) if isinstance(core_roles, dict) else []
    observed_inventory = (
        current_inventory.get("observed", {}) if isinstance(current_inventory, dict) else {}
    )
    physical_core_count = (
        observed_inventory.get("physical_cores") if isinstance(observed_inventory, dict) else None
    )
    backend_kwargs: dict[str, Any] = {
        "count": count,
        "trace_length": int(
            acquisition.get("trace_length", validated.get("data", {}).get("trace_length", 32))
        ),
        "seed": int(validated.get("experiment", {}).get("seed", 1337)),
        "pattern": "random_word" if phase == "reference" else "single_bit",
        "target_bit": int(worker.get("target_bit", 0)) if isinstance(worker, dict) else 0,
        "cache_control": str(acquisition.get("cache_control", "eviction_buffer")),
        "operation": str(acquisition.get("operation", "memory_read")),
        "word_count": int(acquisition.get("word_count", 1024)),
        "lock_memory": bool(acquisition.get("lock_memory", True)),
        "use_native_kernel": bool(worker.get("require_native_kernel", True))
        if isinstance(worker, dict)
        else True,
        "acquisition_session_id": acquisition_id,
        "protocol_identity": str(protocol["version"]),
        "protocol_hash": protocol_hash,
        "configuration_hash": config_fingerprint(validated),
        "code_commit": str(worker.get("code_commit", "resolved_at_freeze_time"))
        if isinstance(worker, dict)
        else "resolved_at_freeze_time",
        "host_inventory_snapshot": current_inventory,
    }
    if (
        isinstance(physical_core_count, int)
        and physical_core_count > 0
        and isinstance(measurement_cores, list)
    ):
        backend_kwargs["cpu_affinity"] = [int(core) for core in measurement_cores]
    backend = backend_factory(**backend_kwargs)
    try:
        schedule_record = protocol["target_and_label_generation"]["requested_schedule"]
        schedule = CodedExcitationSchedule(
            schedule_id=f"{experiment_id}:{phase}:schedule",
            family=cast(ExcitationFamily, str(schedule_record.get("family", "active_quiet"))),
            length=int(schedule_record.get("length", 32)),
            seed=int(schedule_record.get("seed", 1337)),
            operation="read",
        )
        physical_core_ids = (
            tuple(range(physical_core_count))
            if isinstance(physical_core_count, int) and physical_core_count > 0
            else None
        )
        execution = execute_coded_excitation(
            schedule,
            physical_core_ids=physical_core_ids,
        )
        if execution.compliance != "compliant":
            lifecycle.record_deviation(
                requested=schedule.request_record(),
                executed=execution.as_dict(),
                material=True,
                reason="native packet acquisition received a noncompliant excitation execution",
            )
        probes_raw = protocol["implementation"]["probe_versions"]
        probes = tuple((str(item["probe_type"]), str(item["probe_version"])) for item in probes_raw)
        fragmented_source = getattr(backend, "fragmented_samples", None)
        if callable(fragmented_source):
            packets = iter_native_fragmented_packets(
                fragmented_source(tuple(probe_type for probe_type, _version in probes)),
                protocol_id=str(protocol["version"]),
                acquisition_id=acquisition_id,
                probes=probes,
                executed_codebook=execution.executed_code,
                include_labels=phase == "controlled",
                execution_metadata=execution.as_dict(),
            )
            fragment_acquisition = "separate native-v4 entry point per declared probe"
        else:
            packets = iter_fragmented_packets(
                backend.samples(),
                protocol_id=str(protocol["version"]),
                acquisition_id=acquisition_id,
                probes=probes,
                executed_code=execution.executed_code[0] if execution.executed_code else (),
                include_labels=phase == "controlled",
                execution_metadata=execution.as_dict(),
            )
            fragment_acquisition = (
                "fallback trace partition for custom backend; not a multi-probe claim"
            )
        session = getattr(
            backend, "session_provenance", lambda: {"acquisition_session_id": acquisition_id}
        )()
        manifest = write_fragmented_packet_dataset(
            packets,
            root,
            config=validated,
            protocol=protocol,
            purpose=protocol["dataset_purpose"]
            if phase == "controlled"
            else "worker03_reference_baseline_native",
            max_packets_per_shard=int(acquisition.get("max_packets_per_shard", 256)),
            acquisition_record={
                "phase": phase,
                "session": session,
                "requested_settings": protocol["acquisition"],
                "executed_settings": {"backend": type(backend).__name__, "count": count},
                "requested_vs_executed": "separate records; no silent adaptation",
                "excitation_request": schedule.request_record(),
                "excitation_execution": execution.as_dict(),
                "fragment_acquisition": fragment_acquisition,
            },
        )
    finally:
        backend.close()
    lifecycle.set_fingerprint(
        "reference" if phase == "reference" else "evidence", str(manifest["dataset_fingerprint"])
    )
    # A reference-only run is intentionally left at reference_acquisition: it
    # has not collected the controlled phase and must not masquerade as a
    # finalized complete experiment.  The complete workflow advances its
    # outer lifecycle after both datasets have been written.
    if phase == "controlled":
        lifecycle.transition("evidence_finalized")
    return {
        "experiment_id": experiment_id,
        "phase": phase,
        "state": lifecycle.state,
        "manifest": manifest,
        "protocol_hash": protocol_hash,
        "physical_hardware_run": False,
    }


def run_preregistered_worker03_experiment(
    config: dict[str, Any],
    output: str | Path,
    *,
    inventory: dict[str, Any] | None = None,
    reference_samples: int | None = None,
    controlled_samples: int | None = None,
) -> dict[str, Any]:
    """Run the complete bounded reference → acquisition → receiver workflow."""

    validated = validate_config(config)
    protocol = worker03_fragmented_exact_host_protocol(validated)
    protocol_hash = worker03_fragmented_exact_host_protocol_hash(validated)
    root = Path(output)
    lifecycle = ExperimentStateMachine(
        root,
        str(validated.get("experiment", {}).get("name", root.name)),
        protocol_hash=protocol_hash,
    )
    if lifecycle.state == "planned":
        lifecycle.freeze_protocol(protocol, config=validated)
    current_inventory = inventory or collect_worker03_inventory()
    if lifecycle.state == "protocol_frozen":
        if (
            protocol["target"]["inventory_match_required"]
            and current_inventory.get("target_match") != "matched"
        ):
            raise ExperimentProtocolError("worker-03 inventory does not match the frozen target")
        _write_immutable_json(root / "inventory.json", current_inventory)
        lifecycle.transition(
            "inventory_verified", target_match=current_inventory.get("target_match")
        )
    if lifecycle.state != "inventory_verified":
        raise ExperimentProtocolError("complete experiment must start from inventory_verified")
    reference_dir = root / "reference"
    controlled_dir = root / "controlled"
    lifecycle.transition("reference_acquisition")
    reference = acquire_worker03_fragmented(
        validated,
        reference_dir,
        phase="reference",
        sample_count=reference_samples,
        inventory=current_inventory,
    )
    residualizer = NoiseResidualizer().fit_reference(
        iter_packets(reference_dir),
        dataset_fingerprint=reference["manifest"]["dataset_fingerprint"],
    )
    _write_immutable_json(root / "residualizer.json", residualizer.state_record())
    lifecycle.transition("controlled_acquisition")
    controlled = acquire_worker03_fragmented(
        validated,
        controlled_dir,
        phase="controlled",
        sample_count=controlled_samples,
        inventory=current_inventory,
    )
    lifecycle.set_fingerprint("evidence", controlled["manifest"]["dataset_fingerprint"])
    lifecycle.transition("evidence_finalized")

    def raw_controlled_factory() -> Iterator[EvidencePacket]:
        return iter_packets(controlled_dir)

    def controlled_factory() -> Iterator[EvidencePacket]:
        for packet in raw_controlled_factory():
            yield residualizer.transform(packet)

    claim_level = cast(
        ClaimLevel,
        str(
            validated.get("worker03_experiment", {}).get(
                "claim_level", "level_1_exact_host_calibrated"
            )
        ),
    )
    claim_authorization = authorize_claim_level(raw_controlled_factory(), claim_level)
    split = build_fragmented_split(
        raw_controlled_factory(),
        dataset_fingerprint=controlled["manifest"]["dataset_fingerprint"],
        claim_level=claim_level,
        seed=int(validated.get("experiment", {}).get("seed", 1337)),
        residualizer_source_dataset_fingerprint=reference["manifest"]["dataset_fingerprint"],
    )
    leakage = validate_fragmented_split(
        raw_controlled_factory(),
        split,
        dataset_fingerprint=controlled["manifest"]["dataset_fingerprint"],
        residualizer_source_dataset_fingerprint=reference["manifest"]["dataset_fingerprint"],
    )
    write_fragmented_split(str(root / "split.json"), split)
    lifecycle.set_fingerprint("split", str(split["split_fingerprint"]))
    lifecycle.transition("split_frozen", leakage_audit=leakage)
    lifecycle.transition("training")
    candidates = tuple(
        validated.get("worker03_experiment", {})
        .get("receiver", {})
        .get("candidates", PACKET_RECEIVER_LADDER)
    )
    tournament = ReceiverTournament(
        controlled["manifest"]["dataset_fingerprint"],
        split["split_fingerprint"],
        candidates=candidates,
        claim_level=claim_level,
    )
    tournament_report = execute_receiver_tournament(
        tournament,
        packet_factory=controlled_factory,
        split=split,
        max_fragments=len(protocol["implementation"]["probe_versions"]),
        max_payload_length=max(1, int(validated.get("data", {}).get("trace_length", 32))),
        excitation_width=len(protocol["core_roles"]["excitation"]),
        batch_size=32,
        maximum_training_packets=min(4096, max(2, controlled_samples or 64)),
        seed=int(validated.get("experiment", {}).get("seed", 1337)),
        preprocessing_fingerprint=sha256_json(
            {
                "residualizer": residualizer.state_record(),
                "feature_policy": "packet_summary_features v2 only",
            }
        ),
    )
    lifecycle.transition("validation_selection")
    lifecycle.mark_test_evaluated()
    decision = lifecycle.decision(
        outcome="receiver_tournament_complete; claim level remains protocol-gated",
        report={
            "selected_candidate": tournament_report["selection"]["selected_candidate"],
            "test": tournament_report["selected"]["test"],
            "claim_boundary": protocol["claim_boundary"],
        },
    )
    report = {
        "schema": "sensetrace.worker03-preregistered-experiment-report.v1",
        "experiment_id": lifecycle.experiment_id,
        "protocol_id": protocol["version"],
        "protocol_hash": protocol_hash,
        "state": lifecycle.state,
        "reference": reference["manifest"],
        "controlled": controlled["manifest"],
        "residualizer": residualizer.state_record(),
        "split": split,
        "claim_authorization": claim_authorization,
        "leakage_audit": leakage,
        "receiver_tournament": tournament_report,
        "decision": decision,
        "physical_hardware_run": False,
        "physical_worker03_or_fpga_evidence_acquired": False,
        "historical_commodity_gate": "C: primitive unsuitable; not reopened",
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_immutable_json(root / "report.json", report)
    return report


def execute_coded_excitation(
    schedule: CodedExcitationSchedule,
    *,
    step_runner: Callable[[int, tuple[int, ...], str], bool] | None = None,
    core_ids: tuple[int, ...] | None = None,
    physical_core_ids: tuple[int, ...] | None = None,
) -> ExcitationExecution:
    """Execute a bounded codebook callback and retain compliance evidence."""

    schedule.validate()
    assigned = tuple(schedule.role_assignment.roles["excitation"])
    available = tuple(physical_core_ids or assigned)
    if len(set(assigned)) != len(assigned) or any(core not in available for core in assigned):
        raise ExperimentProtocolError(
            "excitation roles do not map to distinct available physical cores"
        )
    executed: list[tuple[int, ...]] = []
    interrupted: list[int] = []
    started = time.monotonic_ns()
    for step in schedule.steps():
        ok = (
            True
            if step_runner is None
            else bool(
                step_runner(
                    step.sequence_position,
                    tuple(int(value) for value in step.active),
                    step.operation,
                )
            )
        )
        if ok:
            executed.append(tuple(int(value) for value in step.active))
        else:
            interrupted.append(step.sequence_position)
            break
    ended = time.monotonic_ns()
    compliance = (
        "compliant" if len(executed) == schedule.length else "partial" if executed else "failed"
    )
    execution = ExcitationExecution(
        schedule_fingerprint=schedule.fingerprint(),
        executed_code=tuple(executed),
        start_clock=started,
        end_clock=ended,
        core_ids=tuple(core_ids or assigned),
        interrupted_positions=tuple(interrupted),
        compliance=cast(Literal["compliant", "partial", "failed"], compliance),
        witness={
            "scheduler_affinity": (
                "callback-observed; OS scheduler result is not inferred"
                if step_runner is not None
                else "not observed; software schedule walk only"
            ),
            "execution_authority": (
                "caller step-runner observation"
                if step_runner is not None
                else "local schedule generation; no hardware execution claim"
            ),
            "compliance_scope": "requested/executed code contract only; physical core execution is not inferred",
            "requested_code_fingerprint": schedule.fingerprint(),
            "executed_step_count": len(executed),
            "timing_uncertainty": "userspace monotonic clock; not hardware timing",
        },
    )
    execution.validate(schedule)
    return execution

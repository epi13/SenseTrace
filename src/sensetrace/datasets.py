"""Dataset manifests, fingerprints, and model feature construction."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .acquisition.controlled import (
    CONTROLLED_INTERFACE_VERSION,
    PHYSICAL_CONTROLLED_EVIDENCE_CONTRACT_VERSION,
    PHYSICAL_CONTROLLED_EVIDENCE_PLANE,
    PHYSICAL_CONTROLLED_REQUIRED_METADATA_FIELDS,
    ControlledAcquisitionProvenance,
    ControlledCommand,
    ControlledInterfaceCapabilities,
    ControlledMemoryTopology,
    ControlledTraceAcquisition,
    ControlledTraceChannel,
)
from .attestation import require_adapter_attestation as require_adapter_attestation_record
from .config import config_fingerprint
from .errors import IntegrityError, SchemaError
from .hashing import sha256_bytes, sha256_json, sha256_text
from .protocol import PHASE1A_COMMODITY_BASELINE_VERSION
from .schema import SCHEMA_VERSION, FeaturePolicy
from .storage import ShardInfo, dataset_fingerprint, load_shards

PHYSICAL_CONTROLLED_REQUIRED_FIELDS = PHYSICAL_CONTROLLED_REQUIRED_METADATA_FIELDS


def write_dataset_manifest(
    run_dir: str | Path,
    *,
    config: dict[str, Any],
    condition: str,
    shard_infos: list[ShardInfo],
    label_stream_fingerprint: str,
    class_balance: dict[str, int | str] | None = None,
    provenance: dict[str, Any] | None = None,
    acquisition_sessions: list[dict[str, Any]] | None = None,
    campaign_id: str | None = None,
    dataset_purpose: str | None = None,
    protocol_identity: str | None = None,
    protocol_hash: str | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    config_hash = config_fingerprint(config)
    manifest_provenance = dict(provenance or {})
    effective_purpose = str(
        dataset_purpose or manifest_provenance.get("dataset_purpose", "generic")
    )
    effective_protocol_identity = protocol_identity or manifest_provenance.get("protocol_identity")
    effective_protocol_hash = protocol_hash or manifest_provenance.get("protocol_hash")
    fingerprint = dataset_fingerprint(shard_infos, config_hash=config_hash)
    manifest = {
        "schema": "sensetrace.dataset-manifest.v1",
        "dataset_schema": SCHEMA_VERSION,
        "condition": condition,
        "dataset_purpose": effective_purpose,
        "created_at": datetime.now(UTC).isoformat(),
        "config_hash": config_hash,
        "dataset_fingerprint": fingerprint,
        "label_stream_fingerprint": label_stream_fingerprint,
        "class_balance": class_balance or {"0": "unavailable", "1": "unavailable"},
        "rows": sum(info.rows for info in shard_infos),
        "shards": [info.as_dict() for info in shard_infos],
    }
    if effective_protocol_identity is not None:
        manifest["protocol_identity"] = str(effective_protocol_identity)
    if effective_protocol_hash is not None:
        manifest["protocol_hash"] = str(effective_protocol_hash)
    if manifest_provenance:
        manifest["provenance"] = manifest_provenance
        evidence_contract = manifest_provenance.get("physical_evidence_contract")
        if isinstance(evidence_contract, dict):
            manifest["physical_evidence_contract"] = evidence_contract
        capabilities = manifest_provenance.get("capabilities")
        if isinstance(capabilities, dict):
            manifest["capabilities"] = capabilities
    if acquisition_sessions is not None:
        manifest["acquisition_sessions"] = acquisition_sessions
    if campaign_id is not None:
        manifest["campaign_id"] = campaign_id
    path = root / "dataset.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def read_dataset_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "dataset.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read dataset manifest {path}") from exc
    if manifest.get("schema") != "sensetrace.dataset-manifest.v1":
        raise IntegrityError("unsupported dataset manifest schema")
    return manifest


def load_dataset(
    run_dir: str | Path,
    *,
    expected_purpose: str | None = None,
    expected_protocol_identity: str | None = None,
    expected_protocol_hash: str | None = None,
    require_adapter_attestation: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[ShardInfo], dict[str, Any]]:
    root = Path(run_dir)
    manifest = read_dataset_manifest(root)
    traces, labels, metadata, shards = load_shards(root)
    expected = dataset_fingerprint(shards, config_hash=manifest["config_hash"])
    if expected != manifest.get("dataset_fingerprint"):
        raise IntegrityError("dataset fingerprint does not match finalized shard evidence")
    if manifest.get("rows") != len(labels):
        raise IntegrityError("dataset manifest row count does not match shards")
    if expected_purpose == "physical_phase1a":
        _validate_physical_phase1a_dataset(
            metadata,
            manifest,
            root,
            expected_protocol_identity=expected_protocol_identity,
            expected_protocol_hash=expected_protocol_hash,
        )
    elif expected_purpose == "physical_controlled_hardware":
        _validate_physical_controlled_dataset(
            traces,
            metadata,
            manifest,
            root,
            require_adapter_attestation=require_adapter_attestation,
        )
    else:
        _validate_physical_timing_provenance(metadata, manifest, root)
        if expected_purpose is not None and manifest.get("dataset_purpose") != expected_purpose:
            raise IntegrityError(
                f"dataset {root} purpose {manifest.get('dataset_purpose')!r} does not match "
                f"expected purpose {expected_purpose!r}"
            )
        if manifest.get("dataset_purpose") == "phase2_mock_controlled":
            _validate_phase2_mock_dataset(metadata, manifest, root)
    _validate_allocation_provenance(metadata, manifest, root)
    return traces, labels, metadata, shards, manifest


def _validate_phase2_mock_dataset(
    metadata: dict[str, np.ndarray], manifest: dict[str, Any], source: Path
) -> None:
    """Validate the mock claim boundary instead of treating it as missing data."""

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise IntegrityError("controlled mock dataset requires a provenance mapping")
    if provenance.get("dataset_purpose") != "phase2_mock_controlled":
        raise IntegrityError("controlled mock provenance has the wrong dataset purpose")
    if provenance.get("evidence_plane") != "controlled_memory_interface_mock":
        raise IntegrityError("controlled mock provenance has the wrong evidence plane")
    if provenance.get("interface_name") != "controlled-memory-interface-mock-v1":
        raise IntegrityError("controlled mock provenance lacks its versioned interface identity")
    topology = provenance.get("topology")
    if not isinstance(topology, dict) or topology.get("source") != "unavailable":
        raise IntegrityError("controlled mock topology must remain explicitly unavailable")
    if topology.get("virtual_addresses_promoted_to_physical") is not False:
        raise IntegrityError("controlled mock cannot promote virtual addresses to topology")
    required = {
        "controlled_interface_name": "controlled-memory-interface-mock-v1",
        "controlled_topology_source": "unavailable",
        "measurement_primitive": "controlled-memory-interface-mock",
    }
    for field, expected in required.items():
        _require_all_values(_metadata_values(metadata, field), expected, field)
    for value in _metadata_values(metadata, "controlled_topology"):
        try:
            record = json.loads(value)
        except json.JSONDecodeError as exc:
            raise IntegrityError("controlled mock contains invalid topology JSON") from exc
        if not isinstance(record, dict) or record.get("source") != "unavailable":
            raise IntegrityError("controlled mock shard contains a physical topology claim")
    config_hash = provenance.get("controller_configuration_hash")
    if not isinstance(config_hash, str) or not config_hash:
        raise IntegrityError("controlled mock lacks controller configuration hash")
    _require_all_values(
        _metadata_values(metadata, "controller_config_hash"), config_hash, "controller_config_hash"
    )


def _validate_physical_controlled_dataset(
    traces: np.ndarray,
    metadata: dict[str, np.ndarray],
    manifest: dict[str, Any],
    source: Path,
    *,
    require_adapter_attestation: bool = False,
) -> None:
    """Validate the complete physical controlled-hardware evidence chain.

    The manifest is a claim about the rows, not an authority that can upgrade
    them. Every duplicated identity is checked against the serialized typed
    objects and the session ledgers before this boundary returns successfully.
    """

    if manifest.get("dataset_purpose") != "physical_controlled_hardware":
        raise IntegrityError(
            f"dataset {source} is not explicitly marked for physical controlled-hardware analysis"
        )
    config_path = source / "config.json"
    try:
        persisted_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(
            "physical controlled-hardware data requires readable config material"
        ) from exc
    if not isinstance(persisted_config, dict):
        raise IntegrityError("physical controlled-hardware config material is malformed")
    if config_fingerprint(persisted_config) != manifest.get("config_hash"):
        raise IntegrityError("physical controlled-hardware config material disagrees with manifest")
    acquisition_config = persisted_config.get("acquisition", {})
    if (
        not isinstance(acquisition_config, dict)
        or acquisition_config.get("backend") != "controlled_hardware"
    ):
        raise IntegrityError(
            "physical controlled-hardware data was not acquired by the hardware backend"
        )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise IntegrityError("physical controlled-hardware data requires provenance")
    contract = manifest.get("physical_evidence_contract")
    if not isinstance(contract, dict):
        raise IntegrityError("physical controlled-hardware data requires an evidence contract")
    if contract.get("version") != PHYSICAL_CONTROLLED_EVIDENCE_CONTRACT_VERSION:
        raise IntegrityError(
            "physical controlled-hardware evidence contract version is unsupported"
        )
    if contract.get("status") != "satisfied":
        raise IntegrityError("physical controlled-hardware evidence contract is not satisfied")
    required_fields = contract.get("required_fields")
    if (
        not isinstance(required_fields, list)
        or not all(isinstance(value, str) for value in required_fields)
        or len(required_fields) != len(set(required_fields))
        or {str(value) for value in required_fields} != PHYSICAL_CONTROLLED_REQUIRED_FIELDS
    ):
        raise IntegrityError("physical controlled-hardware required-field contract is incomplete")
    if traces.ndim != 2 or not np.isfinite(traces).all():
        raise IntegrityError(
            "physical controlled-hardware trace payload is not finite and two-dimensional"
        )

    identity = manifest.get("protocol_identity")
    protocol_hash = manifest.get("protocol_hash")
    if identity != CONTROLLED_INTERFACE_VERSION:
        raise IntegrityError("physical controlled-hardware data has no real interface identity")
    _physical_identity(protocol_hash, "manifest protocol hash")
    expected_provenance = {
        "dataset_purpose": "physical_controlled_hardware",
        "interface_name": CONTROLLED_INTERFACE_VERSION,
        "protocol_identity": identity,
        "protocol_hash": protocol_hash,
        "evidence_plane": PHYSICAL_CONTROLLED_EVIDENCE_PLANE,
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise IntegrityError(f"manifest and provenance disagree on {field}")
    if provenance.get("physical_evidence_contract") != contract:
        raise IntegrityError("manifest and provenance evidence contracts disagree")
    if require_adapter_attestation:
        try:
            require_adapter_attestation_record(
                provenance,
                controller_identity=str(provenance["controller_identity"]),
                firmware_identity=str(provenance["controller_firmware_id"]),
                configuration_fingerprint=str(provenance["controller_config_hash"]),
                target_identity=str(provenance["experiment_target_id"]),
                acquisition_session_identity=str(
                    manifest["acquisition_sessions"][0]["acquisition_session_id"]
                ),
                host_inventory_fingerprint=str(provenance["host_inventory_fingerprint"]),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise IntegrityError(
                f"physical controlled-hardware adapter attestation failed: {exc}"
            ) from exc
    capabilities_record = provenance.get("capabilities")
    if not isinstance(capabilities_record, dict):
        raise IntegrityError("physical controlled-hardware data requires a capability snapshot")
    try:
        capabilities = ControlledInterfaceCapabilities(**capabilities_record)
        capabilities.validate()
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"physical controller capability snapshot is invalid: {exc}") from exc
    if (
        capabilities.interface_name != identity
        or capabilities.protocol_hash != protocol_hash
        or "read" not in capabilities.supported_command_kinds
        or capabilities.topology_source != "controlled_hardware"
    ):
        raise IntegrityError("physical controller capabilities contradict the evidence contract")
    if provenance.get("capabilities") != manifest.get("capabilities"):
        raise IntegrityError("manifest and provenance capability snapshots disagree")
    if (
        not isinstance(provenance.get("claim_boundary"), str)
        or not provenance["claim_boundary"].strip()
    ):
        raise IntegrityError(
            "physical controlled-hardware data requires an explicit claim boundary"
        )

    ledgers = manifest.get("acquisition_sessions")
    if (
        not isinstance(ledgers, list)
        or not ledgers
        or any(not isinstance(item, dict) for item in ledgers)
    ):
        raise IntegrityError("physical controlled-hardware data requires source session ledgers")
    ledger_by_session: dict[str, dict[str, Any]] = {}
    ledger_fields = (
        "interface_name",
        "evidence_plane",
        "protocol_identity",
        "protocol_hash",
        "experiment_target_id",
        "controller_firmware_id",
        "controller_config_hash",
        "device_identity",
        "dimm_identity",
        "calibration_state",
        "hardware_clock_id",
        "sampling_clock_id",
        "acquisition_trigger",
        "trigger_identity",
        "acquisition_configuration_hash",
        "timing_provenance",
        "refresh_relationship",
        "command_provenance",
    )
    for ledger in ledgers:
        session_id = ledger.get("acquisition_session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise IntegrityError("physical session ledger lacks acquisition session identity")
        if ledger.get("status") != "completed":
            raise IntegrityError(
                "physical controlled-hardware data references an incomplete session"
            )
        allocation = ledger.get("allocation_id")
        _physical_identity(allocation, f"session {session_id} allocation identity")
        for field in ledger_fields:
            if ledger.get(field) != provenance.get(field):
                raise IntegrityError(f"session ledger disagrees with manifest on {field}")
            _physical_identity(ledger.get(field), f"session {session_id} {field}")
        if ledger.get("capabilities") != capabilities_record:
            raise IntegrityError(
                f"session {session_id} capability snapshot disagrees with manifest"
            )
        if session_id in ledger_by_session:
            raise IntegrityError(f"duplicate physical acquisition session ledger {session_id}")
        ledger_by_session[session_id] = ledger

    for field in PHYSICAL_CONTROLLED_REQUIRED_FIELDS:
        _metadata_values(metadata, field)
    row_count = len(traces)
    row_values = {
        field: _metadata_values(metadata, field) for field in PHYSICAL_CONTROLLED_REQUIRED_FIELDS
    }
    if any(len(values) != row_count for values in row_values.values()):
        raise IntegrityError("physical controlled-hardware metadata row counts disagree")

    for index in range(row_count):
        command = _decode_controlled_command(row_values["controlled_command"][index])
        typed_provenance = _decode_controlled_provenance(row_values["controlled_provenance"][index])
        topology = _decode_controlled_topology(row_values["controlled_topology"][index])
        acquisition = _decode_controlled_acquisition(
            row_values["controlled_trace_acquisition"][index]
        )
        result = _decode_controlled_result(row_values["controlled_result"][index])
        _validate_physical_row(
            index,
            row_values,
            command,
            typed_provenance,
            topology,
            acquisition,
            result,
            protocol_hash=str(protocol_hash),
            ledger_by_session=ledger_by_session,
        )


def _physical_identity(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrityError(f"physical evidence {field} is missing")
    if value != value.strip():
        raise IntegrityError(f"physical evidence {field} has surrounding whitespace")
    normalized = value.strip().lower()
    if normalized in {"unavailable", "unknown", "none", "null", "n/a"}:
        raise IntegrityError(f"physical evidence {field} is an unavailable placeholder")
    if any(token in normalized for token in ("mock", "synthetic", "virtual", "derived")):
        raise IntegrityError(f"physical evidence {field} uses a synthetic identity")
    return value


def _decode_json_object(value: object, field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"physical evidence {field} is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise IntegrityError(f"physical evidence {field} must be a JSON object")
    return decoded


def _decode_controlled_command(value: object) -> ControlledCommand:
    try:
        command = ControlledCommand(**_decode_json_object(value, "command"))
        command.validate()
        return command
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"physical evidence command contract is invalid: {exc}") from exc


def _decode_controlled_provenance(value: object) -> ControlledAcquisitionProvenance:
    try:
        provenance = ControlledAcquisitionProvenance(**_decode_json_object(value, "provenance"))
        provenance.validate()
        return provenance
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"physical evidence provenance contract is invalid: {exc}") from exc


def _decode_controlled_topology(value: object) -> ControlledMemoryTopology:
    try:
        topology = ControlledMemoryTopology(**_decode_json_object(value, "topology"))
        topology.validate()
        return topology
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"physical evidence topology contract is invalid: {exc}") from exc


def _decode_controlled_acquisition(value: object) -> ControlledTraceAcquisition:
    record = _decode_json_object(value, "trace acquisition")
    channels = record.get("channels")
    if not isinstance(channels, list):
        raise IntegrityError("physical evidence trace acquisition channels are malformed")
    try:
        record["channels"] = tuple(ControlledTraceChannel(**item) for item in channels)
        acquisition = ControlledTraceAcquisition(**record)
        acquisition.validate()
        return acquisition
    except (TypeError, ValueError) as exc:
        raise IntegrityError(
            f"physical evidence trace acquisition contract is invalid: {exc}"
        ) from exc


def _decode_controlled_result(value: object):
    record = _decode_json_object(value, "command result")
    command = _decode_controlled_command(record.get("command"))
    provenance = _decode_controlled_provenance(record.get("provenance"))
    topology = _decode_controlled_topology(record.get("topology"))
    acquisition_record = record.get("acquisition")
    acquisition = (
        None
        if acquisition_record is None
        else _decode_controlled_acquisition(json.dumps(acquisition_record, sort_keys=True))
    )
    try:
        from .acquisition.controlled import ControlledCommandResult

        status_value = record.get("status")
        completed_value = record.get("completed_at_hardware_ticks")
        if status_value not in {"complete", "unsupported", "failed"}:
            raise ValueError("unsupported command result status")
        if isinstance(completed_value, bool) or not isinstance(completed_value, int):
            raise ValueError("invalid command result completion timing")
        result = ControlledCommandResult(
            command=command,
            status=cast(Literal["complete", "unsupported", "failed"], status_value),
            topology=topology,
            provenance=provenance,
            completed_at_hardware_ticks=completed_value,
            acquisition=acquisition,
            failure=record.get("failure"),
        )
        result.validate()
        return result
    except (TypeError, ValueError) as exc:
        raise IntegrityError(
            f"physical evidence command result contract is invalid: {exc}"
        ) from exc


def _validate_physical_row(
    index: int,
    values: dict[str, list[str]],
    command: ControlledCommand,
    provenance: ControlledAcquisitionProvenance,
    topology: ControlledMemoryTopology,
    acquisition: ControlledTraceAcquisition,
    result: Any,
    *,
    protocol_hash: str,
    ledger_by_session: dict[str, dict[str, Any]],
) -> None:
    if topology.source != "controlled_hardware":
        raise IntegrityError(f"physical row {index} lacks hardware-sourced topology")
    topology_fields = {key: value for key, value in topology.as_dict().items() if key != "source"}
    if not any(
        isinstance(value, str)
        and value.strip().lower() not in {"unavailable", "unknown", "none", "null", "n/a"}
        for value in topology_fields.values()
    ):
        raise IntegrityError(f"physical row {index} has no concrete hardware topology")
    if (
        result.command.as_dict() != command.as_dict()
        or result.provenance.as_dict() != provenance.as_dict()
        or result.topology.as_dict() != topology.as_dict()
        or result.acquisition is None
        or result.acquisition.as_dict() != acquisition.as_dict()
    ):
        raise IntegrityError(
            f"physical row {index} serialized objects disagree with command result"
        )
    for field, value in {
        "target": provenance.experiment_target_id,
        "firmware": provenance.controller_firmware_id,
        "controller config": provenance.controller_config_hash,
        "device": provenance.device_identity,
        "dimm": provenance.dimm_identity,
        "calibration": provenance.calibration_state,
        "command clock": provenance.hardware_clock_id,
        "sampling clock": provenance.sampling_clock_id,
        "trigger": provenance.trigger_identity,
        "acquisition config": provenance.acquisition_configuration_hash,
        "timing": provenance.timing_provenance,
        "refresh": provenance.refresh_relationship,
        "command provenance": provenance.command_provenance,
    }.items():
        _physical_identity(value, f"row {index} {field}")
    for field, expected in {
        "controlled_interface_name": CONTROLLED_INTERFACE_VERSION,
        "controlled_protocol_hash": protocol_hash,
        "controlled_target_id": provenance.experiment_target_id,
        "controller_firmware_id": provenance.controller_firmware_id,
        "controller_config_hash": provenance.controller_config_hash,
        "controlled_command_id": command.command_id,
        "controlled_command_sequence_id": command.command_sequence_id,
        "controlled_address_token": command.address_token,
        "controlled_command_clock_id": provenance.hardware_clock_id,
        "controlled_acquisition_id": acquisition.acquisition_id,
        "controlled_sampling_clock_id": acquisition.hardware_clock_id,
        "controlled_trigger_identity": acquisition.trigger_id,
        "controlled_topology_source": topology.source,
        "controlled_timing_provenance": provenance.timing_provenance,
        "controlled_refresh_relationship": provenance.refresh_relationship,
        "controlled_command_provenance": provenance.command_provenance,
        "controlled_acquisition_configuration_hash": provenance.acquisition_configuration_hash,
        "controlled_calibration_state": provenance.calibration_state,
        "controlled_calibration_id": acquisition.channels[0].calibration_id,
        "device_id": provenance.device_identity,
        "dimm_identity": provenance.dimm_identity,
    }.items():
        if values[field][index] != expected:
            raise IntegrityError(f"physical row {index} duplicated provenance disagrees on {field}")
        _physical_identity(values[field][index], f"row {index} {field}")
    if provenance.command_sequence_id != command.command_sequence_id:
        raise IntegrityError(f"physical row {index} command sequence disagrees with provenance")
    if command.command_clock_id != provenance.hardware_clock_id:
        raise IntegrityError(f"physical row {index} command clock disagrees with provenance")
    if acquisition.command_sequence_id != command.command_sequence_id:
        raise IntegrityError(f"physical row {index} acquisition sequence disagrees with command")
    if acquisition.trigger_id != provenance.trigger_identity:
        raise IntegrityError(f"physical row {index} trigger identity disagrees with provenance")
    if acquisition.refresh_relationship != provenance.refresh_relationship:
        raise IntegrityError(f"physical row {index} refresh relationship disagrees with provenance")
    try:
        channel_ids = json.loads(values["controlled_trace_channel_ids"][index])
    except json.JSONDecodeError as exc:
        raise IntegrityError(f"physical row {index} channel identity JSON is malformed") from exc
    expected_channel_ids = [channel.channel_id for channel in acquisition.channels]
    if channel_ids != expected_channel_ids:
        raise IntegrityError(f"physical row {index} trace channel identities disagree")
    for channel in acquisition.channels:
        _physical_identity(channel.channel_id, f"row {index} channel identity")
        _physical_identity(channel.units, f"row {index} channel units")
        _physical_identity(channel.sampling_clock_id, f"row {index} channel sampling clock")
        _physical_identity(channel.calibration_id, f"row {index} channel calibration")
    if values["acquisition_session_id"][index] not in ledger_by_session:
        raise IntegrityError(f"physical row {index} has no matching source session ledger")
    _physical_identity(values["acquisition_session_id"][index], f"row {index} acquisition session")
    _physical_identity(values["allocation_id"][index], f"row {index} allocation")
    ledger = ledger_by_session[values["acquisition_session_id"][index]]
    if values["allocation_id"][index] != ledger.get("allocation_id"):
        raise IntegrityError(
            f"physical row {index} allocation disagrees with source session ledger"
        )
    for field in (
        "experiment_target_id",
        "controller_firmware_id",
        "controller_config_hash",
        "device_identity",
        "dimm_identity",
        "calibration_state",
        "hardware_clock_id",
        "sampling_clock_id",
        "trigger_identity",
        "acquisition_configuration_hash",
        "timing_provenance",
        "refresh_relationship",
        "command_provenance",
    ):
        if provenance.as_dict()[field] != ledger.get(field):
            raise IntegrityError(
                f"physical row {index} provenance disagrees with source session ledger"
            )


def validate_physical_evidence_dataset(
    run_dir: str | Path,
    *,
    expected_purpose: str = "physical_controlled_hardware",
    require_adapter_attestation: bool = False,
) -> dict[str, Any]:
    """Load a dataset through an explicit physical-evidence claim boundary."""

    _traces, _labels, _metadata, _shards, manifest = load_dataset(
        run_dir,
        expected_purpose=expected_purpose,
        require_adapter_attestation=require_adapter_attestation,
    )
    return manifest


def _metadata_values(metadata: dict[str, np.ndarray], field: str) -> list[str]:
    if field not in metadata:
        raise IntegrityError(
            f"physical Phase 1A dataset is missing required shard metadata {field!r}"
        )
    return [str(value) for value in np.asarray(metadata[field], dtype=object)]


def _require_all_values(values: list[str], expected: str, field: str) -> None:
    if not values or any(value != expected for value in values):
        raise IntegrityError(
            f"physical Phase 1A shard metadata {field!r} is not uniformly {expected!r}"
        )


def _validate_physical_phase1a_dataset(
    metadata: dict[str, np.ndarray],
    manifest: dict[str, Any],
    source: Path,
    *,
    expected_protocol_identity: str | None,
    expected_protocol_hash: str | None,
) -> None:
    """Validate the complete physical Phase 1A identity and scope boundary."""

    if manifest.get("dataset_purpose") != "physical_phase1a":
        raise IntegrityError(
            f"dataset {source} is not explicitly marked for physical Phase 1A analysis"
        )
    identity = manifest.get("protocol_identity")
    protocol_hash = manifest.get("protocol_hash")
    if identity != PHASE1A_COMMODITY_BASELINE_VERSION:
        raise IntegrityError(
            "physical Phase 1A requires protocol identity "
            f"{PHASE1A_COMMODITY_BASELINE_VERSION!r}; found {identity!r}"
        )
    if not isinstance(protocol_hash, str) or not protocol_hash:
        raise IntegrityError("physical Phase 1A requires a non-empty protocol hash")
    if expected_protocol_identity is not None and identity != expected_protocol_identity:
        raise IntegrityError(
            "physical Phase 1A protocol identity does not match the analysis boundary"
        )
    if expected_protocol_hash is not None and protocol_hash != expected_protocol_hash:
        raise IntegrityError("physical Phase 1A protocol hash does not match the analysis boundary")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise IntegrityError("physical Phase 1A requires a manifest provenance mapping")
    if provenance.get("protocol_identity") != identity:
        raise IntegrityError("manifest and provenance protocol identities disagree")
    if provenance.get("protocol_hash") != protocol_hash:
        raise IntegrityError("manifest and provenance protocol hashes disagree")
    scope = provenance.get("artificial_timing_perturbation")
    if not isinstance(scope, dict):
        raise IntegrityError("physical Phase 1A requires an explicit timing scope contract")
    required_scope = {
        "allowed": False,
        "timing_perturbation_cycles": 0,
        "timing_perturbation_label": 1,
        "label_correlated": False,
        "applied": False,
        "calibration_namespace": "forbidden",
        "physical_phase1a_forbidden": True,
    }
    if any(scope.get(field) != value for field, value in required_scope.items()):
        raise IntegrityError("physical Phase 1A manifest timing scope is missing or contradictory")

    _require_all_values(
        _metadata_values(metadata, "protocol_identity"), str(identity), "protocol_identity"
    )
    _require_all_values(
        _metadata_values(metadata, "protocol_hash"), str(protocol_hash), "protocol_hash"
    )
    _require_all_values(
        _metadata_values(metadata, "timing_perturbation_cycles"), "0", "timing_perturbation_cycles"
    )
    _require_all_values(
        _metadata_values(metadata, "timing_perturbation_label"), "1", "timing_perturbation_label"
    )
    _require_all_values(
        _metadata_values(metadata, "timing_perturbation_applied"),
        "False",
        "timing_perturbation_applied",
    )
    _require_all_values(
        _metadata_values(metadata, "calibration_namespace"),
        "not_calibration",
        "calibration_namespace",
    )
    _require_all_values(
        _metadata_values(metadata, "artificial_timing_perturbation_allowed"),
        "False",
        "artificial_timing_perturbation_allowed",
    )

    ledgers = manifest.get("acquisition_sessions")
    if (
        not isinstance(ledgers, list)
        or not ledgers
        or any(not isinstance(item, dict) for item in ledgers)
    ):
        raise IntegrityError("physical Phase 1A requires complete source session ledgers")
    ledger_ids: set[str] = set()
    for ledger in ledgers:
        ledger_identity = ledger.get("protocol_identity")
        ledger_hash = ledger.get("protocol_hash")
        if ledger_identity != identity or ledger_hash != protocol_hash:
            raise IntegrityError(
                "source session ledger protocol identity/hash disagrees with manifest"
            )
        if ledger.get("acquisition_scope") != "physical Phase 1A commodity baseline":
            raise IntegrityError("source session ledger is outside the physical Phase 1A scope")
        ledger_scope = ledger.get("timing_perturbation")
        if not isinstance(ledger_scope, dict) or any(
            ledger_scope.get(field) != value
            for field, value in {
                "cycles": 0,
                "label": 1,
                "allowed": False,
                "physical_phase1a_forbidden": True,
                "namespace": "not_calibration",
            }.items()
        ):
            raise IntegrityError("source session ledger timing scope is missing or contradictory")
        ledger_id = str(ledger.get("acquisition_session_id", ""))
        allocation = ledger.get("controlled_memory_region", {})
        if not ledger_id or not isinstance(allocation, dict) or not allocation.get("allocation_id"):
            raise IntegrityError("source session ledger lacks acquisition or allocation identity")
        ledger_ids.add(ledger_id)
    metadata_session_values = set(_metadata_values(metadata, "acquisition_session_id"))
    if metadata_session_values != ledger_ids:
        raise IntegrityError("shard metadata sessions do not match the source session ledgers")

    source_manifest_file = source / "source-manifests.json"
    if source_manifest_file.exists():
        try:
            source_record = json.loads(source_manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError("cannot read combined dataset source-manifests.json") from exc
        embedded = source_record.get("source_manifests")
        if not isinstance(embedded, list) or not embedded:
            raise IntegrityError("combined physical dataset has no embedded source manifests")
        if any(
            not isinstance(item, dict)
            or item.get("dataset_purpose") != "physical_phase1a"
            or item.get("protocol_identity") != identity
            or item.get("protocol_hash") != protocol_hash
            for item in embedded
        ):
            raise IntegrityError("combined source manifests disagree with physical protocol scope")


def _validate_physical_timing_provenance(
    metadata: dict[str, np.ndarray], manifest: dict[str, Any], source: Path
) -> None:
    """Fail closed when baseline evidence carries a calibration perturbation."""

    provenance = manifest.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    protocol_identity = manifest.get("protocol_identity") or provenance.get("protocol_identity")
    if protocol_identity != PHASE1A_COMMODITY_BASELINE_VERSION:
        return

    violations: list[str] = []

    def values(name: str) -> list[object]:
        return list(np.asarray(metadata.get(name, []), dtype=object))

    cycles = values("timing_perturbation_cycles")
    for value in cycles:
        try:
            if int(str(value)) != 0:
                violations.append(f"timing_perturbation_cycles={value}")
        except (TypeError, ValueError):
            violations.append(f"invalid timing_perturbation_cycles={value!r}")
    labels = values("timing_perturbation_label")
    if any(str(value) != "1" for value in labels):
        violations.append("non-default timing_perturbation_label")

    def is_true(value: object) -> bool:
        return (
            isinstance(value, (bool, np.bool_))
            and bool(value)
            or str(value).lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

    if any(is_true(value) for value in values("timing_perturbation_applied")):
        violations.append("timing_perturbation_applied=true")
    namespaces = values("calibration_namespace")
    if any(
        str(value) not in {"", "none", "not_calibration", "unavailable"} for value in namespaces
    ):
        violations.append("calibration namespace present")

    timing_contract = provenance.get("artificial_timing_perturbation")
    if isinstance(timing_contract, dict):
        contract_cycles = timing_contract.get("timing_perturbation_cycles", 0)
        try:
            contract_nonzero = int(str(contract_cycles)) != 0
        except (TypeError, ValueError):
            contract_nonzero = True
        if (
            is_true(timing_contract.get("allowed"))
            or is_true(timing_contract.get("applied"))
            or contract_nonzero
        ):
            violations.append("manifest artificial-timing contract is not physical-zero")
    if violations:
        raise IntegrityError(
            f"physical commodity dataset {source} contains calibration contamination: "
            + ", ".join(sorted(set(violations)))
        )


def _validate_allocation_provenance(
    metadata: dict[str, np.ndarray], manifest: dict[str, Any], source: Path
) -> None:
    """Reject a source dataset that blurs one session across allocations."""

    session_field = (
        "acquisition_session_id" if "acquisition_session_id" in metadata else "session_id"
    )
    if session_field not in metadata or "allocation_id" not in metadata:
        return
    session_values = np.asarray(metadata[session_field]).astype(str)
    allocation_values = np.asarray(metadata["allocation_id"]).astype(str)
    allocations: dict[str, set[str]] = {}
    for session, allocation in zip(session_values, allocation_values, strict=True):
        allocations.setdefault(str(session), set()).add(str(allocation))
    blurred = {session: values for session, values in allocations.items() if len(values) != 1}
    if blurred:
        raise IntegrityError(
            f"dataset {source} has session identities spanning multiple allocation IDs: {blurred}"
        )
    ledgers = manifest.get("acquisition_sessions", [])
    expected: dict[str, str] = {}
    for ledger in ledgers:
        if not isinstance(ledger, dict):
            continue
        status = ledger.get("status")
        if status in {"active", "interrupted", "incomplete"}:
            raise IntegrityError(
                f"dataset {source} references an incomplete acquisition session ({status})"
            )
        session = str(ledger.get("acquisition_session_id", ledger.get("session_id", "")))
        allocation = str(ledger.get("controlled_memory_region", {}).get("allocation_id", ""))
        if session and allocation and allocation not in {"unavailable", "unknown"}:
            expected[session] = allocation
    mismatched = {
        session: (next(iter(values)), expected[session])
        for session, values in allocations.items()
        if session in expected and next(iter(values)) != expected[session]
    }
    if mismatched:
        raise IntegrityError(
            f"dataset {source} allocation metadata disagrees with its session ledger: {mismatched}"
        )


def trace_features(traces: np.ndarray, *, bins: int = 32) -> np.ndarray:
    """Create fixed, transparent summary features without fitting on test data."""

    if traces.ndim != 2:
        raise SchemaError("traces must be [samples, time]")
    if bins < 1 or bins > traces.shape[1]:
        raise SchemaError("feature bins must fit within trace length")
    chunks = np.array_split(traces.astype(np.float32, copy=False), bins, axis=1)
    features = [
        traces.mean(axis=1),
        traces.std(axis=1),
        traces.min(axis=1),
        traces.max(axis=1),
        np.sqrt(np.mean(np.square(traces), axis=1)),
    ]
    features.extend(chunk.mean(axis=1) for chunk in chunks)
    features.extend(chunk.std(axis=1) for chunk in chunks)
    return np.column_stack(features).astype(np.float32, copy=False)


def build_feature_matrix(
    traces: np.ndarray,
    metadata: dict[str, np.ndarray],
    *,
    feature_fields: Iterable[str] = (),
    policy: FeaturePolicy | None = None,
    allow_identity: bool = False,
) -> np.ndarray:
    fields = list(feature_fields)
    (policy or FeaturePolicy()).validate(fields, allow_identity=allow_identity)
    matrix = [trace_features(traces)]
    for field in fields:
        if field not in metadata:
            raise SchemaError(f"requested feature field not present: {field}")
        values = np.asarray(metadata[field])
        if values.dtype.kind in "iufb":
            encoded = values.astype(np.float32)
        elif allow_identity and field in (policy or FeaturePolicy()).prohibited:
            encoded = np.asarray(
                [int(sha256_text(str(value))[:8], 16) / 2**32 for value in values],
                dtype=np.float32,
            )
        else:
            raise SchemaError(f"feature field {field} is not numeric")
        matrix.append(encoded.reshape(-1, 1))
    return np.concatenate(matrix, axis=1)


def ensure_sample_ids(metadata: dict[str, np.ndarray]) -> list[str]:
    if "sample_id" not in metadata:
        raise SchemaError("dataset has no sample_id metadata")
    return [str(value) for value in metadata["sample_id"]]


def fingerprint_split(split: dict[str, Any]) -> str:
    material = dict(split)
    material.pop("split_fingerprint", None)
    return sha256_json(material)


def combine_datasets(
    source_dirs: Iterable[str | Path],
    target_dir: str | Path,
    *,
    config: dict[str, Any],
    condition: str,
    campaign_id: str,
    source_manifest_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Combine finalized session datasets without discarding source provenance.

    Source session manifests remain embedded in ``source_manifests.json`` and
    referenced from the combined manifest. Existing finalized rows are keyed
    by globally unique sample_id, so a crash/restart during combination can
    safely resume without duplicating samples.
    """

    from .storage import ShardWriter, validate_all_shards

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    sources = [Path(path) for path in source_dirs]
    if not sources:
        raise IntegrityError("cannot combine an empty source dataset list")
    loaded = [load_dataset(path) for path in sources]
    source_manifests = [item[4] for item in loaded]
    source_manifest_paths = source_manifest_paths or [
        str(path / "dataset.json") for path in sources
    ]
    source_purposes = {
        str(manifest.get("dataset_purpose", "generic")) for manifest in source_manifests
    }
    if len(source_purposes) != 1:
        raise IntegrityError("cannot combine source datasets with different dataset purposes")
    combined_purpose = next(iter(source_purposes))
    if combined_purpose == "physical_phase1a":
        # Re-open through the strict boundary before any rows are copied.
        loaded = [load_dataset(path, expected_purpose="physical_phase1a") for path in sources]
        source_manifests = [item[4] for item in loaded]
    source_protocols = {
        (manifest.get("protocol_identity"), manifest.get("protocol_hash"))
        for manifest in source_manifests
    }
    if len(source_protocols) != 1:
        raise IntegrityError(
            "cannot combine source datasets with different protocol identities/hashes"
        )
    combined_protocol_identity, combined_protocol_hash = next(iter(source_protocols))
    if combined_purpose == "physical_phase1a" and (
        combined_protocol_identity != PHASE1A_COMMODITY_BASELINE_VERSION
        or not isinstance(combined_protocol_hash, str)
        or not combined_protocol_hash
    ):
        raise IntegrityError(
            "physical Phase 1A source datasets require explicit protocol identity/hash"
        )

    existing_ids: set[str] = set()
    seen_session_ids: dict[str, Path] = {}
    for source, manifest in zip(sources, source_manifests, strict=True):
        for ledger in manifest.get("acquisition_sessions", []):
            if not isinstance(ledger, dict):
                continue
            session_id = str(ledger.get("acquisition_session_id", ledger.get("session_id", "")))
            if not session_id:
                continue
            if session_id in seen_session_ids and seen_session_ids[session_id] != source:
                raise IntegrityError(
                    "one acquisition_session_id appears in multiple source datasets; "
                    "combine only independently materialized sessions"
                )
            seen_session_ids[session_id] = source
    if sorted(target.glob("shard-*.npz")):
        _existing_traces, _existing_labels, existing_metadata, _existing_shards = load_shards(
            target
        )
        existing_ids.update(ensure_sample_ids(existing_metadata))
    row_refs: list[tuple[str, int, int]] = []
    source_ids: set[str] = set()
    for dataset_index, (_traces, _labels, metadata, _shards, _manifest) in enumerate(loaded):
        ids = ensure_sample_ids(metadata)
        for row_index, sample_id in enumerate(ids):
            if sample_id in source_ids:
                raise IntegrityError(
                    f"sample_id {sample_id!r} appears more than once across source datasets"
                )
            source_ids.add(sample_id)
            row_refs.append((sample_id, dataset_index, row_index))
    # Source session directories are not necessarily ordered by sample_id (the
    # IDs include independently generated session UUIDs). Canonicalize the
    # merged stream before writing so validate_all_shards can enforce its
    # monotonic shard-boundary invariant without weakening that invariant.
    row_refs.sort(key=lambda item: item[0])

    writer = ShardWriter(
        target,
        shard_target_mb=float(config.get("acquisition", {}).get("shard_target_mb", 512)),
        max_samples_per_shard=config.get("acquisition", {}).get("max_samples_per_shard"),
    )
    for sample_id, dataset_index, row_index in row_refs:
        if sample_id in existing_ids:
            continue
        traces, labels, metadata, _shards, _manifest = loaded[dataset_index]
        row_metadata = {key: values[row_index] for key, values in metadata.items()}
        writer.add(traces[row_index], int(labels[row_index]), row_metadata)
        existing_ids.add(sample_id)
    writer.finalize()
    shards = validate_all_shards(target)
    _combined_traces, combined_labels, _combined_metadata, _combined_shards = load_shards(target)
    sessions: list[dict[str, Any]] = []
    for manifest in source_manifests:
        sessions.extend(manifest.get("acquisition_sessions", []))
    combined = write_dataset_manifest(
        target,
        config=config,
        condition=condition,
        shard_infos=shards,
        label_stream_fingerprint=sha256_bytes(combined_labels.tobytes()),
        class_balance={
            "0": int(np.sum(combined_labels == 0)),
            "1": int(np.sum(combined_labels == 1)),
        },
        provenance={
            "campaign_combination": "finalized source session datasets combined by globally unique sample_id",
            "source_manifest_paths": source_manifest_paths,
            "source_dataset_fingerprints": [
                manifest.get("dataset_fingerprint", "unavailable") for manifest in source_manifests
            ],
            **(
                {
                    "protocol_identity": combined_protocol_identity,
                    "protocol_hash": combined_protocol_hash,
                    "artificial_timing_perturbation": {
                        "allowed": False,
                        "timing_perturbation_cycles": 0,
                        "timing_perturbation_label": 1,
                        "label_correlated": False,
                        "applied": False,
                        "calibration_namespace": "forbidden",
                        "physical_phase1a_forbidden": True,
                    },
                }
                if combined_purpose == "physical_phase1a"
                else {}
            ),
        },
        acquisition_sessions=sessions,
        campaign_id=campaign_id,
        dataset_purpose=combined_purpose,
        protocol_identity=(str(combined_protocol_identity) if combined_protocol_identity else None),
        protocol_hash=(str(combined_protocol_hash) if combined_protocol_hash else None),
    )
    (target / "source-manifests.json").write_text(
        json.dumps(
            {
                "schema": "sensetrace.campaign-source-manifests.v1",
                "campaign_id": campaign_id,
                "source_manifest_paths": source_manifest_paths,
                "source_manifests": source_manifests,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return combined

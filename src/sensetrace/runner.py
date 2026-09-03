"""Unattended runner and resumable acquisition state machine."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .acquisition.base import AcquisitionBackend
from .acquisition.commodity import CommodityDramBackend
from .acquisition.controlled import SyntheticMockControlledBackend
from .acquisition.synthetic import SyntheticBackend
from .config import config_fingerprint
from .datasets import write_dataset_manifest
from .hashing import sha256_json
from .inventory import collect_inventory
from .journal import Journal
from .safety import storage_status
from .storage import (
    ShardWriter,
    quarantine_invalid_shards,
    quarantine_temporary_shards,
    validate_all_shards,
)


def _git_commit() -> str:
    declared = os.environ.get("SENSETRACE_COMMIT")
    if declared:
        return declared
    source_root = Path(__file__).resolve().parents[2]
    deployed_marker = source_root / ".sensetrace-commit"
    if deployed_marker.is_file():
        value = deployed_marker.read_text(encoding="utf-8").strip()
        if value:
            return value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def new_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class AcquisitionRunner:
    def __init__(self, config: dict[str, Any], run_dir: str | Path, *, run_id: str | None = None):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.journal = Journal(self.run_dir / "events.jsonl")
        self._stop_requested = False

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self._stop_requested = True
        self.journal.append("termination_requested", signal=signum)

    def _backend(
        self, *, start_index: int = 0, condition: str = "null", amplitude: float | None = None
    ) -> AcquisitionBackend:
        data = self.config.get("data", {})
        controls = self.config.get("controls", {}).get("injected_weak_signal", {})
        acquisition = self.config.get("acquisition", {})
        backend_name = str(acquisition.get("backend", "synthetic"))
        if backend_name == "commodity":
            physical = self.config.get("phase1a", {})
            if physical.get("campaign_intent", "historical_reproduction") == "current_scaling":
                raise RuntimeError(
                    "current commodity scaling is closed by the frozen C_primitive_unsuitable gate"
                )
            return CommodityDramBackend(
                count=int(physical.get("samples", min(int(data.get("samples", 128)), 256))),
                trace_length=int(
                    physical.get("trace_length", min(int(data.get("trace_length", 32)), 64))
                ),
                seed=int(self.config.get("experiment", {}).get("seed", 1337)),
                pattern=str(physical.get("pattern", "single_bit")),
                target_bit=int(physical.get("target_bit", 0)),
                word_count=int(physical.get("word_count", 1024)),
                lock_memory=bool(physical.get("lock_memory", True)),
                cache_control=str(physical.get("cache_control", "eviction_buffer")),
                operation=str(physical.get("operation", "memory_read")),
                eviction_bytes=int(physical.get("eviction_bytes", 4 * 1024 * 1024)),
                cpu_affinity=physical.get("cpu_affinity"),
            )
        if backend_name == "controlled_mock":
            mock = self.config.get("phase2", {}).get("controlled_mock", {})
            if not isinstance(mock, dict):
                raise RuntimeError("phase2.controlled_mock configuration is not a mapping")
            count = int(mock.get("count", data.get("samples", 64)))
            trace_length = int(mock.get("trace_length", data.get("trace_length", 32)))
            controller_config_hash = sha256_json(mock)
            return SyntheticMockControlledBackend(
                count=count,
                trace_length=trace_length,
                seed=int(mock.get("seed", self.config.get("experiment", {}).get("seed", 1337))),
                target_id=str(mock.get("target_id", "mock-controlled-target-0000")),
                firmware_id=str(mock.get("firmware_id", "mock-controller-firmware-v0")),
                controller_config_hash=controller_config_hash,
            )
        if backend_name == "controlled_hardware":
            raise RuntimeError(
                "controlled_hardware is an explicit future adapter boundary; no real controller "
                "adapter is registered"
            )
        return SyntheticBackend(
            count=int(data.get("samples", 1000)),
            trace_length=int(data.get("trace_length", 128)),
            seed=int(self.config.get("experiment", {}).get("seed", 1337)),
            condition=condition,
            amplitude_sigma=float(
                amplitude if amplitude is not None else controls.get("amplitude_sigma", 0.1)
            ),
            start_index=int(
                controls.get("start_index", max(0, int(data.get("trace_length", 128)) // 3))
            ),
            width=int(controls.get("width", max(4, int(data.get("trace_length", 128)) // 16))),
            session_count=int(acquisition.get("session_count", 4)),
            device_count=int(acquisition.get("device_count", 2)),
            permute_seed=int(self.config.get("experiment", {}).get("seed", 1337)) + 7919,
        )

    def _resume_index(self) -> tuple[int, list[str]]:
        quarantined_paths = quarantine_temporary_shards(self.run_dir)
        quarantined_paths.extend(quarantine_invalid_shards(self.run_dir))
        infos = validate_all_shards(self.run_dir)
        next_index = 0
        if infos:
            last = infos[-1].last_sample_id
            try:
                next_index = int(last.rsplit("-", 1)[1]) + 1
            except (IndexError, ValueError):
                next_index = sum(info.rows for info in infos)
        return next_index, [path.name for path in quarantined_paths]

    def run(
        self,
        *,
        condition: str = "null",
        amplitude: float | None = None,
        stop_after: int | None = None,
    ) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        config_hash = config_fingerprint(self.config)
        if not (self.run_dir / "run.json").exists():
            _write_json(
                self.run_dir / "run.json",
                {
                    "schema": "sensetrace.run.v1",
                    "run_id": self.run_id,
                    "status": "active",
                    "started_at": datetime.now(UTC).isoformat(),
                    "code_commit": _git_commit(),
                    "configuration_hash": config_hash,
                    "host_id": collect_inventory().get("host_id", "unavailable"),
                },
            )
            (self.run_dir / "host.json").write_text(
                json.dumps(collect_inventory(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        (self.run_dir / "config.json").write_text(
            json.dumps(self.config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        journal_state = self.journal.recover()
        start_index, quarantined = self._resume_index()
        backend = self._backend(condition=condition, amplitude=amplitude)
        if start_index and not backend.recovery_policy.allow_resume:
            self.journal.append(
                "recovery_refused",
                run_id=self.run_id,
                resume_from_sample=start_index,
                backend=backend.name,
                deterministic_replay=backend.recovery_policy.deterministic_replay,
                continuity_requirement=backend.recovery_policy.continuity_requirement,
                decision="fail_closed; start a new acquisition session/run with a fresh identity",
            )
            backend.close()
            raise RuntimeError(
                f"{backend.name} acquisition cannot resume finalized shards in place"
            )
        self.journal.append(
            "recovery_started",
            run_id=self.run_id,
            session_id=f"session-{int(time.time())}",
            boot_id=collect_inventory().get("boot_id", "unavailable"),
            resume_from_sample=start_index,
            quarantined_temporary_shards=quarantined,
            prior_event_count=len(journal_state.events),
        )
        self.journal.append("acquisition_started", condition=condition, start_index=start_index)
        old_handlers = {
            signal.SIGTERM: signal.signal(signal.SIGTERM, self._handle_signal),
            signal.SIGINT: signal.signal(signal.SIGINT, self._handle_signal),
        }
        writer = ShardWriter(
            self.run_dir,
            shard_target_mb=float(self.config.get("acquisition", {}).get("shard_target_mb", 512)),
            max_samples_per_shard=self.config.get("acquisition", {}).get("max_samples_per_shard"),
        )
        acquired = start_index
        stopped_by_limit = False
        try:
            for sample in backend.samples(start_index=start_index):
                if self._stop_requested:
                    break
                if stop_after is not None and acquired - start_index >= stop_after:
                    stopped_by_limit = True
                    break
                safety = self.config.get("safety", {})
                disk = storage_status(
                    self.run_dir,
                    minimum_free_gb=float(safety.get("minimum_free_gb", 0)),
                    minimum_free_percent=float(safety.get("minimum_free_percent", 0)),
                )
                if not disk["safe"]:
                    self.journal.append("storage_guard_stop", storage=disk)
                    stopped_by_limit = True
                    break
                info = writer.add(sample.trace, sample.label, sample.metadata)
                acquired += 1
                if info:
                    self.journal.append(
                        "shard_finalized", **info.as_dict(), next_sample_index=acquired
                    )
            info = writer.finalize()
            if info:
                self.journal.append("shard_finalized", **info.as_dict(), next_sample_index=acquired)
            interrupted = self._stop_requested or stopped_by_limit
            event = "interrupted" if interrupted else "acquisition_completed"
            self.journal.append(
                event,
                next_sample_index=acquired,
                reason="signal"
                if self._stop_requested
                else "test_limit"
                if stopped_by_limit
                else "completed",
            )
            run_path = self.run_dir / "run.json"
            run_record = json.loads(run_path.read_text(encoding="utf-8"))
            run_record.update(
                {
                    "status": "interrupted" if interrupted else "completed",
                    "last_sample_index": acquired - 1,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            _write_json(run_path, run_record)
            if not interrupted:
                finalized = validate_all_shards(self.run_dir)
                provenance_factory = getattr(backend, "manifest_provenance", None)
                manifest_provenance = (
                    provenance_factory(condition=condition)
                    if provenance_factory is not None
                    else None
                )
                session_factory = getattr(backend, "session_provenance", None)
                sessions = (
                    [session_factory(status="completed")] if session_factory is not None else None
                )
                manifest = write_dataset_manifest(
                    self.run_dir,
                    config=self.config,
                    condition=condition,
                    shard_infos=finalized,
                    label_stream_fingerprint=str(
                        getattr(backend, "label_stream_fingerprint", "unavailable")
                    ),
                    provenance=manifest_provenance,
                    acquisition_sessions=sessions,
                    dataset_purpose=(
                        manifest_provenance.get("dataset_purpose")
                        if isinstance(manifest_provenance, dict)
                        else None
                    ),
                    protocol_identity=(
                        manifest_provenance.get("protocol_identity")
                        if isinstance(manifest_provenance, dict)
                        else None
                    ),
                    protocol_hash=(
                        manifest_provenance.get("protocol_hash")
                        if isinstance(manifest_provenance, dict)
                        else None
                    ),
                )
                self.journal.append(
                    "dataset_manifest_written",
                    dataset_fingerprint=manifest["dataset_fingerprint"],
                    dataset_purpose=manifest.get("dataset_purpose"),
                )
        finally:
            close = getattr(backend, "close", None)
            if close is not None:
                close()
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "condition": condition,
            "start_index": start_index,
            "next_sample_index": acquired,
            "status": "interrupted" if self._stop_requested or stopped_by_limit else "completed",
            "quarantined_temporary_shards": quarantined,
        }


def daemon(config_path: str | Path, *, heartbeat_seconds: float = 30.0) -> None:
    """Keep a service alive and record an explicit idle state until stopped."""

    from .config import load_config

    config = load_config(config_path)
    state_dir = Path(config.get("runner", {}).get("state_dir", "/var/lib/sensetrace/state"))
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        state_dir = Path.home() / ".local" / "share" / "sensetrace" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
    journal = Journal(state_dir / "service-events.jsonl")
    journal.append(
        "service_started",
        boot_id=collect_inventory().get("boot_id", "unavailable"),
        mode=config.get("runner", {}).get("mode", "idle"),
    )
    while True:
        journal.append("service_heartbeat", pid=os.getpid())
        time.sleep(heartbeat_seconds)

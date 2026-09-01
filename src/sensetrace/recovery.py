"""Deterministic local recovery exercise used by integration and host smoke tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .config import validate_config
from .runner import AcquisitionRunner


def recovery_test(output: str | Path | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sensetrace-recovery-") as temporary:
        root = Path(output) if output else Path(temporary) / "run"
        config = validate_config(
            {
                "experiment": {"name": "recovery-test", "seed": 404},
                "data": {"samples": 32, "trace_length": 64, "target_balance": 0.5},
                "splits": {
                    "primary": {
                        "group_keys": ["session_id", "device_id", "row_id", "cell_or_offset_id"],
                        "train_fraction": 0.7,
                        "validation_fraction": 0.15,
                        "test_fraction": 0.15,
                    }
                },
                "acquisition": {"shard_target_mb": 1, "max_samples_per_shard": 8},
                "feature_policy": {"prohibit_identity_features": True},
                "training": {"seeds": [11]},
            }
        )
        first = AcquisitionRunner(config, root, run_id=root.name).run(stop_after=13)
        temporary_shard = root / "shard-recovery.npz.tmp"
        temporary_shard.write_bytes(b"intentionally incomplete")
        second = AcquisitionRunner(config, root, run_id=root.name).run()
        events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        final_indices = [
            event.get("next_sample_index")
            for event in events
            if event.get("event") == "shard_finalized"
        ]
        result = {
            "passed": (
                first["status"] == "interrupted"
                and second["status"] == "completed"
                and second["next_sample_index"] == 32
                and bool(second["quarantined_temporary_shards"])
                and len(final_indices) == len(set(final_indices))
            ),
            "first_run": first,
            "second_run": second,
            "finalized_next_sample_indices": final_indices,
            "journal": str(root / "events.jsonl"),
        }
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            (Path(output) / "recovery-test.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

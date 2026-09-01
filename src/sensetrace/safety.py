"""Safety guards for bounded unattended acquisition."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def storage_status(
    path: str | Path, *, minimum_free_gb: float = 20, minimum_free_percent: float = 10
) -> dict[str, Any]:
    usage = shutil.disk_usage(Path(path))
    free_gb = usage.free / (1024**3)
    free_percent = 100.0 * usage.free / usage.total if usage.total else 0.0
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "free_gb": round(free_gb, 3),
        "free_percent": round(free_percent, 3),
        "minimum_free_gb": minimum_free_gb,
        "minimum_free_percent": minimum_free_percent,
        "safe": free_gb >= minimum_free_gb and free_percent >= minimum_free_percent,
    }

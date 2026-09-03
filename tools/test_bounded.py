#!/usr/bin/env python3
"""Run every pytest module in a fresh process with bounded numerical threads."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _peak_rss(log: Path) -> str:
    match = re.findall(r"Maximum resident set size \(kbytes\): (\d+)", log.read_text())
    return f"{int(match[-1]) / 1024:.1f} MiB" if match else "unavailable"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    test_files = sorted((root / "tests").glob("test_*.py"))
    if not test_files:
        print("no test modules found", file=sys.stderr)
        return 2
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    time_binary = shutil.which("/usr/bin/time")
    with tempfile.TemporaryDirectory(prefix="sensetrace-bounded-tests-") as log_dir_name:
        log_dir = Path(log_dir_name)
        for index, test_file in enumerate(test_files, start=1):
            log = log_dir / f"{index:03d}-{test_file.stem}.log"
            command = [sys.executable, "-m", "pytest", "-q", str(test_file)]
            if time_binary:
                command = [time_binary, "-v", *command]
            print(f"[{index}/{len(test_files)}] {test_file.relative_to(root)}", flush=True)
            try:
                with log.open("w", encoding="utf-8") as handle:
                    result = subprocess.run(
                        command,
                        cwd=root,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=300,
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                print(f"  failed: timeout; peak RSS {_peak_rss(log)}", file=sys.stderr)
                return 1
            print(f"  exit={result.returncode}; peak RSS {_peak_rss(log)}")
            if result.returncode != 0:
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                print("--- failing module tail ---", file=sys.stderr)
                print("\n".join(lines[-40:]), file=sys.stderr)
                return result.returncode or 1
    print(f"completed {len(test_files)} test modules in isolated processes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Crash-resilient append-only experiment journal."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import JournalCorruptionError


@dataclass(frozen=True)
class JournalRead:
    events: list[dict[str, Any]]
    trailing_partial: bool = False
    trailing_partial_bytes: int = 0
    trailing_partial_sha256: str | None = None
    previous_file_size: int = 0
    valid_bytes: int = 0


class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {"ts": time.time(), "event": event, **payload}
        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def read(self) -> JournalRead:
        if not self.path.exists():
            return JournalRead([])
        events: list[dict[str, Any]] = []
        with self.path.open("rb") as handle:
            lines = handle.readlines()
            file_size = handle.tell()
        valid_bytes = 0
        for index, line in enumerate(lines):
            if not line.strip():
                valid_bytes += len(line)
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                is_last = index == len(lines) - 1
                if is_last and not line.endswith(b"\n"):
                    return JournalRead(
                        events,
                        trailing_partial=True,
                        trailing_partial_bytes=len(line),
                        trailing_partial_sha256=hashlib.sha256(line).hexdigest(),
                        previous_file_size=file_size,
                        valid_bytes=valid_bytes,
                    )
                raise JournalCorruptionError(
                    f"malformed journal record at line {index + 1}"
                ) from exc
            if not isinstance(record, dict) or not isinstance(record.get("event"), str):
                raise JournalCorruptionError(f"invalid journal record at line {index + 1}")
            # A complete JSON object without its terminating newline is not an
            # appendable record: a later append would concatenate two objects.
            # Treat it as a recoverable tail and preserve its hash in the event
            # emitted by recover().
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                return JournalRead(
                    events,
                    trailing_partial=True,
                    trailing_partial_bytes=len(line),
                    trailing_partial_sha256=hashlib.sha256(line).hexdigest(),
                    previous_file_size=file_size,
                    valid_bytes=valid_bytes,
                )
            events.append(record)
            valid_bytes += len(line)
        return JournalRead(events, previous_file_size=file_size, valid_bytes=valid_bytes)

    def recover(self) -> JournalRead:
        result = self.read()
        if result.trailing_partial:
            # Repair the file before appending the recovery record.  Keeping the
            # invalid bytes in-place would make every future append ambiguous.
            with self.path.open("r+b") as handle:
                handle.truncate(result.valid_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            recovered_size = self.path.stat().st_size
            self.append(
                "journal_recovery",
                action="truncate_trailing_partial_record",
                discarded_byte_count=result.trailing_partial_bytes,
                discarded_bytes_sha256=result.trailing_partial_sha256,
                previous_file_size=result.previous_file_size,
                recovered_file_size=recovered_size,
                recovery_timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            return self.read()
        return result

    @staticmethod
    def last_event(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        for event in reversed(events):
            if event.get("event") == name:
                return event
        return None

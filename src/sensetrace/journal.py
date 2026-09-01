"""Crash-resilient append-only experiment journal."""

from __future__ import annotations

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
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                is_last = index == len(lines) - 1
                if is_last and not line.endswith(b"\n"):
                    return JournalRead(events, trailing_partial=True)
                raise JournalCorruptionError(
                    f"malformed journal record at line {index + 1}"
                ) from exc
            if not isinstance(record, dict) or not isinstance(record.get("event"), str):
                raise JournalCorruptionError(f"invalid journal record at line {index + 1}")
            events.append(record)
        return JournalRead(events)

    def recover(self) -> JournalRead:
        result = self.read()
        if result.trailing_partial:
            self.append("journal_recovery", action="ignored_trailing_partial_record")
        return result

    @staticmethod
    def last_event(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        for event in reversed(events):
            if event.get("event") == name:
                return event
        return None

# ADR-003: Fabric controller, non-root service, conservative recovery

## Decision

The controller uses one Fabric-backed `RemoteHost` client for CLI and Fabric tasks. The system unit runs as `worker-03` with narrow systemd filesystem permissions and `Restart=on-failure`; a user unit is the unprivileged fallback. The installer enables kernel panic/oops reboot settings only when explicitly run as root. Watchdog devices are inventoried but not enabled without provider identification and a verified timeout/recovery test.

## Rationale

Fabric provides programmable SSH orchestration while preserving direct SSH diagnostics. A dedicated service-account boundary reduces accidental privilege. Process crashes, kernel failures, hard hangs, and power loss are separate failure classes and must not be conflated in evidence.

## Consequence

Without privileged sudo, a user unit can be exercised but cannot prove boot persistence, sysctl application, or reboot recovery. The CLI reports that boundary instead of claiming appliance behavior.

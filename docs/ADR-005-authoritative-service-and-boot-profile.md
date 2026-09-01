# ADR-005: Authoritative runner service and reversible boot profiles

## Decision

SenseTrace has one authoritative runner scope. A system unit is authoritative
when installed; otherwise the user unit is an explicit fallback. The controller
reports `system`, `user-fallback`, `invalid-dual-service`, or `not-installed`,
refuses service actions in the dual state, and never silently tries the other
scope. A privileged migration stops and disables the user fallback before
installing/enabling the system unit and records a migration event.

The live operator configuration is distinct from the example configuration.
Deployment initializes a missing configuration and preserves an existing one;
replacement requires an explicit reset request with before/after hashes.

The repository provides a candidate `sensetrace.target` that requires the
verified multi-user, network, SSH, and SenseTrace dependencies. It is installed
without becoming default. The default-target transition and target isolation
are separate, reversible, operator-authorized operations after fresh-SSH
validation.

## Consequence

The current worker remains a graphical, user-service fallback appliance until
privileged authorization is available and the documented isolation/reboot
acceptance sequence passes.

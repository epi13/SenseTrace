# worker-03 appliance readiness — 2026-09-01

## Observed fact

The controller re-inspected the live host before making changes.

- SSH and Fabric connectivity: working.
- Host: `worker-03`; one runner PID (`186479`) under the user fallback.
- Authoritative management mode: `user-fallback`.
- System service: not installed/enabled/active.
- User service: enabled and active.
- Default target: `graphical.target`.
- Active display manager: SDDM.
- Running services: 39.
- CPU governor: `powersave` on all eight CPUs; `intel_pstate`; turbo active;
  `turbostat` available.
- Watchdogs: `intel_oc_wdt` (60 s) and `iTCO_wdt` (30 s) both loaded and
  inactive; both report `nowayout=0`. Systemd runtime watchdog is `0`; reboot
  watchdog is `10min`.
- `sudo -n true`: not authorized; an interactive sudo password was not
  available to the controller session.

## Interpretation

The live state is still the safe pre-transition state recorded after PR #4.
No headless, system-service, dedicated-target, reboot, or watchdog mutation was
performed. The controller now exposes an explicit terminal-only `authorize-sudo`
operation; it never stores or logs credentials. Once authorized, the staged
workflow is system service, fresh-SSH validation, multi-user baseline, target
isolation, and only then default-target/reboot acceptance.

The native deployment path builds the timing library when a compiler is
available, and Phase 1A refuses to run when its configured native requirement is
not met.

## Unresolved confounds

No multi-user baseline, fresh post-reboot SSH acceptance, system-service boot,
dedicated-target isolation, or hardware watchdog reset test has yet been
performed. Display-manager removal, service-count changes, CPU-profile changes,
and watchdog selection remain operator-authorized actions.

## Claim boundary

worker-03 is not yet a proven dedicated headless appliance. The current evidence
supports only the user-service fallback and live read-only inventory.
